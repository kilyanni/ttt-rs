//! Host-side state management for the streaming TTT kernel.
//!
//! This module provides:
//! - `TttD2dStreamingState` - manages a running persistent kernel
//! - Global registry for looking up streaming states by key
//! - `D2dStreamingConfig` - configuration including stream key

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use burn_backend::Shape;
use burn_cubecl::{
    CubeRuntime, FloatElement,
    ops::numeric::{empty_device, zeros_client},
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use thundercube::{
    prelude::{D4, D8, D16, D32, D64, LINE_SIZE},
    streaming::{AsyncStream, GpuPtr},
    util::wait_for_sync,
};
use tracing::trace;

use super::{
    super::super::{
        forward::{InputsLaunch, OutputsLaunch},
        helpers::Params,
    },
    kernel::{
        CTRL_ARRAY_SIZE, D2dStreamingKernelConfig, STATUS_DONE, STATUS_IDLE, STATUS_READY,
        STATUS_SHUTDOWN, fused_ttt_d2d_streaming_kernel,
    },
};
use crate::FusedTttConfig;

/// Key for the streaming state registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamKey {
    pub stream_id: u64,
}

impl StreamKey {
    pub fn new(stream_id: u64) -> Self {
        Self { stream_id }
    }
}

/// Configuration for the streaming kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct D2dStreamingConfig {
    /// Unique identifier for this streaming session
    pub stream_id: u64,
    /// Batch size
    pub batch_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Mini-batch sequence length
    pub mini_batch_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Layer norm epsilon (scaled)
    pub epsilon_scaled: u32,
    /// Number of threads per cube
    pub threads: usize,
    /// Enable debug output in kernel
    pub debug: bool,
}

impl D2dStreamingConfig {
    pub fn new(
        stream_id: u64,
        batch_size: usize,
        num_heads: usize,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            stream_id,
            batch_size,
            num_heads,
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / 1e-9) as u32,
            threads,
            debug: std::env::var("D2D_STREAM_DEBUG").is_ok(),
        }
    }

    /// Set debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * 1e-9
    }

    pub fn kernel_config(&self) -> D2dStreamingKernelConfig {
        D2dStreamingKernelConfig::new(
            FusedTttConfig::new(
                self.mini_batch_len,
                self.head_dim,
                self.epsilon(),
                self.threads,
            ),
            self.debug,
        )
    }

    pub fn key(&self) -> StreamKey {
        StreamKey::new(self.stream_id)
    }

    /// Total number of cubes (single cube iterates through all batch/head pairs)
    pub fn num_cubes(&self) -> usize {
        1
    }

    /// Size of control array
    pub fn ctrl_array_len(&self) -> usize {
        self.num_cubes() * CTRL_ARRAY_SIZE as usize
    }

    /// Shape for QKV buffers: [batch, heads, mini_batch_len, head_dim]
    pub fn qkv_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.num_heads,
            self.mini_batch_len,
            self.head_dim,
        ]
    }

    /// Number of f32 elements in a QKV buffer
    pub fn qkv_len(&self) -> usize {
        self.batch_size * self.num_heads * self.mini_batch_len * self.head_dim
    }

    /// Shape for ttt_lr_eta: [batch, heads, mini_batch_len]
    pub fn eta_shape(&self) -> [usize; 3] {
        [self.batch_size, self.num_heads, self.mini_batch_len]
    }

    /// Number of f32 elements in eta buffer
    pub fn eta_len(&self) -> usize {
        self.batch_size * self.num_heads * self.mini_batch_len
    }

    /// Shape for weight: [batch, heads, head_dim, head_dim]
    pub fn weight_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
        ]
    }

    /// Shape for bias: [batch, heads, head_dim]
    pub fn bias_shape(&self) -> [usize; 3] {
        [self.batch_size, self.num_heads, self.head_dim]
    }
}

/// GPU buffer tensors for the streaming kernel.
pub struct D2dStreamingBufferTensors<R: CubeRuntime> {
    // Input/output buffers (single mini-batch sized)
    pub xq: CubeTensor<R>,
    pub xk: CubeTensor<R>,
    pub xv: CubeTensor<R>,
    pub ttt_lr_eta: CubeTensor<R>,
    pub output: CubeTensor<R>,
    /// Separate output buffer for returning results without blocking on persistent kernel
    pub result_output: CubeTensor<R>,

    // Control array
    pub control: CubeTensor<R>,

    // Weight and bias (updated in-place by kernel)
    pub weight: CubeTensor<R>,
    pub bias: CubeTensor<R>,
    /// Copy of weight for returning results
    pub result_weight: CubeTensor<R>,
    /// Copy of bias for returning results
    pub result_bias: CubeTensor<R>,

    // Constant tensors
    pub token_eta: CubeTensor<R>,
    pub ln_weight: CubeTensor<R>,
    pub ln_bias: CubeTensor<R>,
}

use super::super::super::next_persistent_kernel_stream_id;

/// State for a running streaming TTT kernel.
///
/// Note: This struct is not Send because AsyncStream contains a raw HIP stream pointer.
/// It should only be used from the thread that created it.
pub struct TttD2dStreamingState<R: CubeRuntime> {
    pub config: D2dStreamingConfig,
    pub stream: AsyncStream,
    pub tensors: D2dStreamingBufferTensors<R>,
    /// Client for normal operations (D2D copies, reading results, etc.)
    pub client: ComputeClient<R>,
    /// Client with a separate stream for the persistent kernel.
    /// This prevents the persistent kernel from blocking normal operations.
    pub kernel_client: ComputeClient<R>,
    pub is_initialized: bool,
    // Cached GPU pointers - obtained BEFORE kernel launch to avoid
    // triggering cross-stream synchronization when accessing kernel-written buffers
    cached_xq_ptr: GpuPtr<'static, f32>,
    cached_xk_ptr: GpuPtr<'static, f32>,
    cached_xv_ptr: GpuPtr<'static, f32>,
    cached_eta_ptr: GpuPtr<'static, f32>,
    cached_ctrl_ptr: GpuPtr<'static, u32>,
    cached_output_ptr: GpuPtr<'static, f32>,
    cached_weight_ptr: GpuPtr<'static, f32>,
    cached_bias_ptr: GpuPtr<'static, f32>,
    cached_result_output_ptr: GpuPtr<'static, f32>,
    cached_result_weight_ptr: GpuPtr<'static, f32>,
    cached_result_bias_ptr: GpuPtr<'static, f32>,
}

// SAFETY: TttD2dStreamingState is only accessed from the same thread via the registry lock.
// The AsyncStream's raw pointer is only used for HIP API calls which are thread-safe
// when called from the thread that created the stream.
unsafe impl<R: CubeRuntime> Send for TttD2dStreamingState<R> {}

/// Type-erased wrapper for storing in the global registry.
/// Uses Box<dyn Any> pattern for type erasure.
struct AnyStreamingState(Box<dyn std::any::Any + Send>);

impl AnyStreamingState {
    fn new<R: CubeRuntime + 'static>(state: TttD2dStreamingState<R>) -> Self {
        Self(Box::new(state))
    }

    fn downcast_mut<R: CubeRuntime + 'static>(&mut self) -> Option<&mut TttD2dStreamingState<R>> {
        self.0.downcast_mut()
    }
}

/// Global registry of streaming states.
static STREAMING_REGISTRY: LazyLock<Mutex<HashMap<StreamKey, AnyStreamingState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get or create a streaming state from the global registry.
pub fn get_or_create_d2d_streaming_state<R: CubeRuntime + 'static, F: FloatElement>(
    config: D2dStreamingConfig,
    client: ComputeClient<R>,
    device: R::Device,
    initial_weight: CubeTensor<R>,
    initial_bias: CubeTensor<R>,
    token_eta: CubeTensor<R>,
    ln_weight: CubeTensor<R>,
    ln_bias: CubeTensor<R>,
) -> &'static mut TttD2dStreamingState<R> {
    let key = config.key();

    let mut registry = STREAMING_REGISTRY.lock().unwrap();

    if !registry.contains_key(&key) {
        let state = TttD2dStreamingState::new::<F>(
            config,
            client,
            device,
            initial_weight,
            initial_bias,
            token_eta,
            ln_weight,
            ln_bias,
        );
        registry.insert(key, AnyStreamingState::new(state));
    }

    // SAFETY: We hold the lock and the state exists
    let state = registry.get_mut(&key).unwrap();
    let state_ptr = state.downcast_mut::<R>().unwrap() as *mut TttD2dStreamingState<R>;

    // SAFETY: The registry is static and we're returning a reference that
    // will be used within the same forward_launch call
    unsafe { &mut *state_ptr }
}

/// Remove a streaming state from the registry.
pub fn remove_d2d_streaming_state<R: CubeRuntime + 'static>(
    stream_id: u64,
) -> Option<TttD2dStreamingState<R>> {
    let key = StreamKey::new(stream_id);
    let mut registry = STREAMING_REGISTRY.lock().unwrap();
    registry.remove(&key).and_then(|any| {
        // Try to downcast and extract
        match any.0.downcast::<TttD2dStreamingState<R>>() {
            Ok(boxed) => Some(*boxed),
            Err(_) => None,
        }
    })
}

/// Remove a streaming state by ID, triggering its Drop impl for cleanup.
/// Use this when you don't need the state back - just need to clean up.
pub fn remove_d2d_streaming_state_by_id(stream_id: u64) {
    let key = StreamKey::new(stream_id);
    let mut registry = STREAMING_REGISTRY.lock().unwrap();
    // Remove triggers AnyStreamingState drop -> TttD2dStreamingState::drop() -> signal_shutdown()
    registry.remove(&key);
}

/// Shutdown and remove a streaming state, returning final weight/bias.
pub fn shutdown_d2d_streaming_state<R: CubeRuntime + 'static>(
    stream_id: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    remove_d2d_streaming_state::<R>(stream_id).map(|state| state.shutdown())
}

/// Access a streaming state by ID for benchmarking.
#[allow(dead_code)]
pub fn with_d2d_streaming_state<R: CubeRuntime + 'static, T>(
    stream_id: u64,
    f: impl FnOnce(&TttD2dStreamingState<R>) -> T,
) -> T {
    let key = StreamKey::new(stream_id);
    let mut registry = STREAMING_REGISTRY.lock().unwrap();
    let state = registry
        .get_mut(&key)
        .expect("streaming state not found")
        .downcast_mut::<R>()
        .expect("type mismatch");
    f(state)
}

impl<R: CubeRuntime> TttD2dStreamingState<R> {
    /// Create a new streaming state and launch the persistent kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F: FloatElement>(
        config: D2dStreamingConfig,
        client: ComputeClient<R>,
        device: R::Device,
        initial_weight: CubeTensor<R>,
        initial_bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
    ) -> Self {
        // Create kernel_client FIRST so we can allocate control on its stream.
        // This ensures the kernel sees initialized zeros without cross-stream sync.
        let mut kernel_client = client.clone();
        let kernel_stream_id = next_persistent_kernel_stream_id();
        trace!("[HOST] using kernel stream ID: {}", kernel_stream_id);
        unsafe {
            kernel_client.set_stream(kernel_stream_id);
        }

        // Allocate streaming buffers (single mini-batch sized)
        eprintln!(
            "[HOST] Creating streaming state: batch={}, heads={}, seq={}, dim={}, qkv_shape={:?}",
            config.batch_size,
            config.num_heads,
            config.mini_batch_len,
            config.head_dim,
            config.qkv_shape()
        );
        let xq = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let xk = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let xv = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let ttt_lr_eta_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.eta_shape()),
        );
        let output = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );

        // Allocate control on kernel_client's stream with zeros - kernel will see this
        let ctrl_len = config.ctrl_array_len();
        let control = zeros_client::<R>(
            kernel_client.clone(),
            device.clone(),
            Shape::from([ctrl_len]),
            burn_backend::DType::U32,
        );

        // Allocate result buffers for returning data without blocking on persistent kernel
        let result_output = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let result_weight = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.weight_shape()),
        );
        let result_bias = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.bias_shape()),
        );

        // Allocate proper weight/bias buffers with full 4D shape
        // The initial_weight/bias might be broadcast views without actual memory for each batch
        let weight = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.weight_shape()),
        );
        let bias = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.bias_shape()),
        );

        let tensors = D2dStreamingBufferTensors {
            xq,
            xk,
            xv,
            ttt_lr_eta: ttt_lr_eta_buf,
            output,
            result_output,
            control,
            weight,
            bias,
            result_weight,
            result_bias,
            token_eta,
            ln_weight,
            ln_bias,
        };

        // Create async stream for memory transfers
        let stream = AsyncStream::new();
        eprintln!("[HOST] async stream created");

        // Get all cached pointers BEFORE launching the kernel.
        eprintln!("[HOST] getting cached pointers...");
        // This is critical: get_resource() triggers cross-stream synchronization,
        // so we must obtain all pointers before the persistent kernel starts.
        // After the kernel launches, accessing kernel-written buffers via get_resource()
        // would block forever waiting for the (never-ending) kernel to complete.
        let cached_xq_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xq.handle)) };
        let cached_xk_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xk.handle)) };
        let cached_xv_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xv.handle)) };
        let cached_eta_ptr: GpuPtr<'static, f32> = unsafe {
            std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.ttt_lr_eta.handle))
        };
        let cached_ctrl_ptr: GpuPtr<'static, u32> =
            unsafe { std::mem::transmute(stream.ptr::<u32, R>(&client, &tensors.control.handle)) };
        let cached_output_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.output.handle)) };
        let cached_weight_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.weight.handle)) };
        let cached_bias_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.bias.handle)) };
        let cached_result_output_ptr: GpuPtr<'static, f32> = unsafe {
            std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_output.handle))
        };
        let cached_result_weight_ptr: GpuPtr<'static, f32> = unsafe {
            std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_weight.handle))
        };
        let cached_result_bias_ptr: GpuPtr<'static, f32> = unsafe {
            std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_bias.handle))
        };
        eprintln!("[HOST] cached pointers obtained");

        let mut state = Self {
            config,
            stream,
            tensors,
            client: client.clone(),
            kernel_client,
            is_initialized: false,
            cached_xq_ptr,
            cached_xk_ptr,
            cached_xv_ptr,
            cached_eta_ptr,
            cached_ctrl_ptr,
            cached_output_ptr,
            cached_weight_ptr,
            cached_bias_ptr,
            cached_result_output_ptr,
            cached_result_weight_ptr,
            cached_result_bias_ptr,
        };

        // Initialize weight/bias buffers from initial values
        // The initial values might be broadcast views, so we replicate for each batch
        trace!("[HOST] initializing weight/bias from initial values...");
        let src_weight: GpuPtr<f32> = state.stream.ptr(&client, &initial_weight.handle);
        let src_bias: GpuPtr<f32> = state.stream.ptr(&client, &initial_bias.handle);
        // Use cached pointers for destination
        let dst_weight = state.cached_weight_ptr;
        let dst_bias = state.cached_bias_ptr;

        // Check if source is already the full size or needs replication
        let per_head_weight_size = config.head_dim * config.head_dim;
        let per_head_bias_size = config.head_dim;

        if src_weight.len() == config.num_heads * per_head_weight_size {
            // Source is [num_heads, head_dim, head_dim] - need to replicate for each batch
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_weight_size;
                state.stream.copy_d2d(
                    dst_weight,
                    batch_offset,
                    src_weight,
                    0,
                    config.num_heads * per_head_weight_size,
                );
            }
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_bias_size;
                state.stream.copy_d2d(
                    dst_bias,
                    batch_offset,
                    src_bias,
                    0,
                    config.num_heads * per_head_bias_size,
                );
            }
        } else {
            // Source already has batch dimension - just copy
            let full_weight_len = config.batch_size * config.num_heads * per_head_weight_size;
            let full_bias_len = config.batch_size * config.num_heads * per_head_bias_size;
            state.stream.copy_d2d(
                dst_weight,
                0,
                src_weight,
                0,
                full_weight_len.min(src_weight.len()),
            );
            state
                .stream
                .copy_d2d(dst_bias, 0, src_bias, 0, full_bias_len.min(src_bias.len()));
        }
        state.stream.sync();

        // Control is already initialized to zeros (IDLE) by zeros_client

        // Launch the persistent kernel
        eprintln!("[HOST] launching kernel...");
        state.launch_kernel::<F>();
        eprintln!("[HOST] kernel launched!");
        state.is_initialized = true;

        state
    }

    /// Launch the persistent streaming kernel.
    /// Launches single cube that iterates through all (batch, head) pairs internally.
    fn launch_kernel<F: FloatElement>(&self) {
        let kernel_config = self.config.kernel_config();
        let mini_batch_len = self.config.mini_batch_len;
        let head_dim = self.config.head_dim;
        let threads = self.config.threads;

        // Launch single cube - iterates through all (batch, head) pairs internally
        let cube_count = CubeCount::Static(1, 1, 1);
        let vectorization = LINE_SIZE;

        // Get handle refs with longer lifetime
        let xq_ref = self.tensors.xq.as_handle_ref();
        let xk_ref = self.tensors.xk.as_handle_ref();
        let xv_ref = self.tensors.xv.as_handle_ref();
        let weight_ref = self.tensors.weight.as_handle_ref();
        let bias_ref = self.tensors.bias.as_handle_ref();
        let token_eta_ref = self.tensors.token_eta.as_handle_ref();
        let ttt_lr_eta_ref = self.tensors.ttt_lr_eta.as_handle_ref();
        let ln_weight_ref = self.tensors.ln_weight.as_handle_ref();
        let ln_bias_ref = self.tensors.ln_bias.as_handle_ref();
        let output_ref = self.tensors.output.as_handle_ref();

        // Create InputsLaunch
        let inputs = InputsLaunch::<F, R>::new(
            xq_ref.as_tensor_arg(vectorization),
            xk_ref.as_tensor_arg(vectorization),
            xv_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
            token_eta_ref.as_tensor_arg(vectorization),
            ttt_lr_eta_ref.as_tensor_arg(vectorization),
            ln_weight_ref.as_tensor_arg(vectorization),
            ln_bias_ref.as_tensor_arg(vectorization),
        );

        // Create OutputsLaunch
        let outputs = OutputsLaunch::<F, R>::new(
            output_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
        );

        // Control tensor for Tensor<Atomic<u32>> kernel parameter
        let control_ref = self.tensors.control.as_handle_ref();
        let control_arg = control_ref.as_tensor_arg(1);

        // Dispatch based on tile configuration
        // Use kernel_client which has a separate stream for the persistent kernel
        tile_dispatch!(
            fused_ttt_d2d_streaming_kernel,
            &self.kernel_client,
            cube_count,
            mini_batch_len,
            head_dim,
            threads,
            inputs,
            outputs,
            control_arg,
            kernel_config
        );
    }

    /// Wait for all cubes to complete using two-phase polling.
    /// This prevents reading stale DONE values from previous iterations.
    fn wait_for_cubes_done(&self) {
        let num_cubes = self.config.num_cubes();
        let ctrl_ptr = self.cached_ctrl_ptr;

        for cube in 0..num_cubes {
            // First, wait for status to NOT be DONE (kernel started processing and cleared it)
            loop {
                let status = self.stream.read(ctrl_ptr, cube, 1);
                if status[0] != STATUS_DONE {
                    break;
                }
                std::hint::spin_loop();
            }

            // Now poll for DONE
            loop {
                let status = self.stream.read(ctrl_ptr, cube, 1);
                if status[0] == STATUS_DONE {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        trace!("[HOST] all cubes done, resetting to IDLE");
        // Sync and reset to IDLE
        self.stream.sync();
        let idle_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_IDLE).collect();
        self.stream.write(ctrl_ptr, 0, &idle_signals);
        self.stream.sync();
    }

    /// Feed a mini-batch to the kernel using D2D copies and wait for output.
    ///
    /// This uses GPU-to-GPU copies to transfer data from the input tensors
    /// to the streaming buffers, avoiding CPU round-trips.
    pub fn forward_d2d(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) -> &CubeTensor<R> {
        eprintln!("[HOST] forward_d2d start");
        // Get GPU pointers for source tensors (these are new each call, not kernel-written)
        eprintln!("[HOST] forward_d2d getting src_xq ptr...");
        let src_xq: GpuPtr<f32> = self.stream.ptr(&self.client, &xq.handle);
        eprintln!("[HOST] forward_d2d got src_xq ptr");
        let src_xk: GpuPtr<f32> = self.stream.ptr(&self.client, &xk.handle);
        let src_xv: GpuPtr<f32> = self.stream.ptr(&self.client, &xv.handle);
        let src_eta: GpuPtr<f32> = self.stream.ptr(&self.client, &ttt_lr_eta.handle);
        eprintln!("[HOST] forward_d2d got all src ptrs");

        // Use cached pointers for destination buffers (avoid get_resource on kernel-written buffers)
        let dst_xq = self.cached_xq_ptr;
        let dst_xk = self.cached_xk_ptr;
        let dst_xv = self.cached_xv_ptr;
        let dst_eta = self.cached_eta_ptr;
        let ctrl_ptr = self.cached_ctrl_ptr;

        eprintln!("[HOST] forward_d2d starting D2D copies...");
        // D2D copy input data to streaming buffers
        self.stream
            .copy_d2d(dst_xq, 0, src_xq, 0, self.config.qkv_len());
        self.stream
            .copy_d2d(dst_xk, 0, src_xk, 0, self.config.qkv_len());
        self.stream
            .copy_d2d(dst_xv, 0, src_xv, 0, self.config.qkv_len());
        self.stream
            .copy_d2d(dst_eta, 0, src_eta, 0, self.config.eta_len());
        eprintln!("[HOST] forward_d2d D2D copies issued");

        // Sync to ensure copies are complete before signaling kernel
        self.stream.sync();
        eprintln!("[HOST] forward_d2d D2D copies synced");

        // DEBUG: Verify D2D copy by reading back and comparing
        let src_data = self.stream.read(src_xq, 0, 8); // First 8 values (head 0)
        let dst_data = self.stream.read(dst_xq, 0, 8);
        eprintln!(
            "[HOST] D2D verify head0: src={:?}, dst={:?}",
            &src_data[..4],
            &dst_data[..4]
        );
        let src_data_h1 = self.stream.read(src_xq, 256, 8); // First 8 values of head 1
        let dst_data_h1 = self.stream.read(dst_xq, 256, 8);
        eprintln!(
            "[HOST] D2D verify head1: src={:?}, dst={:?}",
            &src_data_h1[..4],
            &dst_data_h1[..4]
        );

        trace!(
            "[HOST] D2D done, setting READY for {} cubes",
            self.config.num_cubes()
        );

        // Set READY status for all cubes
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();
        self.stream.write(ctrl_ptr, 0, &ready_signals);
        self.stream.sync();
        trace!("[HOST] READY set, polling DONE...");

        self.wait_for_cubes_done();

        // DEBUG: Check output after kernel
        let out_h0 = self.stream.read(self.cached_output_ptr, 0, 4);
        let out_h1 = self.stream.read(self.cached_output_ptr, 256, 4);
        eprintln!("[HOST] Output head0: {:?}", out_h0);
        eprintln!("[HOST] Output head1: {:?}", out_h1);

        // Copy results to separate buffers that can be read without blocking on persistent kernel
        // Use cached pointers to avoid get_resource() which would sync with kernel stream
        let output_ptr = self.cached_output_ptr;
        let result_output_ptr = self.cached_result_output_ptr;
        let weight_ptr = self.cached_weight_ptr;
        let result_weight_ptr = self.cached_result_weight_ptr;
        let bias_ptr = self.cached_bias_ptr;
        let result_bias_ptr = self.cached_result_bias_ptr;

        trace!(
            "[HOST] weight tensor shape: {:?}, ptr capacity: {}",
            &self.tensors.weight.shape,
            weight_ptr.len()
        );
        trace!(
            "[HOST] result_weight ptr capacity: {}",
            result_weight_ptr.len()
        );

        self.stream
            .copy_d2d(result_output_ptr, 0, output_ptr, 0, self.config.qkv_len());
        // Copy only what the source buffer actually has
        let weight_src_len = weight_ptr.len();
        let bias_src_len = bias_ptr.len();
        trace!(
            "[HOST] copying weight: {} elements, bias: {} elements",
            weight_src_len, bias_src_len
        );
        self.stream
            .copy_d2d(result_weight_ptr, 0, weight_ptr, 0, weight_src_len);
        self.stream
            .copy_d2d(result_bias_ptr, 0, bias_ptr, 0, bias_src_len);
        self.stream.sync();

        trace!("[HOST] forward_d2d complete, returning result_output");
        &self.tensors.result_output
    }

    /// Shutdown the kernel and return final weight/bias.
    pub fn shutdown(self) -> (Vec<f32>, Vec<f32>) {
        trace!(
            "[HOST] shutdown: signaling SHUTDOWN to {} cubes",
            self.config.num_cubes()
        );
        // Use cached pointers to avoid get_resource() which would sync with kernel stream
        let ctrl_ptr = self.cached_ctrl_ptr;
        let weight_ptr = self.cached_weight_ptr;
        let bias_ptr = self.cached_bias_ptr;

        // Signal SHUTDOWN to all cubes
        let num_cubes = self.config.num_cubes();
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(ctrl_ptr, 0, &shutdown_signals);
        self.stream.sync();

        // Wait for kernel to finish on the kernel_client's stream
        trace!("[HOST] shutdown: waiting for kernel to finish via wait_for_sync");
        wait_for_sync(&self.kernel_client).expect("sync failed");
        trace!("[HOST] shutdown: kernel finished");

        // Read final weight and bias
        let weight_len = self.config.batch_size
            * self.config.num_heads
            * self.config.head_dim
            * self.config.head_dim;
        let bias_len = self.config.batch_size * self.config.num_heads * self.config.head_dim;

        let weight = self.stream.read(weight_ptr, 0, weight_len);
        let bias = self.stream.read(bias_ptr, 0, bias_len);

        (weight, bias)
    }

    /// Benchmark pure kernel compute time.
    /// Assumes data is already in the buffers from a prior forward_d2d call.
    /// Runs `iterations` cycles of READY→DONE.
    /// Returns per-iteration time in microseconds.
    pub fn bench_compute(&self, warmup: usize, iterations: usize) -> f64 {
        let ctrl_ptr = self.cached_ctrl_ptr;
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();

        // Warmup
        for _ in 0..warmup {
            self.stream.write(ctrl_ptr, 0, &ready_signals);
            self.stream.sync();
            self.wait_for_cubes_done();
        }

        // Timed iterations
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            self.stream.write(ctrl_ptr, 0, &ready_signals);
            self.stream.sync();
            self.wait_for_cubes_done();
        }
        let elapsed = start.elapsed();

        elapsed.as_micros() as f64 / iterations as f64
    }

    /// Signal shutdown to the kernel without waiting for final state.
    /// Called by Drop to ensure the kernel is stopped.
    fn signal_shutdown(&self) {
        if !self.is_initialized {
            return;
        }

        eprintln!(
            "[HOST] signal_shutdown: signaling SHUTDOWN to {} cubes",
            self.config.num_cubes()
        );
        // Use cached pointer to avoid get_resource() which would sync with kernel stream
        let ctrl_ptr = self.cached_ctrl_ptr;

        // Signal SHUTDOWN to all cubes
        let num_cubes = self.config.num_cubes();
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(ctrl_ptr, 0, &shutdown_signals);
        self.stream.sync();

        // Verify the write
        let readback = self.stream.read(ctrl_ptr, 0, num_cubes);
        eprintln!(
            "[HOST] signal_shutdown: control after write = {:?}",
            readback
        );

        // Wait for kernel to finish
        eprintln!("[HOST] signal_shutdown: waiting for kernel to finish");
        if let Err(e) = wait_for_sync(&self.kernel_client) {
            eprintln!(
                "[HOST] signal_shutdown: sync error (may be expected): {:?}",
                e
            );
        }
        // Also sync the main client to ensure all GPU work is done
        eprintln!("[HOST] signal_shutdown: syncing main client");
        if let Err(e) = wait_for_sync(&self.client) {
            eprintln!("[HOST] signal_shutdown: main client sync error: {:?}", e);
        }
        eprintln!("[HOST] signal_shutdown: done");
    }
}

impl<R: CubeRuntime> Drop for TttD2dStreamingState<R> {
    fn drop(&mut self) {
        self.signal_shutdown();
    }
}
