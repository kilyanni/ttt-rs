//! Host-side state management for pointer-based streaming kernel.
//!
//! This provides true zero-copy input by writing tensor addresses to a pointer table
//! that the kernel reads directly.

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use burn::tensor::Shape;
use burn_cubecl::{
    CubeRuntime, FloatElement,
    ops::numeric::{empty_device, zeros_client},
    tensor::CubeTensor,
};
use cubecl::{frontend::ArrayArg, prelude::*};
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
        CTRL_ARRAY_SIZE, PTR_OUTPUT, PTR_TABLE_SIZE, STATUS_DONE, STATUS_IDLE, STATUS_READY,
        STATUS_SHUTDOWN, fused_ttt_streaming_ptr_kernel,
    },
};
use crate::FusedTttConfig;

/// Configuration for pointer-based streaming.
#[derive(Debug, Clone, Copy)]
pub struct PtrStreamingConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub mini_batch_len: usize,
    pub head_dim: usize,
    pub epsilon: f32,
    pub threads: usize,
    pub debug: bool,
}

impl PtrStreamingConfig {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            mini_batch_len,
            head_dim,
            epsilon,
            threads,
            debug: std::env::var("PTR_STREAM_DEBUG").is_ok(),
        }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn num_cubes(&self) -> usize {
        self.batch_size * self.num_heads
    }

    pub fn key(&self, stream_id: u64) -> u64 {
        stream_id
    }
}

/// Type-erased wrapper for storing in the global registry.
struct AnyPtrStreamingState(Box<dyn std::any::Any + Send>);

impl AnyPtrStreamingState {
    fn new<R: CubeRuntime + 'static>(state: TttPtrStreamingState<R>) -> Self {
        Self(Box::new(state))
    }

    fn downcast_mut<R: CubeRuntime + 'static>(&mut self) -> Option<&mut TttPtrStreamingState<R>> {
        self.0.downcast_mut()
    }
}

/// Global registry of pointer-based streaming states.
static PTR_STREAMING_REGISTRY: LazyLock<Mutex<HashMap<u64, AnyPtrStreamingState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// All tensor handles for the streaming kernel.
/// Most are only accessed at launch; kept alive for kernel duration.
pub struct PtrStreamingTensors<R: CubeRuntime> {
    // --- Host-accessed tensors ---
    pub ptr_table: CubeTensor<R>,
    pub control: CubeTensor<R>,
    pub output: CubeTensor<R>,
    pub weight_out: CubeTensor<R>,
    pub bias_out: CubeTensor<R>,
    // --- Array buffers (for HIP pointer loads) ---
    pub xq_buf: CubeTensor<R>,
    pub xk_buf: CubeTensor<R>,
    pub xv_buf: CubeTensor<R>,
    pub eta_buf: CubeTensor<R>,
    // --- Inputs struct tensors ---
    pub xq_scratch: CubeTensor<R>,
    pub xk_scratch: CubeTensor<R>,
    pub xv_scratch: CubeTensor<R>,
    pub weight: CubeTensor<R>,
    pub bias: CubeTensor<R>,
    pub token_eta: CubeTensor<R>,
    pub ttt_lr_eta_scratch: CubeTensor<R>,
    pub ln_weight: CubeTensor<R>,
    pub ln_bias: CubeTensor<R>,
}

use super::super::super::next_persistent_kernel_stream_id;

/// Get or create a pointer-based streaming state from the global registry.
#[allow(clippy::too_many_arguments)]
pub fn get_or_create_ptr_streaming_state<R: CubeRuntime + 'static, F: FloatElement>(
    stream_id: u64,
    config: PtrStreamingConfig,
    client: ComputeClient<R>,
    device: R::Device,
    initial_weight: CubeTensor<R>,
    initial_bias: CubeTensor<R>,
    token_eta: CubeTensor<R>,
    ln_weight: CubeTensor<R>,
    ln_bias: CubeTensor<R>,
) -> &'static mut TttPtrStreamingState<R> {
    let mut registry = PTR_STREAMING_REGISTRY.lock().unwrap();

    if !registry.contains_key(&stream_id) {
        let state = TttPtrStreamingState::new::<F>(
            config,
            client,
            device,
            initial_weight,
            initial_bias,
            token_eta,
            ln_weight,
            ln_bias,
        );
        registry.insert(stream_id, AnyPtrStreamingState::new(state));
    }

    let state = registry.get_mut(&stream_id).unwrap();
    let state_ptr = state.downcast_mut::<R>().unwrap() as *mut TttPtrStreamingState<R>;

    // SAFETY: The registry is static and we're returning a reference that
    // will be used within the same forward_launch call
    unsafe { &mut *state_ptr }
}

/// Remove a pointer-based streaming state from the registry.
pub fn remove_ptr_streaming_state<R: CubeRuntime + 'static>(
    stream_id: u64,
) -> Option<TttPtrStreamingState<R>> {
    let mut registry = PTR_STREAMING_REGISTRY.lock().unwrap();
    registry
        .remove(&stream_id)
        .and_then(|any| match any.0.downcast::<TttPtrStreamingState<R>>() {
            Ok(boxed) => Some(*boxed),
            Err(_) => None,
        })
}

/// Remove a pointer-based streaming state by ID, triggering its Drop impl for cleanup.
pub fn remove_ptr_streaming_state_by_id(stream_id: u64) {
    let mut registry = PTR_STREAMING_REGISTRY.lock().unwrap();
    registry.remove(&stream_id);
}

/// Shutdown and remove a pointer-based streaming state, returning final weight/bias.
pub fn shutdown_ptr_streaming_state<R: CubeRuntime + 'static>(
    stream_id: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    remove_ptr_streaming_state::<R>(stream_id).map(|state| state.shutdown())
}

/// State for pointer-based streaming execution.
pub struct TttPtrStreamingState<R: CubeRuntime> {
    pub config: PtrStreamingConfig,
    pub stream: AsyncStream,
    pub tensors: PtrStreamingTensors<R>,
    /// Raw GPU pointers for async access
    pub ptr_table_ptr: GpuPtr<'static, u64>,
    pub control_ptr: GpuPtr<'static, u32>,
    pub output_ptr: GpuPtr<'static, f32>,
    /// Client for normal GPU operations
    client: ComputeClient<R>,
    /// Client with a separate stream for the persistent kernel.
    /// This prevents the persistent kernel from blocking normal operations.
    kernel_client: ComputeClient<R>,
}

impl<R: CubeRuntime + 'static> TttPtrStreamingState<R> {
    /// Create a new streaming state and launch the persistent kernel.
    #[allow(unused_variables)]
    pub fn new<F: FloatElement>(
        config: PtrStreamingConfig,
        client: ComputeClient<R>,
        device: R::Device,
        // Initial state tensors
        initial_weight: CubeTensor<R>,
        initial_bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
    ) -> Self {
        let stream = AsyncStream::new();
        let num_cubes = config.num_cubes();
        let mini_batch_len = config.mini_batch_len;
        let head_dim = config.head_dim;

        // Create kernel_client FIRST so we can allocate control on its stream.
        // This ensures the kernel sees initialized zeros without cross-stream sync.
        let mut kernel_client = client.clone();
        let kernel_stream_id = next_persistent_kernel_stream_id();
        trace!("ptr_stream: using kernel stream ID: {}", kernel_stream_id);
        unsafe {
            kernel_client.set_stream(kernel_stream_id);
        }

        // Helper to allocate a tensor
        let alloc = |shape: Vec<usize>| {
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(shape))
        };
        let alloc_u64 = |shape: Vec<usize>| {
            empty_device::<R, u64>(client.clone(), device.clone(), Shape::from(shape))
        };

        // --- Allocate all tensors ---
        let ptr_table = alloc_u64(vec![PTR_TABLE_SIZE]);
        // Allocate control on kernel_client's stream with zeros - kernel will see this
        let control = zeros_client::<R>(
            kernel_client.clone(),
            device.clone(),
            Shape::from([num_cubes * CTRL_ARRAY_SIZE]),
            burn_backend::DType::U32,
        );

        // Array buffers for HIP pointer loads
        let qkv_buf_size_per_cube = mini_batch_len * head_dim;
        let eta_buf_size_per_cube = mini_batch_len;
        let xq_buf = alloc(vec![num_cubes * qkv_buf_size_per_cube]);
        let xk_buf = alloc(vec![num_cubes * qkv_buf_size_per_cube]);
        let xv_buf = alloc(vec![num_cubes * qkv_buf_size_per_cube]);
        let eta_buf = alloc(vec![num_cubes * eta_buf_size_per_cube]);

        // Scratch tensors for Inputs
        let xq_scratch = alloc(vec![
            config.batch_size,
            config.num_heads,
            mini_batch_len,
            head_dim,
        ]);
        let xk_scratch = alloc(vec![
            config.batch_size,
            config.num_heads,
            mini_batch_len,
            head_dim,
        ]);
        let xv_scratch = alloc(vec![
            config.batch_size,
            config.num_heads,
            mini_batch_len,
            head_dim,
        ]);
        let ttt_lr_eta_scratch = alloc(vec![config.batch_size, config.num_heads, mini_batch_len]);

        // Outputs
        let output = alloc(vec![
            config.batch_size,
            config.num_heads,
            mini_batch_len,
            head_dim,
        ]);
        let weight_out = alloc(vec![
            config.batch_size,
            config.num_heads,
            head_dim,
            head_dim,
        ]);
        let bias_out = alloc(vec![config.batch_size, config.num_heads, head_dim]);

        // Hidden state
        let weight = alloc(vec![
            config.batch_size,
            config.num_heads,
            head_dim,
            head_dim,
        ]);
        let bias = alloc(vec![config.batch_size, config.num_heads, head_dim]);

        // Get raw pointers for host access
        let ptr_table_ptr: GpuPtr<'static, u64> =
            unsafe { std::mem::transmute(stream.ptr::<u64, R>(&client, &ptr_table.handle)) };
        let control_ptr: GpuPtr<'static, u32> =
            unsafe { std::mem::transmute(stream.ptr::<u32, R>(&client, &control.handle)) };
        let output_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &output.handle)) };

        // Get pointers for weight/bias initialization
        let weight_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &weight.handle)) };
        let bias_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &bias.handle)) };

        // Write output address to pointer table
        let output_addr = output_ptr.address();
        stream.write(ptr_table_ptr, PTR_OUTPUT, &[output_addr]);

        // Initialize weight/bias buffers from initial values
        // The initial values are [num_heads, head_dim, head_dim] - replicate for each batch
        trace!("ptr_stream: initializing weight/bias from initial values...");
        let src_weight: GpuPtr<f32> = stream.ptr(&client, &initial_weight.handle);
        let src_bias: GpuPtr<f32> = stream.ptr(&client, &initial_bias.handle);

        let per_head_weight_size = head_dim * head_dim;
        let per_head_bias_size = head_dim;

        if src_weight.len() == config.num_heads * per_head_weight_size {
            // Source is [num_heads, head_dim, head_dim] - replicate for each batch
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_weight_size;
                stream.copy_d2d(
                    weight_ptr,
                    batch_offset,
                    src_weight,
                    0,
                    config.num_heads * per_head_weight_size,
                );
            }
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_bias_size;
                stream.copy_d2d(
                    bias_ptr,
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
            stream.copy_d2d(
                weight_ptr,
                0,
                src_weight,
                0,
                full_weight_len.min(src_weight.len()),
            );
            stream.copy_d2d(bias_ptr, 0, src_bias, 0, full_bias_len.min(src_bias.len()));
        }
        stream.sync();

        let tensors = PtrStreamingTensors {
            ptr_table,
            control,
            output,
            weight_out,
            bias_out,
            xq_buf,
            xk_buf,
            xv_buf,
            eta_buf,
            xq_scratch,
            xk_scratch,
            xv_scratch,
            weight,
            bias,
            token_eta,
            ttt_lr_eta_scratch,
            ln_weight,
            ln_bias,
        };

        let state = Self {
            config,
            stream,
            tensors,
            ptr_table_ptr,
            control_ptr,
            output_ptr,
            client,
            kernel_client,
        };

        // Launch the persistent kernel
        trace!("ptr_stream: launching persistent kernel");
        state.launch_kernel::<F>();
        trace!("ptr_stream: kernel launched");

        state
    }

    /// Launch the persistent streaming kernel with pointer indirection.
    fn launch_kernel<F: FloatElement>(&self) {
        let fused_config = FusedTttConfig::new(
            self.config.mini_batch_len,
            self.config.head_dim,
            self.config.epsilon,
            self.config.threads,
        );
        let debug = self.config.debug;
        let batch_size = self.config.batch_size as u32;
        let num_heads = self.config.num_heads as u32;
        let mini_batch_len = self.config.mini_batch_len;
        let head_dim = self.config.head_dim;
        let threads = self.config.threads;

        let cube_count = CubeCount::Static(batch_size, num_heads, 1);
        let vectorization = LINE_SIZE;

        // Array sizes (in Line<F> units) - total for all cubes
        let num_cubes = self.config.num_cubes();
        let qkv_arr_len = num_cubes * mini_batch_len * head_dim / LINE_SIZE;
        let eta_arr_len = num_cubes * mini_batch_len / LINE_SIZE;

        // Create ArrayArgs
        let xq_arg = unsafe {
            ArrayArg::from_raw_parts::<F>(&self.tensors.xq_buf.handle, qkv_arr_len, vectorization)
        };
        let xk_arg = unsafe {
            ArrayArg::from_raw_parts::<F>(&self.tensors.xk_buf.handle, qkv_arr_len, vectorization)
        };
        let xv_arg = unsafe {
            ArrayArg::from_raw_parts::<F>(&self.tensors.xv_buf.handle, qkv_arr_len, vectorization)
        };
        let eta_arg = unsafe {
            ArrayArg::from_raw_parts::<F>(&self.tensors.eta_buf.handle, eta_arr_len, vectorization)
        };

        // Get all handle refs (must outlive the Launch structs)
        let xq_scratch_ref = self.tensors.xq_scratch.as_handle_ref();
        let xk_scratch_ref = self.tensors.xk_scratch.as_handle_ref();
        let xv_scratch_ref = self.tensors.xv_scratch.as_handle_ref();
        let weight_ref = self.tensors.weight.as_handle_ref();
        let bias_ref = self.tensors.bias.as_handle_ref();
        let token_eta_ref = self.tensors.token_eta.as_handle_ref();
        let ttt_lr_eta_ref = self.tensors.ttt_lr_eta_scratch.as_handle_ref();
        let ln_weight_ref = self.tensors.ln_weight.as_handle_ref();
        let ln_bias_ref = self.tensors.ln_bias.as_handle_ref();
        let output_ref = self.tensors.output.as_handle_ref();
        let weight_out_ref = self.tensors.weight_out.as_handle_ref();
        let bias_out_ref = self.tensors.bias_out.as_handle_ref();
        let ptr_table_ref = self.tensors.ptr_table.as_handle_ref();
        let control_ref = self.tensors.control.as_handle_ref();

        // Build InputsLaunch
        let inputs = InputsLaunch::<F, R>::new(
            xq_scratch_ref.as_tensor_arg(vectorization),
            xk_scratch_ref.as_tensor_arg(vectorization),
            xv_scratch_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
            token_eta_ref.as_tensor_arg(vectorization),
            ttt_lr_eta_ref.as_tensor_arg(vectorization),
            ln_weight_ref.as_tensor_arg(vectorization),
            ln_bias_ref.as_tensor_arg(vectorization),
        );

        // Build OutputsLaunch
        let outputs = OutputsLaunch::<F, R>::new(
            output_ref.as_tensor_arg(vectorization),
            weight_out_ref.as_tensor_arg(vectorization),
            bias_out_ref.as_tensor_arg(vectorization),
        );

        // Dispatch based on tile configuration
        // Use kernel_client which has a separate stream for the persistent kernel
        tile_dispatch!(
            fused_ttt_streaming_ptr_kernel,
            &self.kernel_client,
            cube_count,
            mini_batch_len,
            head_dim,
            threads,
            ptr_table_ref.as_tensor_arg(1),
            control_ref.as_tensor_arg(1),
            xq_arg,
            xk_arg,
            xv_arg,
            eta_arg,
            inputs,
            outputs,
            fused_config,
            debug
        );
    }

    /// Feed a mini-batch by writing tensor addresses to the pointer table.
    ///
    /// This is true zero-copy - we just write the addresses, no data is copied.
    pub fn feed_mini_batch(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) {
        trace!("ptr_stream: feed_mini_batch start");

        // Get addresses of input tensors, accounting for tensor offsets from slicing.
        // When a tensor is sliced, handle.offset_start contains the byte offset into the buffer.
        let xq_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xq.handle);
        let xk_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xk.handle);
        let xv_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xv.handle);
        let eta_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &ttt_lr_eta.handle);

        // Write addresses to pointer table (base + offset for sliced tensors)
        let addrs = [
            xq_ptr.address() + xq.handle.offset_start.unwrap_or(0),
            xk_ptr.address() + xk.handle.offset_start.unwrap_or(0),
            xv_ptr.address() + xv.handle.offset_start.unwrap_or(0),
            eta_ptr.address() + ttt_lr_eta.handle.offset_start.unwrap_or(0),
            self.output_ptr.address(),
        ];
        self.stream.write(self.ptr_table_ptr, 0, &addrs);

        // Sync to ensure addresses are written before signaling kernel
        // (kernel is on a different stream, so needs explicit sync)
        self.stream.sync();

        trace!(
            "ptr_stream: writing READY to {} cubes",
            self.config.num_cubes()
        );

        // Signal READY to all cubes and sync to ensure visibility across streams
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();
        self.stream.write(self.control_ptr, 0, &ready_signals);
        self.stream.sync();
    }

    /// Wait for all cubes to complete using two-phase polling.
    /// This prevents reading stale DONE values from previous iterations.
    fn wait_for_cubes_done(&self) {
        let num_cubes = self.config.num_cubes();
        if self.config.debug {
            eprintln!("HOST: waiting for {} cubes DONE", num_cubes);
        }

        for cube in 0..num_cubes {
            // First, wait for status to NOT be DONE (kernel started processing and cleared it)
            loop {
                let status = self.stream.read(self.control_ptr, cube, 1);
                if status[0] != STATUS_DONE {
                    break;
                }
                std::hint::spin_loop();
            }

            // Now poll for DONE
            loop {
                let status = self.stream.read(self.control_ptr, cube, 1);
                if status[0] == STATUS_DONE {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        if self.config.debug {
            eprintln!("HOST: all cubes DONE, writing IDLE");
        }

        // Sync and reset to IDLE
        self.stream.sync();
        let idle_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_IDLE).collect();
        self.stream.write(self.control_ptr, 0, &idle_signals);
        self.stream.sync();
    }

    /// Feed inputs and wait, returning the output tensor directly.
    pub fn forward_tensor(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) -> &CubeTensor<R> {
        if self.config.debug {
            eprintln!("HOST: forward_tensor start");
        }
        self.feed_mini_batch(xq, xk, xv, ttt_lr_eta);
        self.wait_for_cubes_done();
        &self.tensors.output
    }

    /// Signal shutdown and retrieve final weight/bias.
    pub fn shutdown(self) -> (Vec<f32>, Vec<f32>) {
        trace!("ptr_stream: shutdown start");
        let num_cubes = self.config.num_cubes();

        // Signal SHUTDOWN to all cubes
        trace!("ptr_stream: writing SHUTDOWN to {} cubes", num_cubes);
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(self.control_ptr, 0, &shutdown_signals);

        // Sync to ensure the SHUTDOWN signal is written
        self.stream.sync();

        // Wait for kernel to exit and write final state
        trace!("ptr_stream: waiting for kernel exit");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Sync GPU to ensure the persistent kernel has finished (on kernel_client's stream)
        wait_for_sync(&self.kernel_client).expect("GPU sync failed");

        // Read final weight and bias
        let weight_len = self.config.batch_size
            * self.config.num_heads
            * self.config.head_dim
            * self.config.head_dim;
        let bias_len = self.config.batch_size * self.config.num_heads * self.config.head_dim;

        trace!(
            "ptr_stream: reading final weight ({}) and bias ({})",
            weight_len, bias_len
        );
        let weight_ptr: GpuPtr<f32> = self
            .stream
            .ptr(&self.client, &self.tensors.weight_out.handle);
        let bias_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &self.tensors.bias_out.handle);

        let weight = self.stream.read(weight_ptr, 0, weight_len);
        let bias = self.stream.read(bias_ptr, 0, bias_len);

        trace!("ptr_stream: shutdown complete");
        (weight, bias)
    }

    /// Signal shutdown to the kernel without waiting for final state.
    /// Called by Drop to ensure the kernel is stopped.
    fn signal_shutdown(&self) {
        trace!("ptr_stream: signal_shutdown start");
        let num_cubes = self.config.num_cubes();

        // Signal SHUTDOWN to all cubes
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(self.control_ptr, 0, &shutdown_signals);
        self.stream.sync();

        // Wait for kernel to finish
        if let Err(e) = wait_for_sync(&self.kernel_client) {
            trace!(
                "ptr_stream: signal_shutdown sync error (may be expected): {:?}",
                e
            );
        }
        trace!("ptr_stream: signal_shutdown done");
    }
}

/// Access a pointer-based streaming state by ID for benchmarking.
#[allow(dead_code)]
pub fn with_ptr_streaming_state<R: CubeRuntime + 'static, T>(
    stream_id: u64,
    f: impl FnOnce(&TttPtrStreamingState<R>) -> T,
) -> T {
    let mut registry = PTR_STREAMING_REGISTRY.lock().unwrap();
    let state = registry
        .get_mut(&stream_id)
        .expect("ptr streaming state not found")
        .downcast_mut::<R>()
        .expect("type mismatch");
    f(state)
}

impl<R: CubeRuntime + 'static> TttPtrStreamingState<R> {
    /// Benchmark pure kernel compute time.
    /// Assumes data is already in the buffers from a prior forward call.
    /// Runs `iterations` cycles of READY→DONE.
    /// Returns per-iteration time in microseconds.
    pub fn bench_compute(&self, warmup: usize, iterations: usize) -> f64 {
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();

        // Warmup
        for _ in 0..warmup {
            self.stream.write(self.control_ptr, 0, &ready_signals);
            self.stream.sync();
            self.wait_for_cubes_done();
        }

        // Timed iterations
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            self.stream.write(self.control_ptr, 0, &ready_signals);
            self.stream.sync();
            self.wait_for_cubes_done();
        }
        let elapsed = start.elapsed();

        elapsed.as_micros() as f64 / iterations as f64
    }
}

impl<R: CubeRuntime + 'static> Drop for TttPtrStreamingState<R> {
    fn drop(&mut self) {
        self.signal_shutdown();
    }
}

// Safety: The streaming state is designed to be used from a single thread
// but the underlying handles are Send
unsafe impl<R: CubeRuntime> Send for TttPtrStreamingState<R> {}
