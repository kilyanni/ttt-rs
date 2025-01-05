//! Async I/O primitives for persistent kernels.
//!
//! When a kernel is running on the default HIP stream, normal `hipMemcpy` calls
//! block waiting for it to finish. This module provides [`AsyncStream`], which
//! creates a separate non-blocking stream for memory transfers.
//!
//! # Example
//!
//! ```ignore
//! use thundercube::streaming::{AsyncStream, GpuPtr};
//!
//! let stream = AsyncStream::new();
//!
//! // Get raw pointers BEFORE launching blocking kernel
//! let data_ptr: GpuPtr<f32> = stream.ptr(&client, &data_handle);
//! let ctrl_ptr: GpuPtr<u32> = stream.ptr(&client, &ctrl_handle);
//!
//! // Launch your persistent kernel...
//! my_kernel::launch(...);
//!
//! // Now you can read/write while the kernel runs
//! stream.write(ctrl_ptr, &[1, 0, 0]);
//!
//! loop {
//!     let data = stream.read(ctrl_ptr, 3);
//!
//!     do_stuff(data);
//! }
//!
//! let result = stream.read(data_ptr, n);
//! ```

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::*;
use cubecl_hip_sys::{
    HIP_SUCCESS, hipDeviceptr_t, hipMemcpyAsync, hipMemcpyDtoDAsync,
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice, hipStream_t,
    hipStreamCreate, hipStreamDestroy, hipStreamSynchronize,
};

/// Raw GPU pointer for direct memory access.
///
/// Obtained via [`AsyncStream::ptr`]. The pointer remains valid as long as
/// the underlying cubecl handle is alive.
///
/// Tracks buffer capacity for bounds checking on read/write operations.
#[derive(Clone, Copy)]
pub struct GpuPtr<'a, T> {
    ptr: hipDeviceptr_t,
    /// Number of elements of type T
    len: usize,
    _marker: PhantomData<&'a T>,
}

impl<T> GpuPtr<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Get the raw device address as u64.
    ///
    /// This can be stored in GPU memory and used for pointer indirection
    /// in persistent kernels.
    pub fn address(&self) -> u64 {
        self.ptr as u64
    }

    /// Get the raw device address with an element offset as u64.
    pub fn address_at(&self, offset: usize) -> u64 {
        assert!(
            offset <= self.len,
            "offset {} exceeds buffer length {}",
            offset,
            self.len
        );
        let offset_bytes = offset * std::mem::size_of::<T>();
        (self.ptr as usize + offset_bytes) as u64
    }
}

/// Mirror of cubecl_hip's internal GpuResource layout.
/// Used to extract the raw pointer from a cubecl binding.
#[repr(C)]
struct GpuResourceCompat {
    ptr: hipDeviceptr_t,
    _binding: *mut std::ffi::c_void,
    _size: u64,
}

/// Async I/O stream for communicating with running kernels.
///
/// Creates a separate HIP stream that doesn't synchronize with the default
/// stream, allowing memory transfers while kernels run.
pub struct AsyncStream {
    stream: hipStream_t,
}

impl AsyncStream {
    pub fn new() -> Self {
        let mut stream: hipStream_t = std::ptr::null_mut();
        unsafe {
            let status = hipStreamCreate(&mut stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamCreate failed with status: {}", status);
            }
        }
        Self { stream }
    }

    /// Extract a raw GPU pointer from a cubecl handle.
    ///
    /// Must be called BEFORE launching a blocking kernel, as this
    /// operation may synchronize with the default stream.
    ///
    /// It is the callers responsibility to ensure that the type
    /// makes sense for the given handle.
    pub fn ptr<'a, T: Pod, R: Runtime>(
        &self,
        client: &'_ ComputeClient<R>,
        handle: &'a cubecl::server::Handle,
    ) -> GpuPtr<'a, T> {
        let binding = handle.clone().binding();
        let resource = client.get_resource(binding);
        let gpu_resource: &GpuResourceCompat =
            unsafe { &*(resource.resource() as *const _ as *const GpuResourceCompat) };
        let size_bytes = gpu_resource._size as usize;
        let elem_size = std::mem::size_of::<T>();
        assert!(
            size_bytes.is_multiple_of(elem_size),
            "buffer size {} is not a multiple of element size {}",
            size_bytes,
            elem_size
        );
        GpuPtr {
            ptr: gpu_resource.ptr,
            len: size_bytes / elem_size,
            _marker: PhantomData,
        }
    }

    /// Host-to-device async write.
    pub fn write<T: Pod>(&self, dst: GpuPtr<T>, offset: usize, data: &[T]) {
        assert!(
            offset + data.len() <= dst.len,
            "write at offset {} of {} elements exceeds buffer capacity of {}",
            offset,
            data.len(),
            dst.len
        );
        let size_bytes = std::mem::size_of_val(data);
        let offset_bytes = offset * std::mem::size_of::<T>();
        unsafe {
            let status = hipMemcpyAsync(
                dst.ptr.byte_add(offset_bytes),
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
                self.stream,
            );
            if status != HIP_SUCCESS {
                panic!("hipMemcpyAsync H2D failed with status: {}", status);
            }
            let status = hipStreamSynchronize(self.stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamSynchronize failed with status: {}", status);
            }
        }
    }

    /// Device-to-host async read.
    pub fn read<T: Pod + Clone>(&self, src: GpuPtr<T>, offset: usize, len: usize) -> Vec<T> {
        assert!(
            offset + len <= src.len,
            "read at offset {} of {} elements exceeds buffer capacity of {}",
            offset,
            len,
            src.len
        );
        let mut data = vec![T::zeroed(); len];
        let size_bytes = std::mem::size_of::<T>() * len;
        let offset_bytes = offset * std::mem::size_of::<T>();
        unsafe {
            let status = hipMemcpyAsync(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                src.ptr.byte_add(offset_bytes),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                self.stream,
            );
            if status != HIP_SUCCESS {
                panic!("hipMemcpyAsync D2H failed with status: {}", status);
            }
            let status = hipStreamSynchronize(self.stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamSynchronize failed with status: {}", status);
            }
        }
        data
    }

    /// Device-to-device async copy.
    ///
    /// Copies `count` elements from `src` (at `src_offset`) to `dst` (at `dst_offset`).
    /// This is a fast GPU-to-GPU copy that doesn't involve the CPU.
    pub fn copy_d2d<T: Pod>(
        &self,
        dst: GpuPtr<T>,
        dst_offset: usize,
        src: GpuPtr<T>,
        src_offset: usize,
        count: usize,
    ) {
        assert!(
            src_offset + count <= src.len,
            "copy source at offset {} of {} elements exceeds buffer capacity of {}",
            src_offset,
            count,
            src.len
        );
        assert!(
            dst_offset + count <= dst.len,
            "copy dest at offset {} of {} elements exceeds buffer capacity of {}",
            dst_offset,
            count,
            dst.len
        );
        let size_bytes = std::mem::size_of::<T>() * count;
        let src_offset_bytes = src_offset * std::mem::size_of::<T>();
        let dst_offset_bytes = dst_offset * std::mem::size_of::<T>();
        unsafe {
            let status = hipMemcpyDtoDAsync(
                dst.ptr.byte_add(dst_offset_bytes),
                src.ptr.byte_add(src_offset_bytes),
                size_bytes,
                self.stream,
            );
            if status != HIP_SUCCESS {
                panic!("hipMemcpyDtoDAsync failed with status: {}", status);
            }
        }
    }

    /// Synchronize the async stream.
    ///
    /// Waits for all pending operations on this stream to complete.
    pub fn sync(&self) {
        unsafe {
            let status = hipStreamSynchronize(self.stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamSynchronize failed with status: {}", status);
            }
        }
    }

    /// Write device addresses to a pointer table in GPU memory.
    ///
    /// This is used for pointer indirection: the kernel reads these addresses
    /// and uses them to access the actual tensor data.
    pub fn write_pointer_table<T: Pod>(&self, table: GpuPtr<u64>, slot: usize, ptr: GpuPtr<T>) {
        let addr = ptr.address();
        self.write(table, slot, &[addr]);
    }
}

impl Default for AsyncStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncStream {
    fn drop(&mut self) {
        unsafe {
            // Sync before destroy to ensure no pending operations block other streams
            hipStreamSynchronize(self.stream);
            hipStreamDestroy(self.stream);
        }
    }
}

/// Raw HIP code injection for CubeCL kernels.
///
/// CubeCL generates predictable variable names based on declaration order:
/// - **Kernel params (Tensor/Array)**: `buffer_0`, `buffer_1`, ... (parameter order)
/// - **Local arrays**: `l_arr_0`, `l_arr_1`, ... (declaration order)
/// - **Shared memory**: `s_0`, `s_1`, ... (declaration order)
/// - **Mutable vars**: `l_mut_N`
/// - **Temporaries**: `l_N`
/// - **Scalars**: `scalars_float[N]`, `scalars_uint[N]`, ...
///
/// Use `CUBECL_DEBUG_LOG=stdout` to verify names for your kernel.
pub mod ptr_inject {
    /// Generates HIP code to load from a pointer stored in a u64 buffer.
    ///
    /// # Arguments
    /// * `ptr_buffer` - Buffer index containing u64 addresses (e.g., 0 for buffer_0)
    /// * `slot` - Index within the pointer buffer
    /// * `dest_arr` - Local array index to load into (e.g., 0 for l_arr_0)
    /// * `count` - Number of float_4 elements to load
    /// * `offset` - Element offset from the pointer address
    pub fn load_from_ptr(
        ptr_buffer: usize,
        slot: usize,
        dest_arr: usize,
        count: usize,
        offset: usize,
    ) -> String {
        format!(
            r#"*/
const uint64 _ptr_addr_{dest_arr} = ((const uint64*)buffer_{ptr_buffer})[{slot}];
const float_4* _ptr_src_{dest_arr} = (const float_4*)(_ptr_addr_{dest_arr} + {offset} * sizeof(float_4));
for (int _i = 0; _i < {count}; _i++) {{ l_arr_{dest_arr}[_i] = _ptr_src_{dest_arr}[_i]; }}
/*"#
        )
    }

    /// Generates HIP code to store to a pointer stored in a u64 buffer.
    ///
    /// # Arguments
    /// * `ptr_buffer` - Buffer index containing u64 addresses
    /// * `slot` - Index within the pointer buffer
    /// * `src_arr` - Local array index to store from (e.g., 0 for l_arr_0)
    /// * `count` - Number of float_4 elements to store
    /// * `offset` - Element offset from the pointer address
    pub fn store_to_ptr(
        ptr_buffer: usize,
        slot: usize,
        src_arr: usize,
        count: usize,
        offset: usize,
    ) -> String {
        format!(
            r#"*/
const uint64 _ptr_addr_st_{src_arr} = ((const uint64*)buffer_{ptr_buffer})[{slot}];
float_4* _ptr_dst_{src_arr} = (float_4*)(_ptr_addr_st_{src_arr} + {offset} * sizeof(float_4));
for (int _i = 0; _i < {count}; _i++) {{ _ptr_dst_{src_arr}[_i] = l_arr_{src_arr}[_i]; }}
/*"#
        )
    }

    /// Generates HIP code to read a single u64 from a buffer into a local variable.
    pub fn read_u64(buffer: usize, index: usize, var_name: &str) -> String {
        format!(
            r#"*/
const uint64 {var_name} = ((const uint64*)buffer_{buffer})[{index}];
/*"#
        )
    }

    /// Generates HIP code to write a single u32 to a buffer.
    pub fn write_u32(buffer: usize, index: usize, value: &str) -> String {
        format!(
            r#"*/
((uint32*)buffer_{buffer})[{index}] = {value};
/*"#
        )
    }

    /// Generates HIP code for an atomic load of u32.
    pub fn atomic_load_u32(buffer: usize, index: usize, var_name: &str) -> String {
        format!(
            r#"*/
const uint32 {var_name} = atomicAdd((uint32*)&buffer_{buffer}[{index}], 0u);
/*"#
        )
    }

    /// Generates HIP code for an atomic store of u32.
    pub fn atomic_store_u32(buffer: usize, index: usize, value: &str) -> String {
        format!(
            r#"*/
atomicExch((uint32*)&buffer_{buffer}[{index}], {value});
/*"#
        )
    }
}
