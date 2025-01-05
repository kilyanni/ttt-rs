//! Test whether CubeCL performs shared memory liveness analysis.
//!
//! Run with:
//!   CUBECL_DEBUG_LOG=stdout cargo run --example smem_liveness --features rocm
//!
//! Two kernels are compiled:
//!
//! 1. `overlapping_smem` — allocates two 8×8 shared memory tiles that are
//!    both live at the same time.  The compiler *must* give them separate
//!    backing storage → 2 smem allocations.
//!
//! 2. `sequential_smem` — allocates one 8×8 tile in an inner scope, uses it
//!    and writes the result to a register tile, then drops it. A *second*
//!    8×8 tile is allocated in a later scope.  If CubeCL performs liveness
//!    analysis, the two tiles can share the same backing storage → 1 smem
//!    allocation.
//!
//! Inspect the generated source (printed by CUBECL_DEBUG_LOG) and count the
//! number of distinct `__shared__` / `var local_shared` declarations.

use cubecl::prelude::*;
use thundercube::{
    prelude::*,
    test_utils::{TestRuntime, client},
};

// ---------------------------------------------------------------------------
// Kernel 1 – both tiles alive simultaneously
// ---------------------------------------------------------------------------
#[cube(launch)]
fn overlapping_smem<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    // Two tiles alive at the same time → must use separate smem
    let mut tile_a = St::<F, D8, D8>::new();
    let mut tile_b = St::<F, D8, D8>::new();

    cube::load_st_direct(input, &mut tile_a, 0, 0, 0);
    sync_cube();

    // Copy a → b element-wise so both are used
    tile_b.copy_from(&tile_a);
    sync_cube();

    // Use both: a += b  (requires both to be live)
    tile_a.add(&tile_b);
    sync_cube();

    cube::store_st_direct(&tile_a, output, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Kernel 2 – tiles in non-overlapping scopes (sequential lifetimes)
// ---------------------------------------------------------------------------
#[cube(launch)]
fn sequential_smem<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    // Phase 1: allocate tile, load, store to registers, drop tile
    let mut rt = Rt::<F, D8, D8>::new();
    rt.zero();
    {
        let mut tile_a = St::<F, D8, D8>::new();
        cube::load_st_direct(input, &mut tile_a, 0, 0, 0);
        sync_cube();

        // Move data into register tile (tile_a no longer needed after this)
        cube::mma_AB(&mut rt, &tile_a, &tile_a);
        sync_cube();
    }
    // tile_a is now out of scope

    // Phase 2: allocate a *new* tile — if liveness analysis works, this
    // can reuse tile_a's backing storage
    {
        let mut tile_b = St::<F, D8, D8>::new();
        cube::store_rt_to_st(&rt, &mut tile_b);
        sync_cube();

        cube::store_st_direct(&tile_b, output, 0, 0, 0);
    }
}

// ---------------------------------------------------------------------------
fn main() {
    let client = client();
    const SIZE: usize = 8 * 8;

    let data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let zeros = vec![0.0f32; SIZE];

    let shape = vec![8usize, 8];
    let strides = vec![8usize, 1];

    // --- Kernel 1: overlapping ---
    println!("=== Launching overlapping_smem (expect 2 smem allocations) ===");
    {
        let h_in = client.create_from_slice(f32::as_bytes(&data));
        let h_out = client.create_from_slice(f32::as_bytes(&zeros));
        let arg_in =
            unsafe { TensorArg::from_raw_parts::<Line<f32>>(&h_in, &strides, &shape, LINE_SIZE) };
        let arg_out =
            unsafe { TensorArg::from_raw_parts::<Line<f32>>(&h_out, &strides, &shape, LINE_SIZE) };

        overlapping_smem::launch::<f32, TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            arg_in,
            arg_out,
        )
        .expect("overlapping_smem launch failed");

        let bytes = client.read_one(h_out);
        let result: Vec<f32> = f32::from_bytes(&bytes).to_vec();
        println!("Result[0..8]: {:?}\n", &result[0..8]);
    }

    // --- Kernel 2: sequential ---
    println!("=== Launching sequential_smem (liveness → could reuse, expect 1 smem alloc) ===");
    {
        let h_in = client.create_from_slice(f32::as_bytes(&data));
        let h_out = client.create_from_slice(f32::as_bytes(&zeros));
        let arg_in =
            unsafe { TensorArg::from_raw_parts::<Line<f32>>(&h_in, &strides, &shape, LINE_SIZE) };
        let arg_out =
            unsafe { TensorArg::from_raw_parts::<Line<f32>>(&h_out, &strides, &shape, LINE_SIZE) };

        sequential_smem::launch::<f32, TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            arg_in,
            arg_out,
        )
        .expect("sequential_smem launch failed");

        let bytes = client.read_one(h_out);
        let result: Vec<f32> = f32::from_bytes(&bytes).to_vec();
        println!("Result[0..8]: {:?}\n", &result[0..8]);
    }

    println!("Done! Check the CUBECL_DEBUG_LOG output above for shared memory declarations.");
    println!("Look for `__shared__` (HIP/CUDA) or shared memory variable declarations.");
    println!("If sequential_smem has fewer distinct shared memory arrays than overlapping_smem,");
    println!("then CubeCL performs liveness-based smem reuse.");
}
