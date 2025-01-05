//! Basic hello_world example for thundercube
//!
//! Demonstrates common operations on:
//! - Shared memory tiles (St) - for cooperative thread work
//! - Register tiles (Rt) - for per-thread SIMD work
//! - Register vectors (Rv) - for per-thread vectors

use cubecl::prelude::*;
use thundercube::{
    prelude::*,
    test_utils::{TestRuntime, client},
};

/// Kernel demonstrating various St (shared memory tile) operations
#[cube(launch)]
fn demo_st_ops<F: Float>(a: &Tensor<Line<F>>, b: &Tensor<Line<F>>, out: &mut Tensor<Line<F>>) {
    // Create shared memory tiles for 8x8 matrices
    let mut tile_a = St::<F, D8, D8>::new();
    let mut tile_b = St::<F, D8, D8>::new();

    // Load from global memory (direct = row-major layout)
    cube::load_st_direct(a, &mut tile_a, 0, 0, 0);
    cube::load_st_direct(b, &mut tile_b, 0, 0, 0);
    sync_cube();

    // Binary ops: tile_a += tile_b
    tile_a.add(&tile_b);
    sync_cube();

    // Create a row vector for broadcast operations
    let mut row_vec = Rv::<F, D8>::new();
    row_vec.fill(F::new(2.0));

    // Multiply each row by the vector (element-wise broadcast)
    tile_a.mul_row(&row_vec);
    sync_cube();

    // Triangular masking: keep only lower triangle
    tile_a.tril();
    sync_cube();

    // Store result
    cube::store_st_direct(&tile_a, out, 0, 0, 0);
}

/// Kernel demonstrating Rt (register tile) operations and matrix multiply
#[cube(launch)]
fn demo_rt_ops<F: Float>(a: &Tensor<Line<F>>, b: &Tensor<Line<F>>, out: &mut Tensor<Line<F>>) {
    // Load matrices into shared memory first
    let mut st_a = St::<F, D8, D8>::new();
    let mut st_b = St::<F, D8, D8>::new();
    cube::load_st_direct(a, &mut st_a, 0, 0, 0);
    cube::load_st_direct(b, &mut st_b, 0, 0, 0);
    sync_cube();

    // Create register tiles for matrix multiplication
    // Rt tiles are distributed across threads (each thread owns a portion)
    let mut rt_result = Rt::<F, D8, D8>::new();
    rt_result.zero();

    // Matrix multiply: result = A @ B
    cube::mma_AB(&mut rt_result, &st_a, &st_b);
    sync_cube();

    // Apply unary operations on the register tile
    rt_result.neg(); // Negate all elements

    // Add a scalar to each row
    let mut row_offset = Rv::<F, D8>::new();
    row_offset.fill(F::new(100.0));
    rt_result.add_row(&row_offset);

    // Store register tile back to shared memory, then to global
    let mut st_result = St::<F, D8, D8>::new();
    cube::store_rt_to_st(&rt_result, &mut st_result);
    sync_cube();

    cube::store_st_direct(&st_result, out, 0, 0, 0);
}

/// Kernel demonstrating Rv (register vector) operations
#[cube(launch)]
fn demo_rv_ops<F: Float>(input: &Tensor<Line<F>>, out: &mut Tensor<Line<F>>) {
    // Load a matrix into shared memory
    let mut st = St::<F, D8, D8>::new();
    cube::load_st_direct(input, &mut st, 0, 0, 0);
    sync_cube();

    // Create register vectors
    let mut col_vec = Rv::<F, D8>::new();
    col_vec.fill(F::new(0.5));

    // Apply column-wise scaling
    st.mul_col(&col_vec);
    sync_cube();

    // Unary ops on vectors
    col_vec.neg();
    col_vec.add_scalar(F::new(1.0));

    // Add as column broadcast
    st.add_col(&col_vec);
    sync_cube();

    cube::store_st_direct(&st, out, 0, 0, 0);
}

fn main() {
    let client = client();
    const SIZE: usize = 8 * 8;

    println!("Hello thundercube!\n");
    println!("Input A: 0, 1, 2, ... 63 (row-major 8x8)");
    println!("Input B: 64, 63, 62, ... 1 (row-major 8x8)");

    // Demo 1: St operations (add, broadcast mul, tril)
    {
        let a: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..SIZE).map(|i| (SIZE - i) as f32).collect();

        let handle_a = client.create_from_slice(f32::as_bytes(&a));
        let handle_b = client.create_from_slice(f32::as_bytes(&b));
        let handle_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; SIZE]));

        let shape = vec![8usize, 8];
        let strides = vec![8usize, 1];
        let arg_a = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides, &shape, LINE_SIZE)
        };
        let arg_b = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_b, &strides, &shape, LINE_SIZE)
        };
        let arg_out = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
        };

        demo_st_ops::launch::<f32, TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            arg_a,
            arg_b,
            arg_out,
        )
        .expect("st ops failed");

        let result_bytes = client.read_one(handle_out);
        let result: Vec<f32> = f32::from_bytes(&result_bytes).to_vec();

        println!("\n=== St Ops: (A + B) * 2 then tril ===");
        for row in 0..8 {
            println!("Row {}: {:?}", row, &result[row * 8..(row + 1) * 8]);
        }
    }

    // Demo 2: Rt operations (matmul, neg, add_row)
    {
        let a: Vec<f32> = (0..SIZE).map(|i| (i % 8) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..SIZE)
            .map(|i| if i / 8 == i % 8 { 1.0 } else { 0.0 })
            .collect(); // identity

        let handle_a = client.create_from_slice(f32::as_bytes(&a));
        let handle_b = client.create_from_slice(f32::as_bytes(&b));
        let handle_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; SIZE]));

        let shape = vec![8usize, 8];
        let strides = vec![8usize, 1];
        let arg_a = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides, &shape, LINE_SIZE)
        };
        let arg_b = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_b, &strides, &shape, LINE_SIZE)
        };
        let arg_out = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
        };

        demo_rt_ops::launch::<f32, TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            arg_a,
            arg_b,
            arg_out,
        )
        .expect("rt ops failed");

        let result_bytes = client.read_one(handle_out);
        let result: Vec<f32> = f32::from_bytes(&result_bytes).to_vec();

        println!("\n=== Rt Ops: -(A @ I) + 100 ===");
        println!("(A has rows [0, 0.1, 0.2, ..., 0.7])");
        for row in 0..4 {
            println!("Row {}: {:?}", row, &result[row * 8..(row + 1) * 8]);
        }
    }

    // Demo 3: Rv operations (col scaling, col add)
    {
        let input: Vec<f32> = (0..SIZE).map(|_| 10.0).collect(); // all 10s

        let handle_in = client.create_from_slice(f32::as_bytes(&input));
        let handle_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; SIZE]));

        let shape = vec![8usize, 8];
        let strides = vec![8usize, 1];
        let arg_in = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_in, &strides, &shape, LINE_SIZE)
        };
        let arg_out = unsafe {
            TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
        };

        demo_rv_ops::launch::<f32, TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            arg_in,
            arg_out,
        )
        .expect("rv ops failed");

        let result_bytes = client.read_one(handle_out);
        let result: Vec<f32> = f32::from_bytes(&result_bytes).to_vec();

        println!("\n=== Rv Ops: 10 * 0.5 + (-0.5 + 1) = 5.5 ===");
        println!("Row 0: {:?}", &result[0..8]);
    }

    println!("\nDone!");
}
