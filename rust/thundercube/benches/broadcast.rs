#![allow(non_snake_case)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use pollster::block_on;
use thundercube::{
    LINE_SIZE,
    cube::{load_st_direct, store_st_direct},
    test_utils::{TestRuntime, client, get_strides, upload},
    tiles::{D4, D8, D16, D32, Dim, DimOrOne, Rt, Rv, St},
};

/// Number of iterations per kernel launch to amortize launch overhead
const BENCH_ITERS: u32 = 1024;

// ==================== RT BROADCAST KERNELS ====================

#[cube(launch)]
fn bench_rt_add_row<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    row: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    let mut rv = Rv::<F, C>::new();
    rt.copy_from_array(a);
    rv.copy_from_array(row);

    #[unroll]
    for _ in 0..iters {
        rt.add_row(&rv);
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_rt_mul_row<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    row: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    let mut rv = Rv::<F, C>::new();
    rt.copy_from_array(a);
    rv.copy_from_array(row);

    #[unroll]
    for _ in 0..iters {
        rt.mul_row(&rv);
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_rt_add_col<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    col: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    let mut rv = Rv::<F, R>::new();
    rt.copy_from_array(a);
    rv.copy_from_array(col);

    #[unroll]
    for _ in 0..iters {
        rt.add_col(&rv);
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_rt_mul_col<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    col: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    let mut rv = Rv::<F, R>::new();
    rt.copy_from_array(a);
    rv.copy_from_array(col);

    #[unroll]
    for _ in 0..iters {
        rt.mul_col(&rv);
    }

    rt.copy_to_array(output);
}

// ==================== ST BROADCAST KERNELS ====================

#[cube(launch)]
fn bench_st_add_row<F: Float, R: Dim, C: Dim>(
    a: &Tensor<Line<F>>,
    row: &Array<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    let mut rv = Rv::<F, C>::new();
    rv.copy_from_array(row);
    load_st_direct(a, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        st.add_row(&rv);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

#[cube(launch)]
fn bench_st_mul_row<F: Float, R: Dim, C: Dim>(
    a: &Tensor<Line<F>>,
    row: &Array<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    let mut rv = Rv::<F, C>::new();
    rv.copy_from_array(row);
    load_st_direct(a, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        st.mul_row(&rv);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

#[cube(launch)]
fn bench_st_add_col<F: Float, R: Dim, C: Dim>(
    a: &Tensor<Line<F>>,
    col: &Array<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    let mut rv = Rv::<F, R>::new();
    rv.copy_from_array(col);
    load_st_direct(a, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        st.add_col(&rv);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

#[cube(launch)]
fn bench_st_mul_col<F: Float, R: Dim, C: Dim>(
    a: &Tensor<Line<F>>,
    col: &Array<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    let mut rv = Rv::<F, R>::new();
    rv.copy_from_array(col);
    load_st_direct(a, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        st.mul_col(&rv);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

/// Macro for RT row/col broadcast benchmarks.
/// Uses single thread (CubeDim::new_1d(1)) to isolate register-tile operation performance
/// without shared memory or thread coordination overhead.
macro_rules! bench_rt_broadcast_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty, $vec_dim:ty) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let vec_dim = <$vec_dim>::VALUE;
        let size = rows * cols;

        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_vec: Vec<f32> = (0..vec_dim).map(|i| i as f32).collect();

        let handle_a = upload(&client, &data_a);
        let handle_vec = upload(&client, &data_vec);
        let handle_out = upload(&client, &vec![0.0f32; size]);

        let param_str = format!("{}x{}", rows, cols);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let a =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_a, size, LINE_SIZE) };
                let vec = unsafe {
                    ArrayArg::from_raw_parts::<Line<f32>>(&handle_vec, vec_dim, LINE_SIZE)
                };
                let output =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, size, LINE_SIZE) };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    a,
                    vec,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

/// Macro for ST row/col broadcast benchmarks
macro_rules! bench_st_broadcast_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty, $vec_dim:ty, $threads:expr) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let vec_dim = <$vec_dim>::VALUE;
        let size = rows * cols;

        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_vec: Vec<f32> = (0..vec_dim).map(|i| i as f32).collect();

        let handle_a = upload(&client, &data_a);
        let handle_vec = upload(&client, &data_vec);
        let handle_out = upload(&client, &vec![0.0f32; size]);

        let shape = vec![rows, cols];
        let strides = get_strides(&shape);

        let param_str = format!("{}x{}_t{}", rows, cols, $threads);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let a = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides, &shape, LINE_SIZE)
                };
                let vec = unsafe {
                    ArrayArg::from_raw_parts::<Line<f32>>(&handle_vec, vec_dim, LINE_SIZE)
                };
                let output = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
                };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d($threads),
                    a,
                    vec,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

fn bench_rt_row_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("rt_row_broadcast");
    // Ensure we measure kernel performance, not launch overhead (BENCH_ITERS handles this internally)
    group.measurement_time(Duration::from_secs(10));

    // add_row
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D4, D4, D4);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D8, D8, D8);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D16, D16, D16);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D32, D32, D32);

    // mul_row
    bench_rt_broadcast_impl!(group, "mul_row", bench_rt_mul_row, D4, D4, D4);
    bench_rt_broadcast_impl!(group, "mul_row", bench_rt_mul_row, D8, D8, D8);
    bench_rt_broadcast_impl!(group, "mul_row", bench_rt_mul_row, D16, D16, D16);
    bench_rt_broadcast_impl!(group, "mul_row", bench_rt_mul_row, D32, D32, D32);

    group.finish();
}

fn bench_rt_col_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("rt_col_broadcast");
    group.measurement_time(Duration::from_secs(10));

    // add_col
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D4, D4, D4);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D8, D8, D8);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D16, D16, D16);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D32, D32, D32);

    // mul_col
    bench_rt_broadcast_impl!(group, "mul_col", bench_rt_mul_col, D4, D4, D4);
    bench_rt_broadcast_impl!(group, "mul_col", bench_rt_mul_col, D8, D8, D8);
    bench_rt_broadcast_impl!(group, "mul_col", bench_rt_mul_col, D16, D16, D16);
    bench_rt_broadcast_impl!(group, "mul_col", bench_rt_mul_col, D32, D32, D32);

    group.finish();
}

fn bench_st_row_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("st_row_broadcast");
    group.measurement_time(Duration::from_secs(10));

    // add_row
    bench_st_broadcast_impl!(group, "add_row", bench_st_add_row, D8, D8, D8, 4);
    bench_st_broadcast_impl!(group, "add_row", bench_st_add_row, D16, D16, D16, 16);
    bench_st_broadcast_impl!(group, "add_row", bench_st_add_row, D32, D32, D32, 64);

    // mul_row
    bench_st_broadcast_impl!(group, "mul_row", bench_st_mul_row, D8, D8, D8, 4);
    bench_st_broadcast_impl!(group, "mul_row", bench_st_mul_row, D16, D16, D16, 16);
    bench_st_broadcast_impl!(group, "mul_row", bench_st_mul_row, D32, D32, D32, 64);

    group.finish();
}

fn bench_st_col_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("st_col_broadcast");
    group.measurement_time(Duration::from_secs(10));

    // add_col
    bench_st_broadcast_impl!(group, "add_col", bench_st_add_col, D8, D8, D8, 4);
    bench_st_broadcast_impl!(group, "add_col", bench_st_add_col, D16, D16, D16, 16);
    bench_st_broadcast_impl!(group, "add_col", bench_st_add_col, D32, D32, D32, 64);

    // mul_col
    bench_st_broadcast_impl!(group, "mul_col", bench_st_mul_col, D8, D8, D8, 4);
    bench_st_broadcast_impl!(group, "mul_col", bench_st_mul_col, D16, D16, D16, 16);
    bench_st_broadcast_impl!(group, "mul_col", bench_st_mul_col, D32, D32, D32, 64);

    group.finish();
}

fn bench_asymmetric_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_broadcast");
    group.measurement_time(Duration::from_secs(10));

    // Test asymmetric tiles for row broadcast
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D8, D16, D16);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D16, D8, D8);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D16, D32, D32);
    bench_rt_broadcast_impl!(group, "add_row", bench_rt_add_row, D32, D16, D16);

    // Test asymmetric tiles for col broadcast
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D8, D16, D8);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D16, D8, D16);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D16, D32, D16);
    bench_rt_broadcast_impl!(group, "add_col", bench_rt_add_col, D32, D16, D32);

    group.finish();
}

criterion_group!(
    benches,
    bench_rt_row_broadcast,
    bench_rt_col_broadcast,
    bench_st_row_broadcast,
    bench_st_col_broadcast,
    bench_asymmetric_broadcast,
);
criterion_main!(benches);
