#![allow(non_snake_case)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use pollster::block_on;
use thundercube::{
    LINE_SIZE,
    cube::{ReduceBuf, load_st_direct, sum_cols, sum_cols_plane, sum_rows, sum_rows_plane},
    test_utils::{TestRuntime, client, get_strides, upload},
    tiles::{D4, D8, D16, D32, Dim, DimOrOne, Rt, Rv, St},
};

/// Number of iterations per kernel launch to amortize launch overhead
const BENCH_ITERS: u32 = 1024;

// ==================== RT REDUCTION KERNELS ====================

#[cube(launch)]
fn bench_rt_sum_rows<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);
    let mut result = Rv::<F, R>::new();

    #[unroll]
    for _ in 0..iters {
        rt.sum_rows(&mut result);
    }

    result.copy_to_array(output);
}

#[cube(launch)]
fn bench_rt_sum_cols<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);
    let mut result = Rv::<F, C>::new();

    #[unroll]
    for _ in 0..iters {
        rt.sum_cols(&mut result);
    }

    result.copy_to_array(output);
}

// ==================== ST REDUCTION KERNELS ====================

#[cube(launch)]
fn bench_st_sum_rows<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);
    sync_cube();

    let mut result = Rv::<F, R>::new();

    #[unroll]
    for _ in 0..iters {
        sum_rows_plane::<F, F, R, C>(&st, &mut result);
    }

    if UNIT_POS == 0 {
        result.copy_to_array(output);
    }
}

#[cube(launch)]
fn bench_st_sum_cols<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);
    sync_cube();

    let mut result = Rv::<F, C>::new();

    #[unroll]
    for _ in 0..iters {
        sum_cols_plane::<F, F, R, C>(&st, &mut result);
    }

    if UNIT_POS == 0 {
        result.copy_to_array(output);
    }
}

#[cube(launch)]
fn bench_st_sum_rows_cube<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut buf = ReduceBuf::<F>::new();
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);
    sync_cube();

    let mut result = Rv::<F, R>::new();

    #[unroll]
    for _ in 0..iters {
        sum_rows::<F, F, R, C>(&st, &mut result, &mut buf);
    }

    if UNIT_POS == 0 {
        result.copy_to_array(output);
    }
}

#[cube(launch)]
fn bench_st_sum_cols_cube<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut buf = ReduceBuf::<F>::new();
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);
    sync_cube();

    let mut result = Rv::<F, C>::new();

    #[unroll]
    for _ in 0..iters {
        sum_cols::<F, F, R, C>(&st, &mut result, &mut buf);
    }

    if UNIT_POS == 0 {
        result.copy_to_array(output);
    }
}

/// Macro for RT reduction benchmarks.
/// Uses single thread (CubeDim::new_1d(1)) to isolate register-tile operation performance
/// without shared memory or thread coordination overhead.
macro_rules! bench_rt_reduce_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty, $out_dim:ty) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let out_dim = <$out_dim>::VALUE;
        let size = rows * cols;

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &vec![0.0f32; out_dim]);

        let param_str = format!("{}x{}", rows, cols);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let input =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_in, size, LINE_SIZE) };
                let output = unsafe {
                    ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, out_dim, LINE_SIZE)
                };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    input,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

/// Macro for ST reduction benchmarks (requires more threads for plane reduce)
macro_rules! bench_st_reduce_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty, $out_dim:ty, $threads:expr) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let out_dim = <$out_dim>::VALUE;
        let size = rows * cols;

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &vec![0.0f32; out_dim]);

        let shape = vec![rows, cols];
        let strides = get_strides(&shape);

        let param_str = format!("{}x{}_t{}", rows, cols, $threads);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let input = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_in, &strides, &shape, LINE_SIZE)
                };
                let output = unsafe {
                    ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, out_dim, LINE_SIZE)
                };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d($threads),
                    input,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

fn bench_rt_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("rt_reductions");
    // Ensure we measure kernel performance, not launch overhead (BENCH_ITERS handles this internally)
    group.measurement_time(Duration::from_secs(10));

    // sum_rows (reduce across columns, output has R elements)
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D4, D4, D4);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D8, D8, D8);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D16, D16, D16);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D32, D32, D32);

    // sum_cols (reduce across rows, output has C elements)
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D4, D4, D4);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D8, D8, D8);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D16, D16, D16);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D32, D32, D32);

    group.finish();
}

fn bench_st_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("st_reductions");
    group.measurement_time(Duration::from_secs(10));

    // Note: ST reductions use plane_reduce, which requires threads >= PLANE_DIM (32)
    // sum_rows
    bench_st_reduce_impl!(group, "sum_rows", bench_st_sum_rows, D8, D8, D8, 32);
    bench_st_reduce_impl!(group, "sum_rows", bench_st_sum_rows, D8, D8, D8, 64);
    bench_st_reduce_impl!(group, "sum_rows", bench_st_sum_rows, D16, D16, D16, 32);
    bench_st_reduce_impl!(group, "sum_rows", bench_st_sum_rows, D16, D16, D16, 64);
    bench_st_reduce_impl!(group, "sum_rows", bench_st_sum_rows, D32, D32, D32, 64);

    // sum_cols
    bench_st_reduce_impl!(group, "sum_cols", bench_st_sum_cols, D8, D8, D8, 32);
    bench_st_reduce_impl!(group, "sum_cols", bench_st_sum_cols, D8, D8, D8, 64);
    bench_st_reduce_impl!(group, "sum_cols", bench_st_sum_cols, D16, D16, D16, 32);
    bench_st_reduce_impl!(group, "sum_cols", bench_st_sum_cols, D16, D16, D16, 64);
    bench_st_reduce_impl!(group, "sum_cols", bench_st_sum_cols, D32, D32, D32, 64);

    group.finish();
}

fn bench_asymmetric_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_reductions");
    group.measurement_time(Duration::from_secs(10));

    // Asymmetric tiles for RT reductions
    // sum_rows: output dim = R
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D8, D16, D8);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D16, D8, D16);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D16, D32, D16);
    bench_rt_reduce_impl!(group, "sum_rows", bench_rt_sum_rows, D32, D16, D32);

    // sum_cols: output dim = C
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D8, D16, D16);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D16, D8, D8);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D16, D32, D32);
    bench_rt_reduce_impl!(group, "sum_cols", bench_rt_sum_cols, D32, D16, D16);

    group.finish();
}

fn bench_thread_scaling_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_thread_scaling");
    group.measurement_time(Duration::from_secs(10));

    // Test 16x16 ST reduction with different thread counts
    for threads in [32u32, 64, 128] {
        let client = client();
        let rows = 16usize;
        let cols = 16usize;
        let size = rows * cols;

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &vec![0.0f32; rows]);

        let shape = vec![rows, cols];
        let strides = get_strides(&shape);

        let param_str = format!("16x16_t{}", threads);

        let elements = size * (BENCH_ITERS as usize);
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_with_input(BenchmarkId::new("sum_rows", &param_str), &(), |b, _| {
            b.iter(|| {
                let input = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_in, &strides, &shape, LINE_SIZE)
                };
                let output =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, rows, LINE_SIZE) };

                bench_st_sum_rows::launch::<f32, D16, D16, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(threads),
                    input,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rt_reductions,
    bench_st_reductions,
    bench_asymmetric_reductions,
    bench_thread_scaling_reductions,
);
criterion_main!(benches);
