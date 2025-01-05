#![allow(non_snake_case)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use pollster::block_on;
use thundercube::{
    LINE_SIZE,
    cube::{load_st_direct, load_st_transpose, store_st_direct, store_st_transpose},
    test_utils::{TestRuntime, client, get_strides, upload},
    tiles::{D4, D8, D16, D32, D64, Dim, DimOrOne, St},
};

/// Number of iterations per kernel launch to amortize launch overhead
const BENCH_ITERS: u32 = 1024;

/// Benchmark kernel for direct load (global -> shared -> global)
#[cube(launch)]
fn bench_load_direct<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();

    #[unroll]
    for _ in 0..iters {
        load_st_direct(input, &mut st, 0, 0, 0);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

/// Benchmark kernel for transpose load (global -> shared transposed -> global)
#[cube(launch)]
fn bench_load_transpose<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();

    #[unroll]
    for _ in 0..iters {
        load_st_transpose(input, &mut st, 0, 0, 0);
    }

    store_st_direct(&st, output, 0, 0, 0);
}

/// Benchmark kernel for direct store
#[cube(launch)]
fn bench_store_direct<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        store_st_direct(&st, output, 0, 0, 0);
    }
}

/// Benchmark kernel for transpose store (global -> shared -> global transposed)
#[cube(launch)]
fn bench_store_transpose<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();
    load_st_direct(input, &mut st, 0, 0, 0);

    #[unroll]
    for _ in 0..iters {
        store_st_transpose(&st, output, 0, 0, 0);
    }
}

/// Benchmark kernel for round-trip load-transpose-store-transpose (double transpose = identity)
#[cube(launch)]
fn bench_round_trip_transpose<F: Float, R: Dim, C: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut st = St::<F, R, C>::new();

    #[unroll]
    for _ in 0..iters {
        load_st_transpose(input, &mut st, 0, 0, 0);
        store_st_transpose(&st, output, 0, 0, 0);
    }
}

/// Macro to run load/store benchmarks with specific tile dimensions
macro_rules! bench_load_store_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty, $num_threads:expr) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;

        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &data);

        let shape = vec![rows, cols];
        let strides = get_strides(&shape);

        // Bytes per iteration * number of iterations
        let bytes = rows * cols * std::mem::size_of::<f32>() * (BENCH_ITERS as usize);
        let param_str = format!("{}x{}_t{}", rows, cols, $num_threads);

        $c.throughput(Throughput::Bytes(bytes as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let input = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_in, &strides, &shape, LINE_SIZE)
                };
                let output = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
                };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d($num_threads),
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

fn bench_load_direct_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_direct");
    // Ensure we measure kernel performance, not launch overhead (BENCH_ITERS handles this internally)
    group.measurement_time(Duration::from_secs(10));

    // Square tiles
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D4, D4, 1);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D8, D8, 4);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D16, D16, 16);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D32, D32, 64);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D64, D64, 64);

    // Asymmetric
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D8, D16, 8);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D16, D8, 8);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D16, D32, 32);
    bench_load_store_impl!(group, "load_direct", bench_load_direct, D32, D16, 32);

    group.finish();
}

fn bench_load_transpose_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_transpose");
    group.measurement_time(Duration::from_secs(10));

    bench_load_store_impl!(group, "load_transpose", bench_load_transpose, D4, D4, 1);
    bench_load_store_impl!(group, "load_transpose", bench_load_transpose, D8, D8, 4);
    bench_load_store_impl!(group, "load_transpose", bench_load_transpose, D16, D16, 16);
    bench_load_store_impl!(group, "load_transpose", bench_load_transpose, D32, D32, 64);
    bench_load_store_impl!(group, "load_transpose", bench_load_transpose, D64, D64, 64);

    group.finish();
}

fn bench_store_direct_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_direct");
    group.measurement_time(Duration::from_secs(10));

    bench_load_store_impl!(group, "store_direct", bench_store_direct, D4, D4, 1);
    bench_load_store_impl!(group, "store_direct", bench_store_direct, D8, D8, 4);
    bench_load_store_impl!(group, "store_direct", bench_store_direct, D16, D16, 16);
    bench_load_store_impl!(group, "store_direct", bench_store_direct, D32, D32, 64);
    bench_load_store_impl!(group, "store_direct", bench_store_direct, D64, D64, 64);

    group.finish();
}

fn bench_store_transpose_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_transpose");
    group.measurement_time(Duration::from_secs(10));

    bench_load_store_impl!(group, "store_transpose", bench_store_transpose, D4, D4, 1);
    bench_load_store_impl!(group, "store_transpose", bench_store_transpose, D8, D8, 4);
    bench_load_store_impl!(
        group,
        "store_transpose",
        bench_store_transpose,
        D16,
        D16,
        16
    );
    bench_load_store_impl!(
        group,
        "store_transpose",
        bench_store_transpose,
        D32,
        D32,
        64
    );
    bench_load_store_impl!(
        group,
        "store_transpose",
        bench_store_transpose,
        D64,
        D64,
        64
    );

    group.finish();
}

fn bench_round_trip_transpose_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip_transpose");
    group.measurement_time(Duration::from_secs(10));

    bench_load_store_impl!(group, "round_trip", bench_round_trip_transpose, D4, D4, 1);
    bench_load_store_impl!(group, "round_trip", bench_round_trip_transpose, D8, D8, 4);
    bench_load_store_impl!(
        group,
        "round_trip",
        bench_round_trip_transpose,
        D16,
        D16,
        16
    );
    bench_load_store_impl!(
        group,
        "round_trip",
        bench_round_trip_transpose,
        D32,
        D32,
        64
    );
    bench_load_store_impl!(
        group,
        "round_trip",
        bench_round_trip_transpose,
        D64,
        D64,
        64
    );

    group.finish();
}

fn bench_thread_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.measurement_time(Duration::from_secs(10));

    // Test 32x32 tiles with different thread counts
    for num_threads in [1u32, 4, 16, 32, 64, 128] {
        let client = client();
        let rows = 32usize;
        let cols = 32usize;

        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &data);

        let shape = vec![rows, cols];
        let strides = get_strides(&shape);

        let bytes = rows * cols * std::mem::size_of::<f32>() * (BENCH_ITERS as usize);
        let param_str = format!("32x32_t{}", num_threads);

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("load_direct", &param_str), &(), |b, _| {
            b.iter(|| {
                let input = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_in, &strides, &shape, LINE_SIZE)
                };
                let output = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_out, &strides, &shape, LINE_SIZE)
                };

                bench_load_direct::launch::<f32, D32, D32, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(num_threads),
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
    bench_load_direct_tile_sizes,
    bench_load_transpose_tile_sizes,
    bench_store_direct_tile_sizes,
    bench_store_transpose_tile_sizes,
    bench_round_trip_transpose_tile_sizes,
    bench_thread_count_scaling,
);
criterion_main!(benches);
