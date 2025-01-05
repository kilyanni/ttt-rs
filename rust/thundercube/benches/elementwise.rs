#![allow(non_snake_case)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use pollster::block_on;
use thundercube::{
    LINE_SIZE,
    test_utils::{TestRuntime, client, upload},
    tiles::{D4, D8, D16, D32, Dim, DimOrOne, Rt},
};

/// Number of iterations per kernel launch to amortize launch overhead
const BENCH_ITERS: u32 = 1024;

// ==================== UNARY OPERATION KERNELS ====================

#[cube(launch)]
fn bench_exp<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);

    #[unroll]
    for _ in 0..iters {
        rt.exp();
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_tanh<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);

    #[unroll]
    for _ in 0..iters {
        rt.tanh();
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_sigmoid<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);

    #[unroll]
    for _ in 0..iters {
        rt.sigmoid();
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_gelu<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);

    #[unroll]
    for _ in 0..iters {
        rt.gelu();
    }

    rt.copy_to_array(output);
}

#[cube(launch)]
fn bench_sqrt<F: Float, R: Dim, C: Dim>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt = Rt::<F, R, C>::new();
    rt.copy_from_array(input);

    #[unroll]
    for _ in 0..iters {
        rt.sqrt();
    }

    rt.copy_to_array(output);
}

// ==================== BINARY OPERATION KERNELS ====================

#[cube(launch)]
fn bench_add<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    b: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt_a = Rt::<F, R, C>::new();
    let mut rt_b = Rt::<F, R, C>::new();
    rt_a.copy_from_array(a);
    rt_b.copy_from_array(b);

    #[unroll]
    for _ in 0..iters {
        rt_a.add(&rt_b);
    }

    rt_a.copy_to_array(output);
}

#[cube(launch)]
fn bench_mul<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    b: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt_a = Rt::<F, R, C>::new();
    let mut rt_b = Rt::<F, R, C>::new();
    rt_a.copy_from_array(a);
    rt_b.copy_from_array(b);

    #[unroll]
    for _ in 0..iters {
        rt_a.mul(&rt_b);
    }

    rt_a.copy_to_array(output);
}

#[cube(launch)]
fn bench_div<F: Float, R: Dim, C: Dim>(
    a: &Array<Line<F>>,
    b: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] iters: u32,
) {
    let mut rt_a = Rt::<F, R, C>::new();
    let mut rt_b = Rt::<F, R, C>::new();
    rt_a.copy_from_array(a);
    rt_b.copy_from_array(b);

    #[unroll]
    for _ in 0..iters {
        rt_a.div(&rt_b);
    }

    rt_a.copy_to_array(output);
}

/// Macro to run unary element-wise benchmarks.
/// Uses single thread (CubeDim::new_1d(1)) to isolate register-tile operation performance
/// without shared memory or thread coordination overhead.
macro_rules! bench_unary_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let size = rows * cols;

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let handle_in = upload(&client, &data);
        let handle_out = upload(&client, &vec![0.0f32; size]);

        let param_str = format!("{}x{}", rows, cols);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let input =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_in, size, LINE_SIZE) };
                let output =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, size, LINE_SIZE) };

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

/// Macro to run binary element-wise benchmarks
macro_rules! bench_binary_impl {
    ($c:expr, $group_name:expr, $kernel:ident, $rows:ty, $cols:ty) => {{
        let client = client();

        let rows = <$rows>::VALUE;
        let cols = <$cols>::VALUE;
        let size = rows * cols;

        let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let handle_a = upload(&client, &data_a);
        let handle_b = upload(&client, &data_b);
        let handle_out = upload(&client, &vec![0.0f32; size]);

        let param_str = format!("{}x{}", rows, cols);

        // Elements per iteration * number of iterations
        let elements = size * (BENCH_ITERS as usize);
        $c.throughput(Throughput::Elements(elements as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let a =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_a, size, LINE_SIZE) };
                let b =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_b, size, LINE_SIZE) };
                let output =
                    unsafe { ArrayArg::from_raw_parts::<Line<f32>>(&handle_out, size, LINE_SIZE) };

                $kernel::launch::<f32, $rows, $cols, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    a,
                    b,
                    output,
                    BENCH_ITERS,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

fn bench_unary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_ops");
    // Ensure we measure kernel performance, not launch overhead (BENCH_ITERS handles this internally)
    group.measurement_time(Duration::from_secs(10));

    // exp
    bench_unary_impl!(group, "exp", bench_exp, D4, D4);
    bench_unary_impl!(group, "exp", bench_exp, D8, D8);
    bench_unary_impl!(group, "exp", bench_exp, D16, D16);
    bench_unary_impl!(group, "exp", bench_exp, D32, D32);

    // tanh
    bench_unary_impl!(group, "tanh", bench_tanh, D4, D4);
    bench_unary_impl!(group, "tanh", bench_tanh, D8, D8);
    bench_unary_impl!(group, "tanh", bench_tanh, D16, D16);
    bench_unary_impl!(group, "tanh", bench_tanh, D32, D32);

    // sigmoid
    bench_unary_impl!(group, "sigmoid", bench_sigmoid, D4, D4);
    bench_unary_impl!(group, "sigmoid", bench_sigmoid, D8, D8);
    bench_unary_impl!(group, "sigmoid", bench_sigmoid, D16, D16);
    bench_unary_impl!(group, "sigmoid", bench_sigmoid, D32, D32);

    // gelu
    bench_unary_impl!(group, "gelu", bench_gelu, D4, D4);
    bench_unary_impl!(group, "gelu", bench_gelu, D8, D8);
    bench_unary_impl!(group, "gelu", bench_gelu, D16, D16);
    bench_unary_impl!(group, "gelu", bench_gelu, D32, D32);

    // sqrt
    bench_unary_impl!(group, "sqrt", bench_sqrt, D4, D4);
    bench_unary_impl!(group, "sqrt", bench_sqrt, D8, D8);
    bench_unary_impl!(group, "sqrt", bench_sqrt, D16, D16);
    bench_unary_impl!(group, "sqrt", bench_sqrt, D32, D32);

    group.finish();
}

fn bench_binary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops");
    group.measurement_time(Duration::from_secs(10));

    // add
    bench_binary_impl!(group, "add", bench_add, D4, D4);
    bench_binary_impl!(group, "add", bench_add, D8, D8);
    bench_binary_impl!(group, "add", bench_add, D16, D16);
    bench_binary_impl!(group, "add", bench_add, D32, D32);

    // mul
    bench_binary_impl!(group, "mul", bench_mul, D4, D4);
    bench_binary_impl!(group, "mul", bench_mul, D8, D8);
    bench_binary_impl!(group, "mul", bench_mul, D16, D16);
    bench_binary_impl!(group, "mul", bench_mul, D32, D32);

    // div
    bench_binary_impl!(group, "div", bench_div, D4, D4);
    bench_binary_impl!(group, "div", bench_div, D8, D8);
    bench_binary_impl!(group, "div", bench_div, D16, D16);
    bench_binary_impl!(group, "div", bench_div, D32, D32);

    group.finish();
}

fn bench_tile_size_scaling_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_tile_scaling");
    group.measurement_time(Duration::from_secs(10));

    // Compare gelu across tile sizes (most compute-intensive unary op)
    bench_unary_impl!(group, "gelu", bench_gelu, D4, D4);
    bench_unary_impl!(group, "gelu", bench_gelu, D8, D8);
    bench_unary_impl!(group, "gelu", bench_gelu, D16, D16);
    bench_unary_impl!(group, "gelu", bench_gelu, D32, D32);

    // Asymmetric tiles
    bench_unary_impl!(group, "gelu", bench_gelu, D8, D16);
    bench_unary_impl!(group, "gelu", bench_gelu, D16, D8);
    bench_unary_impl!(group, "gelu", bench_gelu, D16, D32);
    bench_unary_impl!(group, "gelu", bench_gelu, D32, D16);

    group.finish();
}

criterion_group!(
    benches,
    bench_unary_ops,
    bench_binary_ops,
    bench_tile_size_scaling_unary
);
criterion_main!(benches);
