//! Benchmark for testing various FusedTile configurations (forward and backward).
//!
//! Tests different (mini_batch_len, head_dim, threads) combinations to find optimal thread counts.
//!
//! Usage:
//!   cargo bench --features rocm,tile-tuning --bench ttt-tile-configs-bench

use std::time::Duration;

use burn::tensor::Tensor;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ttt_core::{GpuAutodiffBackend, GpuBackend};
use ttt_fused::{FusedTttBackend, FusedTttConfig, linear_fused_tile::fused_ttt_tile_forward};

fn device<B: FusedTttBackend>() -> B::Device {
    Default::default()
}

/// Force async operations to complete.
fn sync<B: FusedTttBackend, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

/// Benchmark forward pass for a specific (mini_batch_len, head_dim, threads) configuration.
fn bench_forward<B: FusedTttBackend>(
    c: &mut Criterion,
    mini_batch_len: usize,
    head_dim: usize,
    threads: usize,
    device: &B::Device,
) {
    let batch_size = 4;
    let num_heads = 2;

    let xq = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let xk = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let xv = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let weight = Tensor::<B, 4>::random(
        [batch_size, num_heads, head_dim, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let bias = Tensor::<B, 3>::random(
        [batch_size, num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let token_eta = Tensor::<B, 1>::random(
        [mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ttt_lr_eta = Tensor::<B, 3>::random(
        [batch_size, num_heads, mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ln_weight = Tensor::<B, 2>::random(
        [num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ln_bias = Tensor::<B, 2>::random(
        [num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let epsilon = 1e-6f32;
    let config = FusedTttConfig::new(mini_batch_len, head_dim, epsilon, threads);

    let group_name = format!("fwd_{}x{}", mini_batch_len, head_dim);
    let mut group = c.benchmark_group(&group_name);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(
        (batch_size * num_heads * mini_batch_len * head_dim) as u64,
    ));

    group.bench_function(BenchmarkId::new("threads", threads), |b| {
        b.iter(|| {
            let (output, _, _) = fused_ttt_tile_forward::<B>(
                xq.clone(),
                xk.clone(),
                xv.clone(),
                weight.clone(),
                bias.clone(),
                token_eta.clone(),
                ttt_lr_eta.clone(),
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );
            sync(output);
        })
    });

    group.finish();
}

/// Benchmark backward pass for a specific (mini_batch_len, head_dim, threads) configuration.
/// Uses autodiff to compute gradients.
fn bench_backward(
    c: &mut Criterion,
    mini_batch_len: usize,
    head_dim: usize,
    threads: usize,
    device: &<GpuAutodiffBackend as burn::prelude::Backend>::Device,
) {
    let batch_size = 4;
    let num_heads = 2;

    let xq = Tensor::<GpuAutodiffBackend, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let xk = Tensor::<GpuAutodiffBackend, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let xv = Tensor::<GpuAutodiffBackend, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let weight = Tensor::<GpuAutodiffBackend, 4>::random(
        [batch_size, num_heads, head_dim, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let bias = Tensor::<GpuAutodiffBackend, 3>::random(
        [batch_size, num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let token_eta = Tensor::<GpuAutodiffBackend, 1>::random(
        [mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ttt_lr_eta = Tensor::<GpuAutodiffBackend, 3>::random(
        [batch_size, num_heads, mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let ln_weight = Tensor::<GpuAutodiffBackend, 2>::random(
        [num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();
    let ln_bias = Tensor::<GpuAutodiffBackend, 2>::random(
        [num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
    .require_grad();

    let epsilon = 1e-6f32;
    let config = FusedTttConfig::new(mini_batch_len, head_dim, epsilon, threads);

    let group_name = format!("bwd_{}x{}", mini_batch_len, head_dim);
    let mut group = c.benchmark_group(&group_name);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(
        (batch_size * num_heads * mini_batch_len * head_dim) as u64,
    ));

    group.bench_function(BenchmarkId::new("threads", threads), |b| {
        b.iter(|| {
            let (output, _, _) = fused_ttt_tile_forward::<GpuAutodiffBackend>(
                xq.clone(),
                xk.clone(),
                xv.clone(),
                weight.clone(),
                bias.clone(),
                token_eta.clone(),
                ttt_lr_eta.clone(),
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );
            let loss = output.sum();
            let grads = loss.backward();
            let grad_xq = xq.grad(&grads).unwrap();
            sync(grad_xq);
        })
    });

    group.finish();
}

fn bench_tile_configs(c: &mut Criterion) {
    let device = device::<GpuBackend>();

    let tile_sizes = [(8, 32), (8, 64), (16, 32), (16, 64), (32, 32)];
    let thread_counts = [4, 8, 16, 32, 64, 128, 256];

    for (mini_batch_len, head_dim) in tile_sizes {
        for threads in thread_counts {
            if !FusedTttConfig::is_config_supported(mini_batch_len, head_dim, threads) {
                continue;
            }

            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bench_forward::<GpuBackend>(c, mini_batch_len, head_dim, threads, &device);
            }));

            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bench_backward(c, mini_batch_len, head_dim, threads, &device);
            }));
        }
    }
}

criterion_group!(tile_configs, bench_tile_configs);
criterion_main!(tile_configs);
