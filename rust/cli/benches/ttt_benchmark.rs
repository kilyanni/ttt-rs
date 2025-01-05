//! Generic benchmarks for TTT inner models and full models.
//!
//! This benchmark suite provides:
//! - Inner model benchmarks (forward/backward) - just the TTT learner
//! - Full model benchmarks (forward/backward) - complete text generation model
//!
//! # Usage
//!
//! Run all benchmarks:
//!   cargo bench --features rocm --bench ttt-bench-criterion
//!
//! Run specific benchmark:
//!   cargo bench --features rocm --bench ttt-bench-criterion -- inner_forward_linear
//!   cargo bench --features rocm --bench ttt-bench-criterion -- full_forward

use std::{sync::Arc, time::Duration};

use burn::prelude::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use paste::paste;
use ttt_config::{InnerModel, MixPattern, ModelArch, TTTConfig};
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, Qkv, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearAdam,
    TTTMLP, TTTMLP2, TTTMLP3, TTTMLP4, config::ModelConfig,
};
use ttt_data::TrainingTextGenerationBatch;
use ttt_fused::{FusedNaive, FusedNaiveMulti, FusedTile, FusedTileMulti, FusedTttBackend};
use ttt_training::{TTTTextGenerationConfig, TTTTextGenerationModel};

pub fn device<B: FusedTttBackend>() -> B::Device {
    Default::default()
}

/// Force async operations to complete before returning.
fn sync<B: Backend, const D: usize>(tensor: Tensor<B, D>) {
    B::sync(&tensor.device()).unwrap();
}

#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub name: &'static str,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_intermediate: usize,
    pub mini_batch_size: usize,
}

impl BenchConfig {
    pub const fn m12() -> Self {
        Self {
            name: "12m",
            hidden_size: 256,
            num_heads: 4,
            num_layers: 6,
            mlp_intermediate: 512,
            mini_batch_size: 16,
        }
    }

    pub const fn m60() -> Self {
        Self {
            name: "60m",
            hidden_size: 512,
            num_heads: 8,
            num_layers: 6,
            mlp_intermediate: 768,
            mini_batch_size: 16,
        }
    }

    pub const fn m125() -> Self {
        Self {
            name: "125m",
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
            mlp_intermediate: 2048,
            mini_batch_size: 16,
        }
    }

    pub const fn m350() -> Self {
        Self {
            name: "350m",
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 24,
            mlp_intermediate: 2736,
            mini_batch_size: 16,
        }
    }

    pub const fn m760() -> Self {
        Self {
            name: "760m",
            hidden_size: 1536,
            num_heads: 16,
            num_layers: 24,
            mlp_intermediate: 4096,
            mini_batch_size: 16,
        }
    }

    pub const fn b1() -> Self {
        Self {
            name: "1b",
            hidden_size: 2048,
            num_heads: 32,
            num_layers: 24,
            mlp_intermediate: 5504,
            mini_batch_size: 16,
        }
    }

    pub fn to_model_config(&self, vocab_size: usize) -> ModelConfig {
        self.to_model_config_with_threads(vocab_size, None)
    }

    pub fn to_model_config_with_threads(
        &self,
        vocab_size: usize,
        threads: Option<usize>,
    ) -> ModelConfig {
        let arch = Arc::new(ModelArch {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_layers,
            num_heads: self.num_heads,
            intermediate_size: self.mlp_intermediate,
            vocab_size,
        });
        let ttt = Arc::new(TTTConfig {
            mini_batch_size: self.mini_batch_size,
            threads,
            ..TTTConfig::default()
        });
        ModelConfig::new(arch, ttt)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeParams {
    pub batch_size: usize,
    pub seq_length: usize,
    pub vocab_size: usize,
}

impl RuntimeParams {
    pub fn new(batch_size: usize, seq_length: usize, vocab_size: usize) -> Self {
        Self {
            batch_size,
            seq_length,
            vocab_size,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.batch_size * self.seq_length
    }

    pub fn id(&self) -> String {
        format!("b{}_s{}", self.batch_size, self.seq_length)
    }
}

fn create_inner_inputs<B: FusedTttBackend>(
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) -> TTTInputsInner<B> {
    let batch_size = params.batch_size;
    let num_heads = config.num_heads;
    let seq_len = params.seq_length;
    let head_dim = config.head_dim();

    let xq = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );
    let xk = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );
    let xv = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );

    let token_eta = Tensor::arange(1..(seq_len as i64 + 1), device)
        .float()
        .recip();

    let ttt_lr_eta = Tensor::random(
        [batch_size, num_heads, seq_len],
        burn::tensor::Distribution::Uniform(0.01, 0.05),
        device,
    );

    TTTInputsInner {
        qkv: Qkv { xq, xk, xv },
        token_eta,
        ttt_lr_eta,
        start_idx: 0,
    }
}

fn random_logits<B: FusedTttBackend>(
    params: &RuntimeParams,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::random(
        [params.batch_size, params.seq_length],
        burn::tensor::Distribution::Uniform(0.0, params.vocab_size as f64 - 1.0),
        device,
    )
}

fn create_training_batch<B: FusedTttBackend>(
    params: &RuntimeParams,
    device: &B::Device,
) -> TrainingTextGenerationBatch<B> {
    let tokens_inputs = random_logits::<B>(params, device);
    let targets = random_logits::<B>(params, device);
    let mask_pad = Tensor::<B, 2, Bool>::ones([params.batch_size, params.seq_length], device);

    TrainingTextGenerationBatch {
        tokens_inputs,
        targets,
        mask_pad,
    }
}

/// Benchmark inner model forward pass
fn bench_inner_forward<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let model_config = config.to_model_config(params.vocab_size);
    let inner_config = Arc::new(Inner::Config::default());

    let inner: Inner = Inner::new(&model_config, &inner_config, device);

    let group_name = format!("inner_forward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(20));

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("forward", &bench_id), |b| {
        b.iter_batched(
            || {
                let inputs = create_inner_inputs::<B>(config, params, device);
                let state = inner.init_state(params.batch_size);
                (inputs, state)
            },
            |(inputs, mut state)| {
                let output = inner.forward(&mut state, inputs);
                sync(output);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark inner model backward pass (forward + backward)
fn bench_inner_backward<
    B: burn::tensor::backend::AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B>,
>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let model_config = config.to_model_config(params.vocab_size);
    let inner_config = Arc::new(Inner::Config::default());

    let inner: Inner = Inner::new(&model_config, &inner_config, device);

    let group_name = format!("inner_backward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(20));

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("backward", &bench_id), |b| {
        b.iter_batched(
            || {
                let inputs = create_inner_inputs::<B>(config, params, device);
                let state = inner.init_state(params.batch_size);
                (inputs, state)
            },
            |(inputs, mut state)| {
                let output = inner.forward(&mut state, inputs);
                let _grads = output.sum().backward();
                // Sync all GPU operations (including backward)
                let _ = B::sync(device);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark full model forward pass
fn bench_full_forward<B: FusedTttBackend>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
    inner_type: InnerModel,
    inner_name: &str,
) {
    let model_config = config.to_model_config(params.vocab_size);
    let text_gen_config = TTTTextGenerationConfig::new_testing(model_config);
    let mix = MixPattern::uniform(inner_type);
    let model: TTTTextGenerationModel<B> = text_gen_config.init(&mix, device);

    let group_name = format!("full_forward_{}", inner_name);
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(20));

    let bench_id = format!("{}_{}", config.name, params.id());

    group.bench_function(BenchmarkId::new("forward", &bench_id), |b| {
        b.iter_batched(
            || random_logits::<B>(params, device),
            |input| {
                let output = model.forward_inference(input);
                sync(output);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark full model backward pass (forward + backward)
fn bench_full_backward<B: burn::tensor::backend::AutodiffBackend + FusedTttBackend>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
    inner_type: InnerModel,
    inner_name: &str,
) {
    let model_config = config.to_model_config(params.vocab_size);
    let text_gen_config = TTTTextGenerationConfig::new_testing(model_config);
    let mix = MixPattern::uniform(inner_type);
    let model: TTTTextGenerationModel<B> = text_gen_config.init(&mix, device);

    let group_name = format!("full_backward_{}", inner_name);
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(20));

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("backward", &bench_id), |b| {
        b.iter_batched(
            || create_training_batch::<B>(params, device),
            |batch| {
                let output = model.forward_training(batch);
                let _grads = output.loss.backward();
                // Sync all GPU operations (including backward)
                let _ = B::sync(device);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

// Benchmark generation macros

/// Runtime parameters used across all benchmarks
const BENCH_PARAMS: &[RuntimeParams] = &[
    RuntimeParams {
        batch_size: 1,
        seq_length: 2048,
        vocab_size: 1000,
    },
    // RuntimeParams {
    //     batch_size: 1,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 4,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 16,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 64,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 256,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 1024,
    //     seq_length: 32,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 16,
    //     seq_length: 8192,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 4,
    //     seq_length: 8192,
    //     vocab_size: 1000,
    // },
];

const BENCH_CONFIGS: &[BenchConfig] = &[
    // BenchConfig::m12(),
    // BenchConfig::m60(),
    BenchConfig::m125(),
    // BenchConfig::m350(),
    // BenchConfig::m760(),
    // BenchConfig::b1(),
];

/// Map benchmark suffix to InnerModel enum variant
macro_rules! suffix_to_inner_model {
    (linear) => {
        InnerModel::Linear
    };
    (linear_adam) => {
        InnerModel::LinearAdam
    };
    (mlp) => {
        InnerModel::Mlp
    };
    (mlp2) => {
        InnerModel::Mlp2
    };
    (mlp3) => {
        InnerModel::Mlp3
    };
    (mlp4) => {
        InnerModel::Mlp4
    };
    (fused_naive) => {
        InnerModel::FusedNaiveLinear
    };
    (fused_naive_multi) => {
        InnerModel::FusedNaiveMultiLinear
    };
    (fused_linear_tile) => {
        InnerModel::FusedTileLinear
    };
    (fused_linear_tile_multi) => {
        InnerModel::FusedTileMultiLinear
    };
    (fused_linear_tile_d2d_streaming) => {
        InnerModel::D2dStreamingLinear
    };
    (fused_linear_tile_ptr_streaming) => {
        InnerModel::PtrStreamingLinear
    };
}

/// Generate a single benchmark entry function
macro_rules! define_bench {
    // Inner forward benchmarks
    (forward, inner, $fn_name:ident, $inner:ty) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    bench_inner_forward::<GpuBackend, $inner>(c, &config, params, &device);
                }
            }
        }
    };
    // Inner backward benchmarks
    (backward, inner, $fn_name:ident, $inner:ty) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuAutodiffBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    bench_inner_backward::<GpuAutodiffBackend, $inner>(c, &config, params, &device);
                }
            }
        }
    };
    // Full forward benchmarks
    (forward, full, $fn_name:ident, $inner_model:expr, $inner_name:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    bench_full_forward::<GpuBackend>(
                        c,
                        &config,
                        params,
                        &device,
                        $inner_model,
                        $inner_name,
                    );
                }
            }
        }
    };
    // Full backward benchmarks
    (backward, full, $fn_name:ident, $inner_model:expr, $inner_name:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuAutodiffBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    bench_full_backward::<GpuAutodiffBackend>(
                        c,
                        &config,
                        params,
                        &device,
                        $inner_model,
                        $inner_name,
                    );
                }
            }
        }
    };
}

/// Generate all 4 benchmark variants (inner/full × forward/backward) for a model
macro_rules! define_model_benches {
    ($suffix:ident, $inner:ty) => {
        paste! {
            define_bench!(forward, inner, [<bench_inner_forward_ $suffix>], $inner);
            define_bench!(backward, inner, [<bench_inner_backward_ $suffix>], $inner);
            define_bench!(forward, full, [<bench_full_forward_ $suffix>], suffix_to_inner_model!($suffix), stringify!($suffix));
            define_bench!(backward, full, [<bench_full_backward_ $suffix>], suffix_to_inner_model!($suffix), stringify!($suffix));
        }
    };
}

/// Generate all benchmarks and criterion groups for the given models
macro_rules! define_all_benches {
    ($($suffix:ident => $inner:ty),* $(,)?) => {
        // Generate all benchmark functions
        $(define_model_benches!($suffix, $inner);)*

        paste! {
            // Generate criterion groups
            criterion_group!(inner_forward, $([<bench_inner_forward_ $suffix>]),*);
            criterion_group!(inner_backward, $([<bench_inner_backward_ $suffix>]),*);
            criterion_group!(full_forward, $([<bench_full_forward_ $suffix>]),*);
            criterion_group!(full_backward, $([<bench_full_backward_ $suffix>]),*);
        }
    };
}

#[cfg(not(feature = "streaming"))]
define_all_benches!(
    linear => TTTLinear<_>,
    linear_adam => TTTLinearAdam<_>,
    mlp => TTTMLP<_>,
    mlp2 => TTTMLP2<_>,
    mlp3 => TTTMLP3<_>,
    mlp4 => TTTMLP4<_>,
    fused_naive => FusedNaive<_>,
    fused_naive_multi => FusedNaiveMulti<_>,
    fused_linear_tile => FusedTile<_>,
    fused_linear_tile_multi => FusedTileMulti<_>,
);

#[cfg(feature = "streaming")]
define_all_benches!(
    linear => TTTLinear<_>,
    linear_adam => TTTLinearAdam<_>,
    mlp => TTTMLP<_>,
    mlp2 => TTTMLP2<_>,
    mlp3 => TTTMLP3<_>,
    mlp4 => TTTMLP4<_>,
    fused_naive => FusedNaive<_>,
    fused_naive_multi => FusedNaiveMulti<_>,
    fused_linear_tile => FusedTile<_>,
    fused_linear_tile_multi => FusedTileMulti<_>,
    fused_linear_tile_d2d_streaming => ttt_fused::FusedTileD2dStreaming<_>,
    fused_linear_tile_ptr_streaming => ttt_fused::FusedTilePtrStreaming<_>,
);

criterion_main!(inner_forward, inner_backward, full_forward, full_backward);
