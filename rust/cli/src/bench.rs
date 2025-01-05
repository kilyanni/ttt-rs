//! Simple TTT benchmark CLI for parameter sweeps.
//!
//! Usage:
//!   ttt-bench --model-size 125m --ttt-type linear
//!   ttt-bench --model-size 125m --ttt-type fused-naive-linear --batch 8 --json
//!   ttt-bench --model-size 125m --ttt-type linear --dtype bfloat16
//!   ttt-bench --model-size 125m --ttt-type "4*linear,mlp"   # mix pattern

use std::{sync::Arc, time::Instant};

use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use tracing_subscriber::EnvFilter;
use ttt_config::{InnerModel, MixPattern, ModelArch, ModelSize, TTTConfig};
use ttt_core::{
    TTTInnerModel,
    config::ModelConfig,
    test_utils::{TestDims, generate_test_inputs},
};
use ttt_fused::FusedTttBackend;
use ttt_layer::{ModelConfigModelExt, TTTModel, dispatch_ttt_layer_type};

// Backend types for runtime dtype selection
#[cfg(feature = "rocm")]
type BackendF32 = burn::backend::Rocm<f32>;
#[cfg(feature = "rocm")]
type BackendBf16 = burn::backend::Rocm<half::bf16>;

#[cfg(feature = "cuda")]
type BackendF32 = burn::backend::Cuda<f32>;
#[cfg(feature = "cuda")]
type BackendBf16 = burn::backend::Cuda<half::bf16>;

#[cfg(feature = "wgpu")]
type BackendF32 = burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
type BackendBf16 = burn::backend::Wgpu; // wgpu doesn't distinguish

type AutodiffF32 = burn::backend::Autodiff<BackendF32>;
type AutodiffBf16 = burn::backend::Autodiff<BackendBf16>;

#[derive(Parser, Debug)]
#[command(name = "ttt-bench", about = "TTT benchmark for parameter sweeps")]
struct Args {
    #[arg(long, default_value = "125m")]
    model_size: ModelSize,

    /// Layer type mix pattern. Single type (e.g. "linear") or mix (e.g. "4*linear,mlp").
    #[arg(long, default_value = "linear")]
    ttt_type: MixPattern,

    #[arg(long, default_value = "false")]
    backward: bool,

    #[arg(long, default_value = "1")]
    batch: usize,

    #[arg(long, default_value = "2048")]
    seq_len: usize,

    #[arg(long, default_value = "16")]
    mini_batch: usize,

    #[arg(long, default_value = "3")]
    warmup: usize,

    #[arg(long, default_value = "5")]
    repeats: usize,

    #[arg(long, default_value = "1")]
    checkpoint_interval: usize,

    #[arg(long, default_value = "false")]
    json: bool,

    #[arg(long, default_value = "float32")]
    dtype: String,

    /// Benchmark just the inner TTT model (single layer) instead of the full model
    #[arg(long, default_value = "false")]
    inner_only: bool,

    /// Ignored; visible in process list for identification
    #[arg(long, default_value = "")]
    comment: String,
}

#[derive(Serialize)]
struct BenchResult {
    implementation: &'static str,
    model_size: String,
    ttt_type: String,
    backward: bool,
    inner_only: bool,
    batch: usize,
    seq_len: usize,
    mini_batch: usize,
    checkpoint_interval: usize,
    dtype: &'static str,
    time_ms: f64,
    throughput: f64,
}

fn bench_fwd<B: FusedTttBackend>(
    args: &Args,
    config: &ModelConfig,
    mix: &MixPattern,
    device: &B::Device,
) -> f64 {
    let model: TTTModel<B> = config.init_with_mix(mix, device);
    let input_ids = Tensor::<B, 2, Int>::ones([args.batch, args.seq_len], device);

    for _ in 0..args.warmup {
        let _logits = model.forward(input_ids.clone(), 0);
        let _ = B::sync(device);
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let start = Instant::now();
        let _logits = model.forward(input_ids.clone(), 0);
        let _ = B::sync(device);
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn bench_bwd<B: burn::tensor::backend::AutodiffBackend + FusedTttBackend>(
    args: &Args,
    config: &ModelConfig,
    mix: &MixPattern,
    device: &B::Device,
) -> f64 {
    let model: TTTModel<B> = config.init_with_mix(mix, device);
    let input_ids = Tensor::<B, 2, Int>::ones([args.batch, args.seq_len], device);

    for _ in 0..args.warmup {
        let _grads = model.forward(input_ids.clone(), 0).sum().backward();
        let _ = B::sync(device);
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let start = Instant::now();
        let _grads = model.forward(input_ids.clone(), 0).sum().backward();
        let _ = B::sync(device);
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn bench_fwd_inner<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
    args: &Args,
    config: &ModelConfig,
    device: &B::Device,
) -> f64 {
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(config, &inner_config, device);
    let dims = TestDims {
        batch_size: args.batch,
        num_heads: config.arch.num_heads,
        head_dim: config.head_dim(),
        seq_len: args.seq_len,
        mini_batch_size: args.mini_batch,
        iterations: 1,
        checkpoint_interval: args.checkpoint_interval,
    };

    // Pre-allocate inputs and state once, clone per iteration
    let inputs = generate_test_inputs(dims, device);
    let state = inner.init_state(args.batch);
    let _ = B::sync(device);

    for _ in 0..args.warmup {
        let mut s = state.clone();
        let _out = inner.forward(&mut s, inputs.clone());
        let _ = B::sync(device);
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let mut s = state.clone();
        let _ = B::sync(device);
        let start = Instant::now();
        let _out = inner.forward(&mut s, inputs.clone());
        let _ = B::sync(device);
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn bench_bwd_inner<
    B: burn::tensor::backend::AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B>,
>(
    args: &Args,
    config: &ModelConfig,
    device: &B::Device,
) -> f64 {
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(config, &inner_config, device);
    let dims = TestDims {
        batch_size: args.batch,
        num_heads: config.arch.num_heads,
        head_dim: config.head_dim(),
        seq_len: args.seq_len,
        mini_batch_size: args.mini_batch,
        iterations: 1,
        checkpoint_interval: args.checkpoint_interval,
    };

    // Pre-allocate inputs and state once, clone per iteration
    let inputs = generate_test_inputs(dims, device);
    let state = inner.init_state(args.batch);
    let _ = B::sync(device);

    for _ in 0..args.warmup {
        let mut s = state.clone();
        let _grads = inner.forward(&mut s, inputs.clone()).sum().backward();
        let _ = B::sync(device);
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let mut s = state.clone();
        let _ = B::sync(device);
        let start = Instant::now();
        let _grads = inner.forward(&mut s, inputs.clone()).sum().backward();
        let _ = B::sync(device);
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn make_config(
    model_size: ModelSize,
    mini_batch: usize,
    seq_len: usize,
    checkpoint_interval: usize,
) -> ModelConfig {
    let vocab_size = 32000; // match JAX/PyTorch reference configs
    let arch = Arc::new(ModelArch::from_size(model_size, vocab_size));
    let ttt = Arc::new(TTTConfig {
        mini_batch_size: mini_batch,
        max_seq_len: seq_len,
        checkpoint_interval,
        ..TTTConfig::default()
    });
    ModelConfig::new(arch, ttt)
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let config = make_config(
        args.model_size,
        args.mini_batch,
        args.seq_len,
        args.checkpoint_interval,
    );
    let mix = &args.ttt_type;

    let inner_only_type = if args.inner_only {
        assert!(
            mix.is_uniform(),
            "--inner-only requires a single layer type, got mix pattern: {mix}"
        );
        mix.get(0)
    } else {
        InnerModel::default() // unused
    };

    let (time_ms, dtype_str) = match (args.dtype.as_str(), args.backward, args.inner_only) {
        ("float32", false, false) => {
            let device: <BackendF32 as Backend>::Device = Default::default();
            let t = bench_fwd::<BackendF32>(&args, &config, mix, &device);
            (t, "float32")
        }
        ("float32", true, false) => {
            let device: <AutodiffF32 as Backend>::Device = Default::default();
            let t = bench_bwd::<AutodiffF32>(&args, &config, mix, &device);
            (t, "float32")
        }
        ("float32", false, true) => {
            let device: <BackendF32 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_fwd_inner::<BackendF32, inner_only_type>(
                &args, &config, &device
            ));
            (t, "float32")
        }
        ("float32", true, true) => {
            let device: <AutodiffF32 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_bwd_inner::<AutodiffF32, inner_only_type>(
                &args, &config, &device
            ));
            (t, "float32")
        }
        ("bfloat16", false, false) => {
            let device: <BackendBf16 as Backend>::Device = Default::default();
            let t = bench_fwd::<BackendBf16>(&args, &config, mix, &device);
            (t, "bfloat16")
        }
        ("bfloat16", true, false) => {
            let device: <AutodiffBf16 as Backend>::Device = Default::default();
            let t = bench_bwd::<AutodiffBf16>(&args, &config, mix, &device);
            (t, "bfloat16")
        }
        ("bfloat16", false, true) => {
            let device: <BackendBf16 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_fwd_inner::<BackendBf16, inner_only_type>(
                &args, &config, &device
            ));
            (t, "bfloat16")
        }
        ("bfloat16", true, true) => {
            let device: <AutodiffBf16 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_bwd_inner::<AutodiffBf16, inner_only_type>(
                &args, &config, &device
            ));
            (t, "bfloat16")
        }
        _ => panic!("Unknown dtype: {}. Use: float32, bfloat16", args.dtype),
    };

    let throughput = (args.batch * args.seq_len) as f64 / (time_ms / 1000.0);

    let result = BenchResult {
        implementation: "burn",
        model_size: args.model_size.to_string(),
        ttt_type: args.ttt_type.to_string(),
        backward: args.backward,
        inner_only: args.inner_only,
        batch: args.batch,
        seq_len: args.seq_len,
        mini_batch: args.mini_batch,
        checkpoint_interval: args.checkpoint_interval,
        dtype: dtype_str,
        time_ms,
        throughput,
    };

    if args.json {
        println!("{}", serde_json::to_string(&result).unwrap());
    } else {
        println!(
            "Burn TTT Benchmark{}",
            if args.inner_only { " (inner only)" } else { "" }
        );
        println!("==================");
        println!("Model: {} / {}", args.model_size, args.ttt_type);
        println!("Dtype: {}", dtype_str);
        println!("Backward: {}", args.backward);
        println!(
            "Batch: {}, Seq: {}, Mini-batch: {}, Checkpoint interval: {}",
            args.batch, args.seq_len, args.mini_batch, args.checkpoint_interval
        );
        println!();
        println!("Time: {:.2} ms", time_ms);
        println!("Throughput: {:.0} tok/s", throughput);
    }
}
