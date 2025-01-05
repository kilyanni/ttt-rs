# ttt-rs

A Rust implementation of [Test-Time Training (TTT)](https://arxiv.org/abs/2407.04620) language models with custom fused GPU kernels. Built on [Burn](https://burn.dev/) and [CubeCL](https://github.com/tracel-ai/cubecl).

TTT replaces the attention mechanism in transformers with a learned inner model (linear or MLP) that updates its weights at inference time via gradient descent on each input sequence. This project implements the full training and inference pipeline with hand-written tiled GPU kernels for the TTT inner loop.

## Repository Structure

```
persistent-ttt/
├── rust/                    Rust workspace (10 crates)
│   ├── cli/                 Main CLI binary + benchmark tool
│   ├── config/              Shared configuration types
│   ├── core/                Core TTT traits and base implementations
│   ├── data/                Data loading and tokenization
│   ├── fused/               Fused GPU kernels for TTT inner loop
│   ├── kernels/             Kernel infrastructure and activations
│   ├── layer/               Outer TTT layer and full model
│   ├── training/            Training loop, inference, evaluation
│   ├── harness/             Multi-run coordinator with crash recovery
│   └── thundercube/         Custom tiled GPU compute library
├── benches/                 Benchmark suite (JAX, PyTorch, Triton, Burn)
├── scripts/                 Utilities (tokenizer training)
├── nix/                     Nix build definitions
├── Dockerfile.rocm          ROCm dev container
└── Dockerfile.cuda          CUDA dev container
```

## Features

- **Multiple inner model types**: Linear, Linear+Adam, MLP (2-4 layer variants)
- **Fused GPU kernels**: Hand-written CubeCL kernels that fuse the entire TTT inner loop (forward and backward) into single kernel launches, with activation recomputation and shared memory optimization
- **Tiled compute**: Custom `thundercube` library providing `St` (shared memory) and `Rt` (register) tile abstractions with cooperative matrix multiply-accumulate
- **Streaming kernels**: Persistent kernels for pipelined execution (ROCm)
- **Multi-stage checkpointing**: Configurable checkpoint intervals to trade compute for memory
- **Mixed layer types**: Mix different inner models across layers (e.g. `"4*linear,mlp"`)
- **Model sizes**: 12M, 60M, 125M, 350M, 760M parameters
- **Data types**: f32, f16, bf16
- **Pre-tokenized datasets**: Memory-mapped binary format to avoid on-the-fly tokenization

## Getting Started

See **[QUICKSTART.md](QUICKSTART.md)** for step-by-step setup (Docker, manual install, or Nix).

### Prerequisites

- Rust stable (edition 2024)  - or use Docker/Nix
- ROCm 6.x/7.x **or** CUDA 12.x **or** CPU-only
- pkg-config, cmake, OpenSSL, SQLite

## Building

All cargo commands run from the `rust/` directory:

```bash
cd rust

# ROCm (AMD GPUs)
cargo build --release --features rocm

# CUDA (NVIDIA GPUs)
cargo build --release --features cuda

# CPU fallback
cargo build --release --features cpu
```

## Usage

### Training

```bash
# Train on TinyStories with pre-tokenized data (default)
ttt train --size 125m --layer-type fused-tile-multi --epochs 10 --batch 32 --lr 1e-3

# Resume from checkpoint
ttt train --resume ./artifacts

# Custom tokenizer
ttt train --size 60m --tokenizer EleutherAI/gpt-neox-20b
```

Key training flags:
- `--size`  - Model size preset (12m, 60m, 125m, 350m, 760m)
- `--layer-type`  - Inner model: `linear`, `mlp`, `fused-naive-multi`, `fused-tile-multi`
- `--epochs`, `--batch`, `--lr`  - Training hyperparameters
- `--seq-len`  - Maximum sequence length (default: 2048)
- `--mini-batch`  - TTT mini-batch size (default: 16)
- `--dtype`  - Data type: `f32`, `f16`, `bf16`
- `--seed`  - RNG seed for reproducibility
- `--out`  - Artifact output directory

### Inference

```bash
# Single generation
ttt generate ./artifacts "Once upon a time"

# Interactive session
ttt interactive ./artifacts
```

### Evaluation

```bash
# Evaluate on validation data
ttt eval ./artifacts --samples 1000 --batch 32

# Evaluate at a different sequence length
ttt eval ./artifacts --max-seq-len 4096

# Evaluate a specific checkpoint
ttt eval ./artifacts --checkpoint 5
```

### Utilities

```bash
# Inspect a training run
ttt info ./artifacts
ttt info ./artifacts --verbose

# Export metrics to CSV
ttt export-metrics ./runs/* -o metrics.csv --metrics loss,perplexity

# Shell completions
ttt completions bash > ~/.bash_completion.d/ttt
ttt completions fish > ~/.config/fish/completions/ttt.fish
ttt completions zsh > "${fpath[1]}/_ttt"
```

### Training Harness

The `ttt-harness` binary coordinates parallel training runs with VRAM estimation and automatic crash recovery:

```bash
ttt-harness run --config harness.toml
ttt-harness status --state harness_state.json
ttt-harness reset --run-id <id>
```

### Tokenizer Training

```bash
# Via Nix
nix run .#train-tokenizer

# Directly
python scripts/train_tokenizer.py
```

## Benchmarking

The `benches/` directory provides a cross-implementation benchmark suite:

```bash
# Burn/CubeCL (this project)
cargo run --release --features rocm --bin ttt-bench -- \
  --model-size 125m --ttt-type linear --backward

# JAX reference
nix develop .#bench-jax
python benchmark.py --model-size 125m --ttt-type linear --backward

# PyTorch reference
nix develop .#bench-pytorch
python benchmark.py --model-size 125m --ttt-type linear

# Automated sweep
./benches/sweep.fish
```

See [`benches/README.md`](benches/README.md) for details.

## Docker

```bash
# ROCm
docker build -f Dockerfile.rocm -t ttt-rocm .

# CUDA
docker build -f Dockerfile.cuda -t ttt-cuda .
```

The Docker images include the full Rust toolchain for on-server development, pre-built binaries, and Python packages for tokenizer training.

## CubeCL Kernel Cache

CubeCL caches compiled GPU kernels and autotune results to avoid recompilation. The cache location is set in `cubecl.toml`:

- **`rust/cubecl.toml`** uses `cache = "target"`  - writes to `./target/hip/` and `./target/autotune/`
- **`benches/cubecl.toml`** uses `cache = "global"`  - writes to `~/.config/cubecl/` (or platform equivalent)

**The cache goes stale easily.** Cache keys are based on kernel type signatures and generic parameters, *not* source code. If you change kernel internals without changing the type signature, CubeCL will silently serve the old compiled kernel. When things behave unexpectedly after a kernel change, clear the cache:

```bash
# Clear all caches (global + any target-local ones)
find . -name "hip-kernel.json.log" -delete
find ~/.config -path "*/cubecl/*" -name "hip-kernel.json.log" -delete
rm -rf ./target/hip ./target/autotune
```

Set `CUBECL_DEBUG_LOG=stdout` to verify that kernels are actually being recompiled.

## Crate Overview

| Crate | Description |
|---|---|
| `ttt-cli` | CLI binary (`ttt`) and benchmark binary (`ttt-bench`) |
| `ttt-config` | Model sizes, layer types, training params, serialization |
| `ttt-core` | `TTTInnerModel` trait, Linear/MLP implementations, QKV projections |
| `ttt-data` | HuggingFace tokenizer wrapper, text/pre-tokenized dataset loaders |
| `ttt-fused` | Fused CubeCL kernels: naive, tiled, multi-stage, streaming variants |
| `ttt-kernels` | `FusedKernel` trait, GELU activation kernels |
| `ttt-layer` | Outer TTT layer, block composition, full transformer model |
| `ttt-training` | Training loop with checkpointing, inference, evaluation |
| `ttt-harness` | Multi-run coordinator with VRAM estimation and crash recovery |
| `thundercube` | Tiled GPU compute: `St`/`Rt` tiles, MMA, reductions, broadcasts |

## License

[MIT](LICENSE)
