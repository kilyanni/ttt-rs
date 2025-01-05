# TTT ROCm Benchmarks

Benchmark infrastructure for comparing TTT (Test-Time Training) implementations on AMD ROCm GPUs.
Compares JAX, PyTorch, Triton Kernels, and Burn (this project) implementations.

## Hardware Tested

- **GPU**: AMD Radeon RX 6800 (RDNA2, gfx1030)
- **ROCm Version**: 7.1.x (via nixpkgs)

## Available Implementations

| Shell | Description |
|-------|-------------|
| `bench-jax` | JAX implementation with ROCm 7.1.1 plugin |
| `bench-pytorch` | PyTorch reference implementation |
| `bench-kernels` | Triton kernels |
| `bench-burn` | Burn/CubeCL implementation (this project) |

## Quick Start

```bash
# Enter a specific implementation shell (from project root)
nix develop .#bench-jax
nix develop .#bench-pytorch
nix develop .#bench-kernels
nix develop .#bench-burn
```

## Running Benchmarks

All benchmarks use consistent CLI arguments:

```
--model-size   Model size (125m, 350m, 760m, 1.3b, 1b)
--ttt-type     TTT layer type (linear, mlp)
--backward     Include backward pass (gradient computation)
--batch        Batch size (default: 1)
--seq-len      Sequence length (default: 2048)
--mini-batch   TTT mini-batch size (default: 16, seq-len must be divisible)
--dtype        Data type (float16, bfloat16, float32)
--warmup       Warmup runs (default: 3)
--repeats      Benchmark runs (default: 5)
--json         Output results as JSON (for scripting/sweeps)
--inner-only   Only benchmark the inner layer
--checkpoint-interval   Interval for gradient checkpointing
--comment      Ignored, meant for making processes recognizable in `nvtop` and similar
```

### JAX (Recommended - Fastest)

```bash
nix develop .#bench-jax
python benchmark.py --model-size 125m --ttt-type linear
# With backward pass (training)
python benchmark.py --model-size 125m --ttt-type linear --backward
```

### PyTorch

```bash
nix develop .#bench-pytorch
python benchmark.py --model-size 125m --ttt-type linear
```

### Kernels (Triton) - Inference Only

```bash
nix develop .#bench-kernels
python benchmark.py --model-size 1b --ttt-type linear
```

## Automated Sweeps

```bash
# Run full progressive sweep
./benches/sweep.fish

# Check progress
./benches/sweep.fish --status

# Export results to CSV
./benches/sweep.fish --csv > results.csv

# Retry failed benchmarks
./benches/sweep.fish --clear-failures
```
