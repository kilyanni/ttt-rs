"""Benchmark script for TTT Kernels (Triton decode, forward pass only).

NOTE: Triton kernels have been patched for ROCm (fp32 casts for sqrt/sigmoid).
      Use --dtype float32 for best compatibility on AMD GPUs.

NOTE: The kernels implementation is inference-only (optimized for decode).
      Backward pass is not supported as it requires cache_params.
"""

import argparse
import json
import time
import torch

from ttt.configuration_ttt import TTT_STANDARD_CONFIGS, TTTConfig
from ttt.generation import TTTCache
from ttt.modeling_ttt import TTTForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTT Kernels")
    parser.add_argument("--model-size", type=str, default="1b",
                        choices=list(TTT_STANDARD_CONFIGS.keys()),
                        help="Model size")
    parser.add_argument("--ttt-type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="TTT layer type (linear or mlp)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast kernel variant")
    parser.add_argument("--backward", action="store_true",
                        help="Not supported - kernels are inference-only")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--mini-batch", type=int, default=16,
                        help="TTT mini-batch size (seq-len must be divisible by this)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of benchmark runs")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type (float32 recommended for ROCm)")
    parser.add_argument("--inner-only", action="store_true",
                        help="Not supported - kernels require full model with cache")
    parser.add_argument("--comment", type=str, default="",
                        help="Ignored; visible in process list for identification")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for scripting)")
    args = parser.parse_args()

    if args.inner_only:
        print("\nERROR: --inner-only is not supported for kernels implementation.")
        print("The Triton kernels require the full model with cache_params.")
        print("Use the PyTorch implementation for inner-only benchmarks.")
        return

    if args.backward:
        print("\nERROR: --backward is not supported for kernels implementation.")
        print("The Triton kernels are optimized for inference (decode) only.")
        print("Use the PyTorch implementation for training benchmarks.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    # Build model name from ttt-type and fast flag
    model_name = f"ttt-{args.ttt_type}"
    if args.fast:
        model_name += "-fast"

    # Validate seq_len is divisible by mini_batch
    if args.seq_len % args.mini_batch != 0:
        print(f"ERROR: seq-len ({args.seq_len}) must be divisible by mini-batch ({args.mini_batch})")
        return

    if not args.json:
        print(f"\n{'='*60}")
        print(f"TTT Kernels Benchmark (Forward Pass)")
        print(f"{'='*60}")
        print(f"Model size: {args.model_size}")
        print(f"TTT type: {args.ttt_type}" + (" (fast)" if args.fast else ""))
        print(f"Batch: {args.batch}, Seq len: {args.seq_len}, Mini-batch: {args.mini_batch}")
        print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
        print(f"Dtype: {args.dtype}")
        print(f"{'='*60}\n")

    # Create model
    config = TTTConfig(**TTT_STANDARD_CONFIGS[args.model_size], vocab_size=32000)
    config.seq_modeling_block = model_name
    config.mini_batch_size = args.mini_batch
    config.use_compile = True
    config.dtype = dtype
    config.fused_add_norm = True
    config.residual_in_fp32 = True

    if not args.json:
        print(f"Hidden size: {config.hidden_size}")
        print(f"Layers: {config.num_hidden_layers}")

    model = TTTForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    if not args.json:
        print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    # Create cache (kernels always need cache_params for initial W/b states)
    cache = TTTCache(args.batch, model)
    cache.allocate_inference_cache()

    # Create input
    input_ids = torch.randint(1, 32000, (args.batch, args.seq_len),
                              dtype=torch.long, device=device)

    # Warmup
    if not args.json:
        print(f"\nWarmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        with torch.no_grad():
            cache.reset(args.seq_len, args.batch, model)
            _ = model(input_ids=input_ids, cache_params=cache, is_prefill=True)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    if not args.json:
        print(f"Benchmarking ({args.repeats} runs)...")
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeats):
        with torch.no_grad():
            cache.reset(args.seq_len, args.batch, model)
            _ = model(input_ids=input_ids, cache_params=cache, is_prefill=True)
    if device == "cuda":
        torch.cuda.synchronize()
    avg_time = (time.time() - start) / args.repeats

    tokens_per_batch = args.batch * args.seq_len
    throughput = tokens_per_batch / avg_time

    if args.json:
        result = {
            "implementation": "kernels",
            "model_size": args.model_size,
            "ttt_type": args.ttt_type,
            "fast": args.fast,
            "backward": False,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "mini_batch": args.mini_batch,
            "dtype": args.dtype,
            "num_params": num_params,
            "time_ms": avg_time * 1000,
            "throughput": throughput,
            "device": device + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""),
        }
        print(json.dumps(result))
    else:
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch}")
        print(f"Sequence length: {args.seq_len}")
        print(f"Forward time: {avg_time * 1000:.1f}ms")
        print(f"Throughput: {throughput:.1f} tok/s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
