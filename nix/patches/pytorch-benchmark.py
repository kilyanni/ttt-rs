"""Benchmark script for TTT PyTorch reference implementation (forward and backward pass)."""

import argparse
import json
import time
import torch

from ttt import (
    TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTLinear, TTTMLP,
    permute_qk, undo_permute_qk, apply_rotary_pos_emb,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTT PyTorch")
    parser.add_argument("--model-size", type=str, default="125m",
                        choices=list(TTT_STANDARD_CONFIGS.keys()),
                        help="Model size")
    parser.add_argument("--ttt-type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="TTT layer type")
    parser.add_argument("--backward", action="store_true",
                        help="Include backward pass (gradient computation)")
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
                        help="Benchmark just the inner TTT layer (single layer)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for optimized kernels")
    parser.add_argument("--comment", type=str, default="",
                        help="Ignored; visible in process list for identification")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for scripting)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    # Validate seq_len is divisible by mini_batch
    if args.seq_len % args.mini_batch != 0:
        print(f"ERROR: seq-len ({args.seq_len}) must be divisible by mini-batch ({args.mini_batch})")
        return

    if not args.json:
        print(f"\n{'='*60}")
        print(f"TTT PyTorch Benchmark ({'Forward + Backward' if args.backward else 'Forward Pass'})")
        print(f"{'='*60}")
        print(f"Model size: {args.model_size}")
        print(f"TTT type: {args.ttt_type}")
        print(f"Batch: {args.batch}, Seq len: {args.seq_len}, Mini-batch: {args.mini_batch}")
        print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
        print(f"Dtype: {args.dtype}")
        if args.compile:
            print(f"torch.compile: enabled")
        print(f"{'='*60}\n")

    # Create model
    config = TTTConfig(**TTT_STANDARD_CONFIGS[args.model_size], vocab_size=32000)
    config.ttt_layer_type = args.ttt_type
    config.mini_batch_size = args.mini_batch
    config.max_position_embeddings = args.seq_len

    if not args.json:
        print(f"Hidden size: {config.hidden_size}")
        print(f"Layers: {config.num_hidden_layers}")
        if args.inner_only:
            print("Mode: inner TTT layer only")

    if args.inner_only:
        # Benchmark just the inner TTT loop (W/b update scan), NOT the full TTT layer.
        # Pre-computes QKV projections, RoPE, and eta outside the timing loop
        # to match Burn's inner-only scope.
        layer_cls = TTTLinear if args.ttt_type == "linear" else TTTMLP
        layer = layer_cls(config=config, layer_idx=0).to(device=device, dtype=dtype)
        if args.compile:
            layer = torch.compile(layer)

        num_params = sum(p.numel() for p in layer.parameters())
        if not args.json:
            print(f"Layer parameters: {num_params:,}")

        B, L = args.batch, args.seq_len
        hidden_states = torch.randn(B, L, config.hidden_size, dtype=dtype, device=device)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)

        # Pre-compute QKV, RoPE, and eta (not timed)
        if not args.json:
            print("Pre-computing QKV and eta...")
        with torch.no_grad():
            XQ, XK, XV = layer.get_qkv_projections(hidden_states)
            XQ = XQ.reshape(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)
            XK = XK.reshape(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)
            XV = XV.reshape(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)

            cos, sin = layer.rotary_emb(XV, position_ids % layer.mini_batch_size)
            XQ, XK = permute_qk(XQ, XK)
            XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
            XQ, XK = undo_permute_qk(XQ, XK)

            num_mini_batch = L // layer.mini_batch_size
            raw_inputs = {
                "XQ": XQ[:, :, :num_mini_batch * layer.mini_batch_size],
                "XK": XK[:, :, :num_mini_batch * layer.mini_batch_size],
                "XV": XV[:, :, :num_mini_batch * layer.mini_batch_size],
                "X": hidden_states[:, :num_mini_batch * layer.mini_batch_size],
            }
            ttt_inputs = layer.get_ttt_inputs(raw_inputs, layer.mini_batch_size, None)

        if args.backward:
            layer.train()
            # Detach pre-computed inputs so grads only flow through ttt params (W1, b1)
            ttt_inputs = {k: v.detach() for k, v in ttt_inputs.items()}

            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                layer.zero_grad()
                out, _ = layer.ttt(ttt_inputs, layer.mini_batch_size, None)
                out.sum().backward()
            if device == "cuda":
                torch.cuda.synchronize()

            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(args.repeats):
                layer.zero_grad()
                out, _ = layer.ttt(ttt_inputs, layer.mini_batch_size, None)
                out.sum().backward()
            if device == "cuda":
                torch.cuda.synchronize()
            avg_time = (time.time() - start) / args.repeats
        else:
            layer.eval()

            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                with torch.no_grad():
                    _ = layer.ttt(ttt_inputs, layer.mini_batch_size, None)
            if device == "cuda":
                torch.cuda.synchronize()

            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(args.repeats):
                with torch.no_grad():
                    _ = layer.ttt(ttt_inputs, layer.mini_batch_size, None)
            if device == "cuda":
                torch.cuda.synchronize()
            avg_time = (time.time() - start) / args.repeats

    else:
        # Full model benchmark (default)
        model = TTTForCausalLM(config).to(device=device, dtype=dtype)
        if args.compile:
            model = torch.compile(model)

        if args.backward:
            model.train()
        else:
            model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        if not args.json:
            print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

        # Create input
        input_ids = torch.randint(1, 32000, (args.batch, args.seq_len),
                                  dtype=torch.long, device=device)

        if args.backward:
            # Forward + backward pass
            # Use sum() as loss to match Burn benchmark (no cross-entropy overhead)
            # Warmup
            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                model.zero_grad()
                out = model(input_ids=input_ids)
                out.logits.sum().backward()
            if device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(args.repeats):
                model.zero_grad()
                out = model(input_ids=input_ids)
                out.logits.sum().backward()
            if device == "cuda":
                torch.cuda.synchronize()
            avg_time = (time.time() - start) / args.repeats

        else:
            # Forward pass only
            # Warmup
            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                with torch.no_grad():
                    _ = model(input_ids=input_ids)
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
                    _ = model(input_ids=input_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            avg_time = (time.time() - start) / args.repeats

    tokens_per_batch = args.batch * args.seq_len
    throughput = tokens_per_batch / avg_time

    if args.json:
        result = {
            "implementation": "pytorch-compile" if args.compile else "pytorch",
            "model_size": args.model_size,
            "ttt_type": args.ttt_type,
            "backward": args.backward,
            "inner_only": args.inner_only,
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
        print(f"{'Forward + Backward' if args.backward else 'Forward'} time: {avg_time * 1000:.1f}ms")
        print(f"Throughput: {throughput:.1f} tok/s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
