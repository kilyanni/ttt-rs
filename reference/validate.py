#!/usr/bin/env python3
"""
Validation dispatch script for TTT components.
Directly dispatches to the official PyTorch reference implementation.

Usage:
    python -m reference.validate                    # Run with defaults
    python -m reference.validate --H 5 --D 8        # Custom dimensions
    python -m reference.validate --seed 123         # Custom seed
"""

import argparse
import os
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = (
    ""  # Force CPU to save any driver-related hassle. We're not pushing much data anyways.
)

from pathlib import Path

import torch
from safetensors.torch import save_file

from . import (
    Block,
    TTTConfig,
    TTTForCausalLM,
    TTTLinear,
    TTTMLP,
    ln_fused_l2_bwd,
    ln_fwd,
    permute_qk,
    undo_permute_qk,
)


@dataclass
class ValidationConfig:
    B: int = 2
    L: int = 16
    H: int = 4
    D: int = 16
    mini_batch_size: int = 16
    intermediate_mult: int = 4
    vocab_size: int = 1000
    num_layers: int = 2
    seed: int = 42
    use_gate: bool = True
    conv_kernel: int = 4
    pre_conv: bool = False
    share_qk: bool = True
    tie_word_embeddings: bool = True
    layer_type: str = "linear"

    def __post_init__(self):
        assert self.D % 2 == 0, f"D must be even (got {self.D})"
        assert self.L % self.mini_batch_size == 0, f"L must be divisible by mini_batch_size (L={self.L}, mini_batch_size={self.mini_batch_size})"

    @property
    def hidden_size(self):
        return self.H * self.D

    @property
    def intermediate_size(self):
        return self.hidden_size * self.intermediate_mult


def validate_ln_fused_l2_bwd(save_dir: Path, cfg: ValidationConfig):
    """Dispatch to reference ln_fused_l2_bwd"""
    print("\n--- ln_fused_l2_bwd ---")
    torch.manual_seed(cfg.seed)

    B, H, K, D = cfg.B, cfg.H, cfg.L, cfg.D

    x = torch.randn(B, H, K, D)
    target = torch.randn(B, H, K, D)
    gamma = torch.ones(H, 1, D)
    beta = torch.zeros(H, 1, D)

    result = ln_fused_l2_bwd(x, target, gamma, beta, eps=1e-6)

    print(f"  shape: [{B}, {H}, {K}, {D}]")
    print(f"  result: mean={result.mean():.6f}, std={result.std():.6f}")

    save_file(
        {
            "ln_fused_x": x.contiguous(),
            "ln_fused_target": target.contiguous(),
            "ln_fused_gamma": gamma.contiguous(),
            "ln_fused_beta": beta.contiguous(),
            "ln_fused_result": result.contiguous(),
        },
        save_dir / "ln_fused.safetensors",
    )
    return True


def validate_ln_fwd(save_dir: Path, cfg: ValidationConfig):
    """Dispatch to reference ln_fwd"""
    print("\n--- ln_fwd ---")
    torch.manual_seed(cfg.seed)

    B, H, K, D = cfg.B, cfg.H, cfg.L, cfg.D

    x = torch.randn(B, H, K, D)
    gamma = torch.ones(H, 1, D)
    beta = torch.zeros(H, 1, D)

    result = ln_fwd(x, gamma, beta, eps=1e-6)

    print(f"  shape: [{B}, {H}, {K}, {D}]")
    print(f"  result: mean={result.mean():.6f}, std={result.std():.6f}")

    save_file(
        {
            "ln_fwd_x": x.contiguous(),
            "ln_fwd_gamma": gamma.contiguous(),
            "ln_fwd_beta": beta.contiguous(),
            "ln_fwd_result": result.contiguous(),
        },
        save_dir / "ln_fwd.safetensors",
    )
    return True


def validate_permute_qk(save_dir: Path, cfg: ValidationConfig):
    """Dispatch to reference permute_qk/undo_permute_qk"""
    print("\n--- permute_qk ---")
    torch.manual_seed(cfg.seed)

    B, H, L, D = cfg.B, cfg.H, cfg.L, cfg.D

    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)

    q_perm, k_perm = permute_qk(q.clone(), k.clone())
    q_undo, k_undo = undo_permute_qk(q_perm.clone(), k_perm.clone())

    diff = (q - q_undo).abs().max().item()
    print(f"  shape: [{B}, {H}, {L}, {D}]")
    print(f"  roundtrip diff: {diff:.2e}")

    save_file(
        {
            "permute_q_in": q.contiguous(),
            "permute_k_in": k.contiguous(),
            "permute_q_out": q_perm.contiguous(),
            "permute_k_out": k_perm.contiguous(),
        },
        save_dir / "permute_qk.safetensors",
    )
    return diff < 1e-6


def validate_ttt_linear_full(save_dir: Path, cfg: ValidationConfig):
    """
    Dispatch to reference TTTLinear.ttt() method.
    This tests the ACTUAL implementation, not a manual reimplementation.
    """
    print("\n--- TTTLinear.ttt (full forward) ---")
    torch.manual_seed(cfg.seed)

    B, H, K, D = cfg.B, cfg.H, cfg.mini_batch_size, cfg.D
    hidden_size = cfg.hidden_size

    config = TTTConfig(
        hidden_size=hidden_size,
        num_attention_heads=H,
        mini_batch_size=K,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
    )

    ttt = TTTLinear(config, layer_idx=0)
    ttt.eval()
    print(f"  shape: B={B}, H={H}, K={K}, D={D}")

    XQ = torch.full((B, H, 1, K, D), 0.1)
    XK = torch.full((B, H, 1, K, D), 0.2)
    XV = torch.full((B, H, 1, K, D), 0.3)

    X = torch.randn(B, 1, K, hidden_size)

    token_eta, ttt_lr_eta = ttt.get_eta(X, mini_batch_step_offset=0, mini_batch_size=K)
    eta = token_eta * ttt_lr_eta

    inputs = {
        "XQ": XQ,
        "XK": XK,
        "XV": XV,
        "eta": eta,
        "token_eta": token_eta,
        "ttt_lr_eta": ttt_lr_eta,
    }

    # Direct dispatch to reference ttt() method
    output, last_params = ttt.ttt(
        inputs,
        mini_batch_size=K,
        last_mini_batch_params_dict=None,
        cache_params=None,
    )

    print(f"  output: {output.shape}")
    print(f"  output stats: mean={output.mean():.6f}, std={output.std():.6f}")
    print(f"  W1 final: {last_params['W1_states'].shape}")
    print(f"  b1 final: {last_params['b1_states'].shape}")

    # Initial state
    W1_init = ttt.W1.unsqueeze(0).expand(B, -1, -1, -1)
    b1_init = ttt.b1.unsqueeze(0).expand(B, -1, -1, -1)

    # Outputs - output is [B, L, C], need to reshape for comparison
    # But ttt() returns [B, L, hidden_size], we need [B, H, K, D]
    output_reshaped = output.reshape(B, K, H, D).transpose(1, 2)  # [B, H, K, D]

    save_file(
        {
            "XQ": XQ[:, :, 0].contiguous(),  # [B, H, K, D]
            "XK": XK[:, :, 0].contiguous(),
            "XV": XV[:, :, 0].contiguous(),
            "token_eta": (ttt.token_idx + ttt.learnable_token_idx).contiguous(),
            "ttt_lr_eta": ttt_lr_eta[:, :, 0, 0, :].contiguous(),  # [B, H, K]
            "W1_init": W1_init.contiguous(),
            "b1_init": b1_init.squeeze(-2).contiguous(),
            "ln_weight": ttt.ttt_norm_weight.contiguous(),
            "ln_bias": ttt.ttt_norm_bias.contiguous(),
            "output_expected": output_reshaped.contiguous(),
            "W1_last_expected": last_params["W1_states"].contiguous(),
            "b1_last_expected": last_params["b1_states"].squeeze(-2).contiguous(),
        },
        save_dir / "ttt_linear.safetensors",
    )

    return True


def validate_ttt_mlp_full(save_dir: Path, cfg: ValidationConfig):
    """
    Dispatch to reference TTTMLP.ttt() method.
    This tests the ACTUAL implementation, not a manual reimplementation.
    """
    print("\n--- TTTMLP.ttt (full forward) ---")
    torch.manual_seed(cfg.seed)

    B, H, K, D = cfg.B, cfg.H, cfg.mini_batch_size, cfg.D
    hidden_size = cfg.hidden_size

    config = TTTConfig(
        hidden_size=hidden_size,
        num_attention_heads=H,
        mini_batch_size=K,
        ttt_layer_type="mlp",
        ttt_base_lr=1.0,
    )

    ttt = TTTMLP(config, layer_idx=0)
    ttt.eval()
    print(f"  shape: B={B}, H={H}, K={K}, D={D}")

    XQ = torch.full((B, H, 1, K, D), 0.1)
    XK = torch.full((B, H, 1, K, D), 0.2)
    XV = torch.full((B, H, 1, K, D), 0.3)

    X = torch.randn(B, 1, K, hidden_size)

    token_eta, ttt_lr_eta = ttt.get_eta(X, mini_batch_step_offset=0, mini_batch_size=K)
    eta = token_eta * ttt_lr_eta

    inputs = {
        "XQ": XQ,
        "XK": XK,
        "XV": XV,
        "eta": eta,
        "token_eta": token_eta,
        "ttt_lr_eta": ttt_lr_eta,
    }

    # Direct dispatch to reference ttt() method
    output, last_params = ttt.ttt(
        inputs,
        mini_batch_size=K,
        last_mini_batch_params_dict=None,
        cache_params=None,
    )

    print(f"  output: {output.shape}")
    print(f"  output stats: mean={output.mean():.6f}, std={output.std():.6f}")
    print(f"  W1 final: {last_params['W1_states'].shape}")
    print(f"  b1 final: {last_params['b1_states'].shape}")
    print(f"  W2 final: {last_params['W2_states'].shape}")
    print(f"  b2 final: {last_params['b2_states'].shape}")

    # Initial state
    W1_init = ttt.W1.unsqueeze(0).expand(B, -1, -1, -1)
    b1_init = ttt.b1.unsqueeze(0).expand(B, -1, -1, -1)
    W2_init = ttt.W2.unsqueeze(0).expand(B, -1, -1, -1)
    b2_init = ttt.b2.unsqueeze(0).expand(B, -1, -1, -1)

    # Outputs - output is [B, L, C], need to reshape for comparison
    output_reshaped = output.reshape(B, K, H, D).transpose(1, 2)  # [B, H, K, D]

    save_file(
        {
            "XQ": XQ[:, :, 0].contiguous(),  # [B, H, K, D]
            "XK": XK[:, :, 0].contiguous(),
            "XV": XV[:, :, 0].contiguous(),
            "token_eta": (ttt.token_idx + ttt.learnable_token_idx).contiguous(),
            "ttt_lr_eta": ttt_lr_eta[:, :, 0, 0, :].contiguous(),  # [B, H, K]
            "W1_init": W1_init.contiguous(),
            "b1_init": b1_init.squeeze(-2).contiguous(),
            "W2_init": W2_init.contiguous(),
            "b2_init": b2_init.squeeze(-2).contiguous(),
            "ln_weight": ttt.ttt_norm_weight.contiguous(),
            "ln_bias": ttt.ttt_norm_bias.contiguous(),
            "output_expected": output_reshaped.contiguous(),
            "W1_last_expected": last_params["W1_states"].contiguous(),
            "b1_last_expected": last_params["b1_states"].squeeze(-2).contiguous(),
            "W2_last_expected": last_params["W2_states"].contiguous(),
            "b2_last_expected": last_params["b2_states"].squeeze(-2).contiguous(),
        },
        save_dir / "ttt_mlp.safetensors",
    )

    return True


def _add_ttt_inner_weights(tensors: dict, ttt_layer, layer_type: str, prefix: str = ""):
    """Add inner model weights to tensors dict based on layer type."""
    tensors[f"{prefix}W1"] = ttt_layer.W1.contiguous()
    tensors[f"{prefix}b1"] = ttt_layer.b1.contiguous()
    if layer_type == "mlp":
        tensors[f"{prefix}W2"] = ttt_layer.W2.contiguous()
        tensors[f"{prefix}b2"] = ttt_layer.b2.contiguous()


def validate_ttt_layer_forward(save_dir: Path, cfg: ValidationConfig):
    """
    Dispatch to reference TTT layer forward() - the complete TTT layer.
    Tests QKV projections, convolutions, RoPE, gating, and inner loop.
    """
    layer_class = TTTLinear if cfg.layer_type == "linear" else TTTMLP
    print(f"\n--- {layer_class.__name__}.forward (full layer) ---")
    torch.manual_seed(cfg.seed)

    B, L, H, D = cfg.B, cfg.L, cfg.H, cfg.D
    hidden_size = cfg.hidden_size
    mini_batch_size = cfg.mini_batch_size

    config = TTTConfig(
        hidden_size=hidden_size,
        num_attention_heads=H,
        mini_batch_size=mini_batch_size,
        ttt_layer_type=cfg.layer_type,
        ttt_base_lr=1.0,
        share_qk=cfg.share_qk,
        use_gate=cfg.use_gate,
        conv_kernel=cfg.conv_kernel,
        pre_conv=cfg.pre_conv,
    )

    ttt_layer = layer_class(config, layer_idx=0)
    ttt_layer.eval()
    print(f"  shape: B={B}, L={L}, H={H}, D={D}, mini_batch={mini_batch_size}")

    # Input hidden states [B, L, hidden_size]
    hidden_states = torch.randn(B, L, hidden_size) * 0.1

    # Position IDs [B, L]
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        output = ttt_layer(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            cache_params=None,
        )

    print(f"  input: {hidden_states.shape}")
    print(f"  output: {output.shape}")
    print(f"  output stats: mean={output.mean():.6f}, std={output.std():.6f}")

    tensors = {
        "input": hidden_states.contiguous(),
        "output_expected": output.contiguous(),
        "q_proj_weight": ttt_layer.q_proj.weight.contiguous(),
        "v_proj_weight": ttt_layer.v_proj.weight.contiguous(),
        "o_proj_weight": ttt_layer.o_proj.weight.contiguous(),
        "ttt_norm_weight": ttt_layer.ttt_norm_weight.contiguous(),
        "ttt_norm_bias": ttt_layer.ttt_norm_bias.contiguous(),
        "post_norm_weight": ttt_layer.post_norm.weight.contiguous(),
        "post_norm_bias": ttt_layer.post_norm.bias.contiguous(),
        "lr_weight": ttt_layer.learnable_ttt_lr_weight.contiguous(),
        "lr_bias": ttt_layer.learnable_ttt_lr_bias.contiguous(),
        "token_idx": ttt_layer.token_idx.contiguous(),
        "learnable_token_idx": ttt_layer.learnable_token_idx.contiguous(),
        "config_mini_batch_size": torch.tensor([mini_batch_size], dtype=torch.int64),
        "config_share_qk": torch.tensor([1 if cfg.share_qk else 0], dtype=torch.int64),
        "config_use_gate": torch.tensor([1 if cfg.use_gate else 0], dtype=torch.int64),
    }
    _add_ttt_inner_weights(tensors, ttt_layer, cfg.layer_type)
    if ttt_layer.use_gate:
        tensors["g_proj_weight"] = ttt_layer.g_proj.weight.contiguous()
    if cfg.share_qk:
        # Convolutions only exist when share_qk=True
        tensors["conv_q_weight"] = ttt_layer.conv_q.weight.contiguous()
        tensors["conv_q_bias"] = ttt_layer.conv_q.bias.contiguous()
        tensors["conv_k_weight"] = ttt_layer.conv_k.weight.contiguous()
        tensors["conv_k_bias"] = ttt_layer.conv_k.bias.contiguous()
    else:
        tensors["k_proj_weight"] = ttt_layer.k_proj.weight.contiguous()

    save_file(tensors, save_dir / f"ttt_layer_{cfg.layer_type}.safetensors")

    return True


def validate_block_forward(save_dir: Path, cfg: ValidationConfig):
    """
    Dispatch to reference Block.forward() - full block with TTT + FFN.
    Saves ALL parameters needed to reconstruct the block in Rust.
    """
    print(f"\n--- Block.forward (TTT-{cfg.layer_type} + FFN) ---")
    torch.manual_seed(cfg.seed)

    B, L, H, D = cfg.B, cfg.L, cfg.H, cfg.D
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    mini_batch_size = cfg.mini_batch_size

    config = TTTConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=H,
        mini_batch_size=mini_batch_size,
        ttt_layer_type=cfg.layer_type,
        ttt_base_lr=1.0,
        share_qk=cfg.share_qk,
        use_gate=cfg.use_gate,
        conv_kernel=cfg.conv_kernel,
        pre_conv=cfg.pre_conv,
    )

    block = Block(config, layer_idx=0)
    block.eval()
    print(f"  shape: B={B}, L={L}, H={H}, D={D}, intermediate={intermediate_size}")

    hidden_states = torch.randn(B, L, hidden_size) * 0.1
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        output = block(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            cache_params=None,
        )

    print(f"  input: {hidden_states.shape}")
    print(f"  output: {output.shape}")
    print(f"  output stats: mean={output.mean():.6f}, std={output.std():.6f}")

    ttt = block.seq_modeling_block
    tensors = {
        "input": hidden_states.contiguous(),
        "output_expected": output.contiguous(),
        # RMS norm parameters
        "seq_norm_weight": block.seq_norm.weight.contiguous(),
        "ffn_norm_weight": block.ffn_norm.weight.contiguous(),
        # TTT layer parameters
        "q_proj_weight": ttt.q_proj.weight.contiguous(),
        "v_proj_weight": ttt.v_proj.weight.contiguous(),
        "o_proj_weight": ttt.o_proj.weight.contiguous(),
        "ttt_norm_weight": ttt.ttt_norm_weight.contiguous(),
        "ttt_norm_bias": ttt.ttt_norm_bias.contiguous(),
        "post_norm_weight": ttt.post_norm.weight.contiguous(),
        "post_norm_bias": ttt.post_norm.bias.contiguous(),
        "lr_weight": ttt.learnable_ttt_lr_weight.contiguous(),
        "lr_bias": ttt.learnable_ttt_lr_bias.contiguous(),
        "token_idx": ttt.token_idx.contiguous(),
        "learnable_token_idx": ttt.learnable_token_idx.contiguous(),
        # MLP parameters
        "up_proj_weight": block.mlp.up_proj.weight.contiguous(),
        "gate_proj_weight": block.mlp.gate_proj.weight.contiguous(),
        "down_proj_weight": block.mlp.down_proj.weight.contiguous(),
        # Config values as scalar tensors for Rust to read
        "config_mini_batch_size": torch.tensor([mini_batch_size], dtype=torch.int64),
        "config_pre_conv": torch.tensor([1 if cfg.pre_conv else 0], dtype=torch.int64),
        "config_share_qk": torch.tensor([1 if cfg.share_qk else 0], dtype=torch.int64),
        "config_use_gate": torch.tensor([1 if cfg.use_gate else 0], dtype=torch.int64),
    }
    _add_ttt_inner_weights(tensors, ttt, cfg.layer_type)
    if ttt.use_gate:
        tensors["g_proj_weight"] = ttt.g_proj.weight.contiguous()
    if cfg.share_qk:
        # Convolutions only exist when share_qk=True
        tensors["conv_q_weight"] = ttt.conv_q.weight.contiguous()
        tensors["conv_q_bias"] = ttt.conv_q.bias.contiguous()
        tensors["conv_k_weight"] = ttt.conv_k.weight.contiguous()
        tensors["conv_k_bias"] = ttt.conv_k.bias.contiguous()
    else:
        tensors["k_proj_weight"] = ttt.k_proj.weight.contiguous()
    if cfg.pre_conv:
        tensors["pre_conv_norm_weight"] = block.conv.norm.weight.contiguous()
        tensors["pre_conv_weight"] = block.conv.conv.weight.contiguous()
        tensors["pre_conv_bias"] = block.conv.conv.bias.contiguous()

    save_file(tensors, save_dir / f"block_{cfg.layer_type}.safetensors")

    return True


def validate_full_model(save_dir: Path, cfg: ValidationConfig):
    """
    Dispatch to reference TTTForCausalLM.forward() - full model.
    """
    print(f"\n--- TTTForCausalLM.forward (full model, {cfg.layer_type}) ---")
    torch.manual_seed(cfg.seed)

    B, L, H, D = cfg.B, cfg.L, cfg.H, cfg.D
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    mini_batch_size = cfg.mini_batch_size
    vocab_size = cfg.vocab_size
    num_layers = cfg.num_layers

    config = TTTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=H,
        mini_batch_size=mini_batch_size,
        ttt_layer_type=cfg.layer_type,
        ttt_base_lr=1.0,
        share_qk=cfg.share_qk,
        tie_word_embeddings=cfg.tie_word_embeddings,
        use_gate=cfg.use_gate,
        conv_kernel=cfg.conv_kernel,
        pre_conv=cfg.pre_conv,
    )

    model = TTTForCausalLM(config)
    model.eval()
    print(f"  shape: B={B}, L={L}, H={H}, D={D}, vocab={vocab_size}, layers={num_layers}")

    input_ids = torch.randint(0, vocab_size, (B, L))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits

    print(f"  input_ids: {input_ids.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}")

    tensors = {
        "input_ids": input_ids.contiguous(),
        "logits_expected": logits.contiguous(),
        "embed_weight": model.model.embed_tokens.weight.contiguous(),
        "final_norm_weight": model.model.norm.weight.contiguous(),
        # Config values as scalar tensors for Rust to read
        "config_mini_batch_size": torch.tensor([mini_batch_size], dtype=torch.int64),
        "config_num_layers": torch.tensor([num_layers], dtype=torch.int64),
        "config_pre_conv": torch.tensor([1 if cfg.pre_conv else 0], dtype=torch.int64),
        "config_share_qk": torch.tensor([1 if cfg.share_qk else 0], dtype=torch.int64),
        "config_tie_word_embeddings": torch.tensor([1 if cfg.tie_word_embeddings else 0], dtype=torch.int64),
        "config_use_gate": torch.tensor([1 if cfg.use_gate else 0], dtype=torch.int64),
    }

    # lm_head weight if not tied
    if not cfg.tie_word_embeddings:
        tensors["lm_head_weight"] = model.lm_head.weight.contiguous()

    # Save each layer's parameters with layer prefix
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"layer_{layer_idx}_"
        ttt = layer.seq_modeling_block

        # RMS norm parameters
        tensors[f"{prefix}seq_norm_weight"] = layer.seq_norm.weight.contiguous()
        tensors[f"{prefix}ffn_norm_weight"] = layer.ffn_norm.weight.contiguous()

        # TTT layer parameters
        tensors[f"{prefix}q_proj_weight"] = ttt.q_proj.weight.contiguous()
        if not cfg.share_qk:
            tensors[f"{prefix}k_proj_weight"] = ttt.k_proj.weight.contiguous()
        tensors[f"{prefix}v_proj_weight"] = ttt.v_proj.weight.contiguous()
        tensors[f"{prefix}o_proj_weight"] = ttt.o_proj.weight.contiguous()
        if ttt.use_gate:
            tensors[f"{prefix}g_proj_weight"] = ttt.g_proj.weight.contiguous()
        if cfg.share_qk:
            # Convolutions only exist when share_qk=True
            tensors[f"{prefix}conv_q_weight"] = ttt.conv_q.weight.contiguous()
            tensors[f"{prefix}conv_q_bias"] = ttt.conv_q.bias.contiguous()
            tensors[f"{prefix}conv_k_weight"] = ttt.conv_k.weight.contiguous()
            tensors[f"{prefix}conv_k_bias"] = ttt.conv_k.bias.contiguous()
        _add_ttt_inner_weights(tensors, ttt, cfg.layer_type, prefix)
        tensors[f"{prefix}ttt_norm_weight"] = ttt.ttt_norm_weight.contiguous()
        tensors[f"{prefix}ttt_norm_bias"] = ttt.ttt_norm_bias.contiguous()
        tensors[f"{prefix}post_norm_weight"] = ttt.post_norm.weight.contiguous()
        tensors[f"{prefix}post_norm_bias"] = ttt.post_norm.bias.contiguous()
        tensors[f"{prefix}lr_weight"] = ttt.learnable_ttt_lr_weight.contiguous()
        tensors[f"{prefix}lr_bias"] = ttt.learnable_ttt_lr_bias.contiguous()
        tensors[f"{prefix}token_idx"] = ttt.token_idx.contiguous()
        tensors[f"{prefix}learnable_token_idx"] = ttt.learnable_token_idx.contiguous()

        # MLP parameters
        tensors[f"{prefix}up_proj_weight"] = layer.mlp.up_proj.weight.contiguous()
        tensors[f"{prefix}gate_proj_weight"] = layer.mlp.gate_proj.weight.contiguous()
        tensors[f"{prefix}down_proj_weight"] = layer.mlp.down_proj.weight.contiguous()

        # Pre-conv parameters if enabled
        if cfg.pre_conv:
            tensors[f"{prefix}pre_conv_norm_weight"] = layer.conv.norm.weight.contiguous()
            tensors[f"{prefix}pre_conv_weight"] = layer.conv.conv.weight.contiguous()
            tensors[f"{prefix}pre_conv_bias"] = layer.conv.conv.bias.contiguous()

    save_file(tensors, save_dir / f"full_model_{cfg.layer_type}.safetensors")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="TTT Reference Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--B", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--L", type=int, default=16, help="Sequence length (default: 16)")
    parser.add_argument("--H", type=int, default=4, help="Number of heads (default: 4)")
    parser.add_argument("--D", type=int, default=16, help="Head dimension, must be even (default: 16)")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Mini-batch size, must divide L (default: 16)")
    parser.add_argument("--intermediate_mult", type=int, default=4, help="Intermediate size multiplier (default: 4)")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size (default: 1000)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--use_gate", action=argparse.BooleanOptionalAction, default=True, help="Use gating (default: True)")
    parser.add_argument("--conv_kernel", type=int, default=4, help="Convolution kernel size (default: 4)")
    parser.add_argument("--pre_conv", action=argparse.BooleanOptionalAction, default=False, help="Use pre-convolution (default: False)")
    parser.add_argument("--share_qk", action=argparse.BooleanOptionalAction, default=True, help="Share Q/K projection (default: True)")
    parser.add_argument("--tie_word_embeddings", action=argparse.BooleanOptionalAction, default=True, help="Tie word embeddings (default: True)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: validation_data_reference)")

    args = parser.parse_args()

    cfg = ValidationConfig(
        B=args.B,
        L=args.L,
        H=args.H,
        D=args.D,
        mini_batch_size=args.mini_batch_size,
        intermediate_mult=args.intermediate_mult,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        seed=args.seed,
        use_gate=args.use_gate,
        conv_kernel=args.conv_kernel,
        pre_conv=args.pre_conv,
        share_qk=args.share_qk,
        tie_word_embeddings=args.tie_word_embeddings,
    )

    print("=" * 70)
    print("TTT Reference Validation - Direct Dispatch")
    print("=" * 70)
    print(f"Config: B={cfg.B}, L={cfg.L}, H={cfg.H}, D={cfg.D}, mini_batch={cfg.mini_batch_size}, seed={cfg.seed}")
    print(f"        use_gate={cfg.use_gate}, conv_kernel={cfg.conv_kernel}, pre_conv={cfg.pre_conv}")
    print(f"        share_qk={cfg.share_qk}, tie_word_embeddings={cfg.tie_word_embeddings}")

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path(__file__).parent.parent / "validation_data_reference"
    save_dir.mkdir(exist_ok=True)

    results = []

    # Component-level tests
    results.append(("ln_fused_l2_bwd", validate_ln_fused_l2_bwd(save_dir, cfg)))
    results.append(("ln_fwd", validate_ln_fwd(save_dir, cfg)))
    results.append(("permute_qk", validate_permute_qk(save_dir, cfg)))

    # TTT inner model tests
    results.append(("ttt_linear_inner", validate_ttt_linear_full(save_dir, cfg)))
    results.append(("ttt_mlp_inner", validate_ttt_mlp_full(save_dir, cfg)))

    # Run layer/block/model tests for both linear and mlp
    from dataclasses import replace
    for layer_type in ["linear", "mlp"]:
        cfg_layer = replace(cfg, layer_type=layer_type)

        # Full layer test (QKV + conv + RoPE + gate + inner)
        results.append((f"ttt_layer_forward_{layer_type}", validate_ttt_layer_forward(save_dir, cfg_layer)))

        # Full block test (layer + FFN + residuals)
        results.append((f"block_forward_{layer_type}", validate_block_forward(save_dir, cfg_layer)))

        # Full model test (embedding + blocks + LM head)
        results.append((f"full_model_{layer_type}", validate_full_model(save_dir, cfg_layer)))

    print("\n" + "=" * 70)
    print("Results:")
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\nData saved to: {save_dir}")
    return all(r[1] for r in results)


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
