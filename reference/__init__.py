# Reference TTT implementation from official PyTorch repo
# CPU-only for compatibility

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

from .ttt import (
    TTTConfig,
    TTTLinear,
    TTTMLP,
    TTTBase,
    Block,
    TTTModel,
    TTTForCausalLM,
    ln_fwd,
    ln_fused_l2_bwd,
    permute_qk,
    undo_permute_qk,
    apply_rotary_pos_emb,
    rotate_half,
    RMSNorm,
    SwiGluMLP,
    RotaryEmbedding,
)
