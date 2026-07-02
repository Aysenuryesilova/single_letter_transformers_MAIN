"""Model configuration for the tiny DeepSeek-V3-style transformer.

DeepSeek-V3's two signature ideas (vs the dense Qwen3 baseline):

  * MLA (Multi-head Latent Attention): keys and values are not projected
    directly from the hidden state. Instead the hidden state is compressed
    into one small "KV latent" per token, and K/V are re-expanded from it.
    The tiny latent is all you would ever cache — that is the whole point.
    RoPE gets its own small "decoupled" dimensions, shared across heads.

  * MoE (Mixture of Experts): instead of one big MLP per block, many small
    expert MLPs — each token is routed to only `top_k` of them, plus one
    always-on shared expert. Load is balanced *aux-loss-free* with a per-expert
    bias that is nudged up/down depending on how busy the expert was.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30          # 30 Turkish chars (incl. newline)
    hidden_size: int = 32         # model / embedding dimension
    num_layers: int = 3           # first layer dense, the rest MoE (see below)
    num_heads: int = 4            # attention heads (MLA has no GQA; all heads share the KV latent)
    max_seq_len: int = 24         # longest sequence we ever feed in (names are short)
    rope_theta: float = 10000.0   # RoPE base frequency
    rms_norm_eps: float = 1e-6    # epsilon inside RMSNorm

    # ---- MLA (Multi-head Latent Attention) --------------------------------
    q_lora_rank: int = 16         # rank of the query compression
    kv_lora_rank: int = 8         # rank of the KV compression (the tiny "cache")
    qk_nope_head_dim: int = 8     # per-head q/k dims WITHOUT position info
    qk_rope_head_dim: int = 4     # decoupled RoPE dims (k part is shared across heads)
    v_head_dim: int = 8           # per-head value dims

    # ---- MoE ---------------------------------------------------------------
    first_dense_layers: int = 1   # DeepSeek-V3 keeps the first layers dense
    n_routed_experts: int = 4     # small expert MLPs to route between
    n_shared_experts: int = 1     # always-on experts (fused into one MLP)
    top_k: int = 2                # experts chosen per token
    moe_intermediate_size: int = 32   # hidden dim of ONE expert (smaller than the dense MLP)
    intermediate_size: int = 64   # hidden dim of the dense layers' SwiGLU
    bias_update_speed: float = 1e-3   # gamma: step size of the load-balancing bias
