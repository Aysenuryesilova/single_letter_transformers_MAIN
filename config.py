"""Model configuration for the tiny Qwen3-style transformer.

Everything the model needs to know about its own shape lives in this one
dataclass. The defaults are deliberately tiny so training runs on a CPU in
seconds while still using the real Qwen3 dense recipe.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30        # 30 Turkish chars (incl. newline)
    hidden_size: int = 32       # model / embedding dimension
    num_layers: int = 2         # number of transformer blocks
    num_heads: int = 4          # number of query heads
    num_kv_heads: int = 2       # number of key/value heads (GQA: 2 query heads share each KV head)
    head_dim: int = 8           # dimension per head (= hidden_size / num_heads)
    intermediate_size: int = 64   # SwiGLU hidden dimension (~2x hidden_size)
    max_seq_len: int = 24       # longest sequence we ever feed in (names are short)
    rope_theta: float = 10000.0  # RoPE base frequency
    rms_norm_eps: float = 1e-6  # epsilon inside RMSNorm
