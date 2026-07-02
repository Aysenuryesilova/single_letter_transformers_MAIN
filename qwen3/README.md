# qwen3 — Qwen3 dense (the baseline)

The clean modern decoder that the other two folders are variations of.

**Read in this order:** `config.py` → `rms_norm.py` → `rotary.py` → `attention.py`
→ `mlp.py` → `block.py` → `model.py`.

Signature ingredients:
- **Pre-norm** blocks: `x += attn(norm(x)); x += mlp(norm(x))`
- **RMSNorm** (no mean subtraction, no bias)
- **QK-Norm**: RMSNorm on per-head queries/keys, then **RoPE**
- **GQA**: fewer key/value heads than query heads
- **SwiGLU** MLP, all linears bias-free, tied input/output embeddings

Run: `python3 train.py` then `python3 generate.py 20`.

Bonus: `python3 train_tiny3.py` trains the same recipe squeezed to
`hidden_size = 3` (one head, `head_dim = 2` — the smallest shape that still runs
the full Qwen3 recipe) and saves `tiny_qwen3.pt`; the demo notebook loads both.
