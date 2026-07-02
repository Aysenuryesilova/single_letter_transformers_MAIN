"""Train the tiny DeepSeek-V3-style (MLA + MoE) model on Turkish names.

Run:  python train.py
"""

import os

import torch

from config import ModelConfig
from model import TinyDeepSeek
from moe import MoE
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_isimler.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 16
STEPS = 5000           # sparse routing needs a little longer to settle than the dense models
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Tokenizer + data
# ---------------------------------------------------------------------------
tokenizer = CharTokenizer.from_file(DATA_FILE)
vocab_size = tokenizer.vocab_size

text = open(DATA_FILE, encoding="utf-8").read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model. The MoE point in one line: total params > params active per token.
# ---------------------------------------------------------------------------
cfg = ModelConfig(vocab_size=vocab_size)
model = TinyDeepSeek(cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
expert_params = sum(p.numel() for layer in model.layers if isinstance(layer.mlp, MoE)
                    for p in layer.mlp.experts[0].parameters())
n_moe_layers = sum(isinstance(layer.mlp, MoE) for layer in model.layers)
per_expert = expert_params // max(n_moe_layers, 1)
n_active = n_params - per_expert * (cfg.n_routed_experts - cfg.top_k) * n_moe_layers
print(f"device={device}  vocab_size={vocab_size}  parameters={n_params:,} "
      f"(active per token ~{n_active:,})")
print(f"layers: {['dense' if not isinstance(l.mlp, MoE) else 'moe' for l in model.layers]}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def sample_names(n: int = 10, max_new_tokens: int = 20):
    model.eval()
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0,
                         top_k=None, eos_id=tokenizer.eos_id)
    model.train()
    return [tokenizer.decode(row[1:]).split("\n")[0] for row in out.tolist()]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(1, STEPS + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % EVAL_EVERY == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")

# Show how evenly the load balancer spread tokens (ideal: ~0.25 per expert).
# `load` is recorded by each MoE during its last training forward pass.
for i, layer in enumerate(model.layers):
    if isinstance(layer.mlp, MoE):
        print(f"layer {i} expert load: {[round(v, 2) for v in layer.mlp.load.tolist()]}")

print("\nbaseline loss (uniform guessing): %.4f" % torch.log(torch.tensor(float(vocab_size))))
print("\nsample names:")
for name in sample_names(10):
    print("  ", name)

torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "cfg": cfg},
           "tiny_deepseek.pt")
print("\nsaved checkpoint to tiny_deepseek.pt")
