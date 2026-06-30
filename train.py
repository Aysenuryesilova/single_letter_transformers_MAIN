"""Train the tiny Qwen3-style model to generate Turkish names.

Run:  python train.py

This script is intentionally linear and dependency-free (just torch). It reads
the cleaned names, builds a trivial character vocabulary inline, trains with a
plain loop, then samples a few names.
"""

import torch

from config import ModelConfig
from model import TinyQwen
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_FILE = "temiz_isimler.txt"
BATCH_SIZE = 64
BLOCK_SIZE = 16        # context length used during training (<= cfg.max_seq_len)
STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Tokenizer (character level). See tokenizer.py.
# ---------------------------------------------------------------------------
tokenizer = CharTokenizer.from_file(DATA_FILE)
vocab_size = tokenizer.vocab_size

text = open(DATA_FILE, encoding="utf-8").read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # [N] whole corpus as ids


def get_batch():
    """Sample BATCH_SIZE random windows. Targets are inputs shifted by one."""
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
cfg = ModelConfig(vocab_size=vocab_size)
model = TinyQwen(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={device}  vocab_size={vocab_size}  parameters={n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# Sampling helper: generate a few names starting from the newline token.
# ---------------------------------------------------------------------------
def sample_names(n: int = 10, max_new_tokens: int = 20):
    model.eval()
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0, top_k=None)
    model.train()
    names = []
    for row in out.tolist():
        # Drop the leading newline, then keep up to the next newline.
        s = tokenizer.decode(row[1:])
        names.append(s.split("\n")[0])
    return names


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

print("\nbaseline loss (uniform guessing): %.4f" % (torch.log(torch.tensor(float(vocab_size)))))
print("\nsample names:")
for name in sample_names(10):
    print("  ", name)

torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "cfg": cfg},
           "tiny_qwen.pt")
print("\nsaved checkpoint to tiny_qwen.pt")
