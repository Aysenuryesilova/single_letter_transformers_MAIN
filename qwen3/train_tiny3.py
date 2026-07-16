"""Train an *extra* tiny Qwen3-style model with hidden_size = 3.

Same task and code as train.py, but the model is shrunk to the smallest shape
that still runs the full Qwen3 recipe:

  * hidden_size    = 3   (the whole point)
  * num_heads      = 1   single query head
  * num_kv_heads   = 1   (GQA degenerates to plain attention)
  * head_dim       = 2   RoPE needs an even head_dim, so 2 is the minimum
  * intermediate_size = 6

Run:  python train_tiny3.py
"""

import os

import torch

from config import ModelConfig
from model import TinyQwen
from tokenizer import CharTokenizer

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_isimler.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 16
STEPS = 8000
LEARNING_RATE = 3e-3
EVAL_EVERY = 500
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

tokenizer = CharTokenizer.from_file(DATA_FILE)
vocab_size = tokenizer.vocab_size

text = open(DATA_FILE, encoding="utf-8").read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# The only real change from train.py: a much smaller ModelConfig.
cfg = ModelConfig(
    vocab_size=vocab_size,
    hidden_size=3,
    num_layers=2,
    num_heads=1,
    num_kv_heads=1,
    head_dim=2,
    intermediate_size=2,
)
model = TinyQwen(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={device}  vocab_size={vocab_size}  parameters={n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def sample_names(n: int = 10, max_new_tokens: int = 20):
    model.eval()
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0,
                         top_k=None, eos_id=tokenizer.eos_id)
    model.train()
    names = []
    for row in out.tolist():
        s = tokenizer.decode(row[1:])
        names.append(s.split("\n")[0])
    return names


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
           "tiny_qwen3.pt")
print("\nsaved checkpoint to tiny_qwen3.pt")
