"""Generate Turkish names from a trained checkpoint.

Run:  python generate.py            # 20 names, default settings
      python generate.py 50         # 50 names
      python generate.py 50 0.8     # 50 names, temperature 0.8

Lower temperature -> safer, more common names. Higher -> more varied/creative.
"""

import sys
import torch

from model import TinyDeepSeek
from tokenizer import CharTokenizer

CHECKPOINT = "tiny_deepseek.pt"


def load():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    tokenizer = CharTokenizer(ckpt["chars"])
    model = TinyDeepSeek(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_names(model, tokenizer, n: int, temperature: float, top_k=None):
    # Every name starts from the newline (start-of-name) token.
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    names = []
    for row in out.tolist():
        # Drop the leading newline, then keep everything up to the next newline.
        name = tokenizer.decode(row[1:]).split("\n")[0]
        if name:
            names.append(name)
    return names


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    model, tokenizer = load()
    for name in generate_names(model, tokenizer, n, temperature):
        print(name)


if __name__ == "__main__":
    main()
