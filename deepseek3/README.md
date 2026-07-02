# deepseek3 — DeepSeek-V3 sparse (MLA + MoE)

Same skeleton as [`qwen3/`](../qwen3/), with both halves of the block swapped out
for DeepSeek-V3's signature pieces.

**The new files to read:** [`mla.py`](mla.py) and [`moe.py`](moe.py).

**MLA (Multi-head Latent Attention).** K and V are not projected straight from
the hidden state — the token is first compressed into a tiny **KV latent**
(`kv_lora_rank` wide) and K/V for *all* heads are re-expanded from it. At
inference only the latent would ever be cached, which is the whole point
(GQA shrinks the cache by sharing heads; MLA shrinks it harder without sharing).
RoPE can't pass through the re-expansion, so position lives in a few extra
**decoupled rope dims**: queries get their own per head, keys get one shared
copy, and q/k are the concatenation `[nope | rope]`.

**MoE with aux-loss-free balancing.** Layers after the first replace the MLP
with 4 small expert MLPs plus one always-on **shared expert**. A sigmoid router
picks the **top-2** experts per token — selection uses `scores + expert_bias`,
but output weights use the raw scores, so the bias only steers *who* gets
picked. After every training step, overloaded experts get their bias nudged
down and idle ones up (`bias_update_speed`), balancing load **without an
auxiliary loss** fighting the LM objective. `train.py` prints the final
per-expert load (ideal: `0.25` each) and the active-vs-total parameter count.

**Simplified for clarity:** no inference-time weight absorption (real MLA never
materializes K/V), 4 routed experts instead of 256, no node-limited routing or
sequence-wise balance loss, no multi-token prediction; the expert dispatch is a
plain Python loop.

Run: `python3 train.py` then `python3 generate.py 20`.
