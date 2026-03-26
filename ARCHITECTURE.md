# Architecture notes

This is a walkthrough of the GPT-2 implementation in `model.py`, written as I was learning it. It covers the pieces I had to think hardest about.

---

## Overview

A GPT-style model is a stack of transformer decoder blocks that takes a sequence of token indices and predicts the next token at each position. The architecture here follows GPT-2 closely, minus the biases (`bias=False`) and with a different vocab size (68,096 from Sarvam's tokenizer vs GPT-2's 50,257).

The full forward pass:

```
token indices (B, T)
    → token embeddings (B, T, C) + position embeddings (T, C)
    → dropout
    → N × transformer block
    → LayerNorm
    → lm_head linear (B, T, vocab_size)
    → logits, loss
```

---

## Embeddings

Two learned lookup tables:
- `wte`: token embedding, shape `(vocab_size, n_embd)` — maps each token id to a vector
- `wpe`: position embedding, shape `(block_size, n_embd)` — maps each position 0..T-1 to a vector

These are summed and dropped before entering the blocks. Unlike sinusoidal positional encodings, both are learned from scratch.

---

## Causal self-attention

Each attention head learns to route information between positions, but only backwards — a token at position `t` can only attend to positions ≤ `t`. This is the "causal" constraint that makes autoregressive generation possible.

**QKV projection.** The Q, K, V matrices for all heads are computed in a single batched linear layer `c_attn`:

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

This is a `(C → 3C)` projection, then split into three `(C,)` chunks. Each is then reshaped from `(B, T, C)` into `(B, n_head, T, head_dim)` to separate the heads.

**Scaled dot-product attention.** We use PyTorch's `scaled_dot_product_attention` with `is_causal=True`, which fuses the softmax, masking, and dropout into an efficient kernel (FlashAttention when available):

```python
y = F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=True)
```

The `is_causal=True` flag applies the upper-triangular mask automatically — no need to register a manual bias buffer.

After attention, the heads are concatenated back and projected through `c_proj` to get the output.

---

## The transformer block

Each block applies attention and MLP in sequence, both with residual connections and pre-norm (LayerNorm before the sublayer, not after):

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

The residual stream `x` flows through all N blocks essentially unmodified — each block adds small corrections to it. This is what makes deep networks trainable: gradients flow back through the additions without vanishing.

**MLP.** A two-layer feed-forward net with a 4× expansion:

```
C → 4C (linear + GELU) → C (linear + dropout)
```

GELU is used instead of ReLU because it has a smoother gradient near zero, which helps early in training.

---

## Weight tying

The token embedding matrix `wte` and the final projection `lm_head` share weights:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This means the same `(vocab_size, n_embd)` matrix is used both to embed tokens at the input and to project the final hidden state to logits at the output. The reasoning: a token's embedding and its "score" as a prediction should live in the same space.

This also reduces the parameter count significantly — with `vocab_size=68096` and `n_embd=512`, weight tying saves ~35M parameters, which is why the reported count (~29M) is much less than the naive total (~64M).

---

## Initialization

Weights are initialized to `N(0, 0.02)`, following GPT-2. The residual projections (`c_proj`) get an additional scaling by `1/sqrt(2 * n_layer)`:

```python
torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

The idea: with N residual blocks, each adding to the stream, the variance of the stream grows as O(N) without this correction. Scaling each block's output by `1/sqrt(2N)` keeps the residual stream variance stable at initialization.

---

## The generate loop

Generation is autoregressive: feed a prompt, sample the next token, append it to the sequence, repeat.

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]          # crop if too long
    logits, _ = self(idx_cond)               # forward pass
    logits = logits[:, -1, :] / temperature  # last position only
    if top_k:
        # zero out all but the top k logits
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, 1)
    idx = torch.cat((idx, idx_next), dim=1)
```

A few things worth noting:

**Last position only.** At inference time, `forward()` only computes the lm_head on the final position (`x[:, [-1], :]`), not all T positions. This is a minor optimization since we only need the next-token distribution.

**Temperature.** Dividing logits by temperature < 1 sharpens the distribution (more repetitive, higher-confidence output). Temperature > 1 flattens it (more random). At temperature → 0 the model becomes greedy (always picks the argmax).

**Top-k sampling.** Before softmax, all logits outside the top-k are set to `-inf`. This prevents sampling from the long tail of unlikely tokens, which improves coherence without sacrificing diversity.

**No KV cache.** Every step re-runs the full forward pass over the entire sequence. A KV cache would store the K and V tensors from previous steps so only the new token needs to be processed — O(1) per step instead of O(T). This model doesn't implement one, so generation gets slower as the sequence grows.

---

## Parameter count

With `vocab_size=68096`, `n_embd=512`, `n_layer=8`, `n_head=8`:

| Component | Parameters |
|---|---|
| wte (token embedding) | 68096 × 512 = 34.9M |
| wpe (position embedding) | 256 × 512 = 0.1M |
| 8 × attention (QKV + proj) | 8 × (3×512² + 512²) = 8.4M |
| 8 × MLP (fc + proj) | 8 × (512×2048 + 2048×512) = 16.8M |
| 8 × LayerNorm (×2 per block) | negligible |
| **Total** | ~60M |
| minus wpe (not in count) | −0.1M |
| minus wte (weight-tied, counted once) | −34.9M |
| **Reported** | **~29M** |

The `get_num_params()` method subtracts `wpe` from the count (position embeddings don't really "train" the model in the same sense) and does not subtract `wte` because it's also `lm_head` — those weights are actively used.
