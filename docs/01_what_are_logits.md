# What Are Logits?

A language model does not predict a token directly. It predicts a real-valued **score**
for every token in the vocabulary. Those scores are called **logits**. For a vocabulary
of size V the model outputs a vector z ∈ ℝᵛ, where `z_i` is the unnormalized preference
for token `i`.

Think of the vocabulary as a scoreboard. After each forward pass the model writes a
number next to every possible next token. Higher means preferred; lower means disfavored.
The numbers are not probabilities — they can be any real value, and only their relative
order and spacing matter.

---

## The vocabulary scoreboard

```
  After generating "The quick brown ___", model produces one score per token:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  token         score (logit)   bar                                       │
  │  ────────────  ─────────────   ─────────────────────────────             │
  │  "fox"              3.10       ██████████████████████████                │
  │  "dog"              2.40       ████████████████████                      │
  │  "cat"              1.20       ██████████                                │
  │  "wolf"             0.50       █████                                     │
  │  "the"              0.30       ████                                      │
  │  "."               -0.80       ██                                        │
  │  "jumped"          -1.50       █                                         │
  │  ... 50,249 more vocabulary tokens ...                                   │
  └──────────────────────────────────────────────────────────────────────────┘

  Key facts:
  • Values can be positive, negative, or zero.
  • They carry no constraint — they do not sum to 1, and negatives are fine.
  • Only relative differences matter, not absolute scale.
  • A logit of 3.10 vs 2.40 is a stronger preference than 1.10 vs 0.40
    by the same gap, but the gap matters more after exponentiation (softmax).
```

---

## Softmax: turning logits into probabilities

A sampler needs a valid probability distribution: every entry non-negative and the
whole vector summing to 1. The softmax function provides that conversion:

```
  p_i = exp(z_i) / Σ_j exp(z_j)
```

Applied to the scoreboard above (six tokens for clarity):

```
  ┌──────────┬─────────┬──────────────┬─────────────┬──────────────────────────┐
  │  token   │  logit  │  exp(logit)  │  prob       │  probability bar         │
  ├──────────┼─────────┼──────────────┼─────────────┼──────────────────────────┤
  │  "fox"   │   3.10  │   22.198     │   0.549     │  ████████████████████    │
  │  "dog"   │   2.40  │   11.023     │   0.273     │  ██████████              │
  │  "cat"   │   1.20  │    3.320     │   0.082     │  ███                     │
  │  "the"   │   0.30  │    1.350     │   0.033     │  █                       │
  │  "."     │  -0.80  │    0.449     │   0.011     │  ░                       │
  │  "jumped"│  -1.50  │    0.223     │   0.006     │  ░                       │
  ├──────────┼─────────┼──────────────┼─────────────┼──────────────────────────┤
  │  total   │         │   38.563     │   1.000     │                          │
  └──────────┴─────────┴──────────────┴─────────────┴──────────────────────────┘

  Softmax is monotone: the ranking is preserved.
  Softmax is nonlinear: a +0.7 logit gap near the top (fox vs dog) gives 2× the
  probability ratio, while the same gap lower down (. vs jumped) barely matters.
```

---

## The full decoding chain

```
  raw logits  (shape: batch × vocab_size)
       │
       ▼
  ┌────────────────────────────────────────────┐
  │          LogitsProcessorList               │
  │                                            │
  │   ┌────────────────────────────────────┐   │
  │   │  TemperatureLogitsWarper           │   │   global rescale
  │   └───────────────┬────────────────────┘   │
  │                   ▼                        │
  │   ┌────────────────────────────────────┐   │
  │   │  RepetitionPenaltyLogitsProcessor  │   │   context-sensitive
  │   └───────────────┬────────────────────┘   │
  │                   ▼                        │
  │   ┌────────────────────────────────────┐   │
  │   │  TopKLogitsWarper                  │   │   hard truncation
  │   └───────────────┬────────────────────┘   │
  └───────────────────┼────────────────────────┘
                      │
                      ▼
                  softmax
                      │
                      ▼
             probability distribution
                      │
                      ▼
                multinomial sample  →  next token id
```

Logit processors live between the model output and softmax. They are the last chance
to shape the generated distribution before a token is committed.

---

## A worked numerical example

Vocabulary: five tokens. Raw logits from the model:

```
  token   logit    bar
  ──────  ───────  ───────────────────────────────
  "The"    2.40    ████████████████████
  "cat"    1.20    █████████████
  "sat"    0.10    ████
  "on"    -0.70    ██
  "."     -1.00    █
```

### Step 1 — TemperatureLogitsWarper(temperature=0.8)

Divide every logit by `T = 0.8`. Gaps widen; the leading token gets relatively
more dominant.

```
  z'_i = z_i / 0.8

  token   before   after    change    bar (after)
  ──────  ───────  ───────  ────────  ────────────────────────────────
  "The"    2.40     3.00    +0.60     ████████████████████████████████
  "cat"    1.20     1.50    +0.30     ██████████████████
  "sat"    0.10     0.13    +0.03     █████
  "on"    -0.70    -0.88    -0.18     ██
  "."     -1.00    -1.25    -0.25     █

  Temperature < 1 stretches the gap between logits proportionally.
  Temperature > 1 compresses it.
  Temperature = 1 is the identity.
```

### Step 2 — TopKLogitsWarper(top_k=3)

Keep the 3 highest-scoring tokens; set the rest to -∞.

```
  token   logit    kept?   bar (after masking)
  ──────  ───────  ──────  ────────────────────────────────────────────
  "The"    3.00     ✓      ████████████████████████████████
  "cat"    1.50     ✓      ██████████████████
  "sat"    0.13     ✓      █████
  "on"    -0.88     ✗      ░░░░░░░░░░░░░░░░░░░░░░░░░░░  (-inf)
  "."     -1.25     ✗      ░░░░░░░░░░░░░░░░░░░░░░░░░░░  (-inf)

  Setting to -inf guarantees exp(-inf) = 0, giving exact zero probability
  after softmax. Setting to 0 would not achieve this — a logit of 0 still
  gets a positive probability after softmax.
```

### Step 3 — Softmax

Exponentiate and normalize over the surviving logits:

```
  token   logit    exp(logit)   probability   probability bar
  ──────  ───────  ──────────   ───────────   ──────────────────────────
  "The"    3.00     20.086        0.782        ████████████████████████████████
  "cat"    1.50      4.482        0.174        ███████
  "sat"    0.13      1.138        0.044        ██
  "on"    -inf       0.000        0.000
  "."     -inf       0.000        0.000
  ──────           ──────────   ───────────
  sum               25.706        1.000
```

### Step 4 — Multinomial sample

Draw one token from the probability distribution [0.782, 0.174, 0.044, 0, 0].
"The" is most likely but not certain. Each step is stochastic (unless temperature ≈ 0).

---

## Temperature's effect on distribution shape

Temperature is the most direct knob for controlling how concentrated the distribution is.

```
  Same raw logits: [2.4, 1.2, 0.1, -0.7, -1.0]

  ┌─ T = 0.5 (sharper — dominant token much more likely) ───────────────────────┐
  │  "The"   ████████████████████████████████  0.906                            │
  │  "cat"   ██                                0.082                            │
  │  "sat"   ░                                 0.009                            │
  │  "on"    ░                                 0.002                            │
  │  "."     ░                                 0.001                            │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ T = 1.0 (unchanged — model's natural distribution) ───────────────────────┐
  │  "The"   ████████████████████              0.675                            │
  │  "cat"   ██████                            0.203                            │
  │  "sat"   ██                                0.068                            │
  │  "on"    █                                 0.031                            │
  │  "."     ░                                 0.023                            │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ T = 2.0 (flatter — mass spread across more tokens) ───────────────────────┐
  │  "The"   █████████████                     0.442                            │
  │  "cat"   ████████                          0.243                            │
  │  "sat"   █████                             0.140                            │
  │  "on"    ███                               0.093                            │
  │  "."     ██                                0.082                            │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why work in log-space? Four reasons

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Reason 1: Underflow safety                                                 │
  │                                                                             │
  │  Rare tokens can have probability ≈ 10⁻³⁰.                                 │
  │  In float16, that number underflows to exactly 0.0 — information lost.     │
  │  In log-space: log(10⁻³⁰) ≈ -69.1 — perfectly representable.              │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Reason 2: Simple, composable operations                                    │
  │                                                                             │
  │  Temperature:    z' = z / T         (one division, no re-normalization)    │
  │  Hard masking:   z' = -inf          (one assignment)                       │
  │  Additive bias:  z' = z + b         (stays additive, no simplex math)      │
  │                                                                             │
  │  In probability space each of these requires renormalization afterward.    │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Reason 3: Translation invariance of softmax                                │
  │                                                                             │
  │  softmax(z) == softmax(z + c) for any constant c                           │
  │                                                                             │
  │  Practical consequence: subtract max(z) before calling exp() to prevent    │
  │  overflow. The distribution is identical. This trick is free in log-space.  │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Reason 4: Simplex constraint deferred                                      │
  │                                                                             │
  │  Probabilities must satisfy p_i ≥ 0 and Σ p_i = 1 after every edit.       │
  │  Maintaining this through multiple processing steps requires repeated       │
  │  normalization passes.                                                      │
  │  Log-space avoids all that bookkeeping until the single final softmax.     │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Where processors sit in the generation loop

Each autoregressive step follows this exact path:

```
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║  Input: current token sequence  [t₁, t₂, ..., t_n]                       ║
  ║      │                                                                    ║
  ║      ▼                                                                    ║
  ║  ┌────────────────────────────┐                                           ║
  ║  │   token embedding lookup   │  maps each id → dense vector              ║
  ║  └────────────────┬───────────┘                                           ║
  ║                   ▼                                                       ║
  ║  ┌─────────────────────────────────────────────────────┐                  ║
  ║  │   transformer blocks (N layers)                     │                  ║
  ║  │   • multi-head self-attention (reads KV-cache)      │                  ║
  ║  │   • feed-forward MLP                                │                  ║
  ║  │   • layer normalization                             │                  ║
  ║  └────────────────┬────────────────────────────────────┘                  ║
  ║                   │                                                       ║
  ║                   ▼                                                       ║
  ║  ┌────────────────────────────┐                                           ║
  ║  │   LM head projection       │  hidden_size → vocab_size                 ║
  ║  │   raw logits: ℝ^vocab_size  │                                           ║
  ║  └────────────────┬───────────┘                                           ║
  ║                   │                                                       ║
  ║   ╔═══════════════╪══════════════════════════════════╗                    ║
  ║   ║               ▼      LogitsProcessorList         ║                    ║
  ║   ║   ┌─────────────────────────────────────────┐   ║                    ║
  ║   ║   │  processor 0  (e.g. TemperatureWarper)  │   ║                    ║
  ║   ║   └───────────────────┬─────────────────────┘   ║                    ║
  ║   ║                       ▼                         ║                    ║
  ║   ║   ┌─────────────────────────────────────────┐   ║                    ║
  ║   ║   │  processor 1  (e.g. RepetitionPenalty)  │   ║                    ║
  ║   ║   └───────────────────┬─────────────────────┘   ║                    ║
  ║   ║                       ▼                         ║                    ║
  ║   ║   ┌─────────────────────────────────────────┐   ║                    ║
  ║   ║   │  processor 2  (e.g. TopKWarper)         │   ║                    ║
  ║   ║   └───────────────────┬─────────────────────┘   ║                    ║
  ║   ╚═══════════════════════╪══════════════════════════╝                    ║
  ║                           │                                               ║
  ║                           ▼                                               ║
  ║  ┌────────────────────────────┐                                           ║
  ║  │   softmax → sample         │  → next token id  t_{n+1}                 ║
  ║  └────────────────┬───────────┘                                           ║
  ║                   ▼                                                       ║
  ║  append t_{n+1}; update KV-cache; loop or stop                            ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
```

**What processors can and cannot do:**

| Can do | Cannot do |
|---|---|
| Rescale any logit | Modify model weights |
| Hard-mask any token to -inf | Access hidden states or attention |
| Apply context-sensitive penalties | Change the vocabulary size |
| Carry per-batch stateful accumulators | Affect KV-cache content |
| Compose with any other processor | Alter the token embedding table |

Processors are purely a transformation on the final score vector. That constraint
keeps them fast, composable, and safe to chain in any order — with the caveat that
ordering changes semantics (see [Architecture](02_architecture.md)).
