# Processor Reference

This reference documents all six built-in processors. Each section gives the
mathematical rule, a step-by-step visualization of what the distribution looks like
before and after, configuration guidance, interaction effects, and known failure modes.

---

## Quick-reference table

| Processor | Uses prefix? | Operation | Key param | Valid range | Default |
|---|---|---|---|---|---|
| `TemperatureLogitsWarper` | No | `z / T` | `temperature` | `(0, ∞)` | `1.0` |
| `TopKLogitsWarper` | No | keep top k | `top_k` | `[1, ∞)` | `50` |
| `TopPLogitsWarper` | No | keep nucleus | `top_p` | `(0, 1]` | `1.0` |
| `RepetitionPenaltyLogitsProcessor` | Yes | ratio penalty | `penalty` | `[1.0, ∞)` | `1.0` |
| `TypicalDecodingLogitsWarper` | No | entropy mask | `mass` | `(0, 1]` | `0.9` |
| `FrequencyPenaltyLogitsProcessor` | Yes | additive penalty | `penalty` | `[0, ∞)` | `0.0` |

---

## TemperatureLogitsWarper

### Formula

```
  z'_i = z_i / T          for T >= 1e-7

  if T < 1e-7:             (near-zero temperature → greedy approximation)
      z'_i = z_i           if i == argmax(z)
      z'_i = -inf          otherwise
```

### What it does

Temperature rescales the gap between all logits simultaneously. Lower temperature
widens the gaps (leading token dominates more). Higher temperature compresses the
gaps (mass spreads across more tokens).

```
  Raw logits: [2.4, 1.2, 0.1, -0.7, -1.0]

  ┌─ T = 0.5  (sharper distribution) ─────────────────────────────────────────┐
  │  logits:  [4.80, 2.40, 0.20, -1.40, -2.00]                               │
  │  probs:   [0.906, 0.082, 0.009, 0.002, 0.001]                             │
  │                                                                            │
  │  t0  █████████████████████████████████████  0.906  "The"                  │
  │  t1  ████                                   0.082  "cat"                  │
  │  t2  ░                                      0.009  "sat"                  │
  │  t3  ░                                      0.002  "on"                   │
  │  t4  ░                                      0.001  "."                    │
  └────────────────────────────────────────────────────────────────────────────┘

  ┌─ T = 1.0  (model's own distribution, no change) ──────────────────────────┐
  │  logits:  [2.40, 1.20, 0.10, -0.70, -1.00]                               │
  │  probs:   [0.675, 0.203, 0.068, 0.031, 0.023]                             │
  │                                                                            │
  │  t0  ████████████████████████████            0.675  "The"                  │
  │  t1  █████████                               0.203  "cat"                  │
  │  t2  ███                                     0.068  "sat"                  │
  │  t3  ██                                      0.031  "on"                   │
  │  t4  █                                       0.023  "."                    │
  └────────────────────────────────────────────────────────────────────────────┘

  ┌─ T = 2.0  (flatter distribution) ─────────────────────────────────────────┐
  │  logits:  [1.20, 0.60, 0.05, -0.35, -0.50]                               │
  │  probs:   [0.442, 0.243, 0.140, 0.093, 0.082]                             │
  │                                                                            │
  │  t0  ██████████████████                      0.442  "The"                  │
  │  t1  ██████████                              0.243  "cat"                  │
  │  t2  ██████                                  0.140  "sat"                  │
  │  t3  ████                                    0.093  "on"                   │
  │  t4  ████                                    0.082  "."                    │
  └────────────────────────────────────────────────────────────────────────────┘
```

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `temperature` | float | `(0, ∞)` | `< 1` sharpens; `= 1` identity; `> 1` flattens |

### Interactions

- Applied before top-k or top-p, temperature changes which tokens fall inside the
  nucleus or kept set. Lower temperature makes the set more conservative (fewer tokens
  survive at the same top-p threshold).
- Applied before repetition penalty, temperature changes the effective magnitude of the
  penalty because penalty operates as a ratio on the already-scaled logit values.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `temperature <= 0` |
| Near-greedy behavior | `temperature` very close to 0 |
| Semantically incoherent output | `temperature` far above 1.5 |

---

## TopKLogitsWarper

### Formula

```
  S = indices of the k largest scores in scores[b, :]   (per batch row b)

  z'_i = z_i           if i ∈ S
  z'_i = filter_value  otherwise  (default: -inf)
```

Effective k is `max(min(top_k, vocab_size), min_tokens_to_keep)`.

### What it does

Top-k is a **hard count filter**. It keeps exactly k tokens regardless of how spread
out the distribution is. When the model is confident, k tokens may cover 99% of the
mass. When the model is uncertain, k tokens may cover only 30%.

```
  Raw scores (sorted descending for clarity):

  t0  ████████████████████  3.60  ─────────────── ✓ kept (k=4)
  t1  ████████████████      3.10  ─────────────── ✓ kept
  t2  ████████████          2.50  ─────────────── ✓ kept
  t3  ████████              1.80  ─────────────── ✓ kept
  t4  ████                  0.70  ──────cut────── ✗ masked
  t5  ███                   0.40                  ✗ masked
  t6  ██                   -0.30                  ✗ masked
  t7  █                    -1.20                  ✗ masked

  After TopKLogitsWarper(top_k=4):
  t0  ████████████████████  3.60  ✓
  t1  ████████████████      3.10  ✓
  t2  ████████████          2.50  ✓
  t3  ████████              1.80  ✓
  t4  ░░░░░░░░░░░░░░░░░░░░  -inf  ✗
  t5  ░░░░░░░░░░░░░░░░░░░░  -inf  ✗
  t6  ░░░░░░░░░░░░░░░░░░░░  -inf  ✗
  t7  ░░░░░░░░░░░░░░░░░░░░  -inf  ✗
```

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `top_k` | int | `[1, ∞)` | Number of tokens to keep |
| `filter_value` | float | any | Score written to masked positions (default: `-inf`) |
| `min_tokens_to_keep` | int | `[1, ∞)` | Safety floor; ensures at least this many tokens survive |

### Interactions

- Lower temperature before top-k makes the kept set more conservative: the logit gap
  between ranks narrows so fewer tokens have meaningfully similar scores.
- Top-k before repetition penalty leaves room for a repeated token to survive even
  after penalty. Top-k after penalty can exclude it entirely.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `top_k < 1` |
| All-inf row (empty distribution) | Custom `filter_value` + `min_tokens_to_keep=0` |
| Softmax includes unlikely tokens | `filter_value` is finite rather than `-inf` |

---

## TopPLogitsWarper

### Formula

```
  Sort tokens descending by probability: p_(1) ≥ p_(2) ≥ ... ≥ p_(V)

  Find the smallest prefix K such that:
      Σ_{j=1..K} p_(j)  ≥  top_p

  Keep that prefix; mask everything else.

  z'_i = z_i           if rank(i) ≤ K
  z'_i = filter_value  otherwise
```

The token that first pushes the cumulative sum over `top_p` is **included**, not
excluded. This matches the Holtzman et al. (2020) nucleus definition.

### What it does

Top-p is an **adaptive count filter**. It keeps a variable number of tokens depending
on how concentrated the distribution is. For a sharp distribution, the nucleus may
contain 2–3 tokens. For a flat distribution, it may contain hundreds.

```
  Sharp distribution (model is confident):
  ─────────────────────────────────────────────────────
  Cumulative mass after adding each token (sorted):

  t0   prob=0.72   cumsum=0.72   ██████████████████████████████  cumsum < 0.90
  t1   prob=0.15   cumsum=0.87   ██████████████████████████████  cumsum < 0.90
  t2   prob=0.06   cumsum=0.93   ██████████████████████████████  cumsum ≥ 0.90  ← cutoff
  t3   prob=0.04   cumsum=0.97   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  masked
  t4   prob=0.03   cumsum=1.00   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  masked

  Nucleus size = 3 tokens  (covers 93% of mass)

  ─────────────────────────────────────────────────────
  Flat distribution (model is uncertain):
  ─────────────────────────────────────────────────────
  t0   prob=0.21   cumsum=0.21   not enough
  t1   prob=0.19   cumsum=0.40   not enough
  t2   prob=0.15   cumsum=0.55   not enough
  t3   prob=0.12   cumsum=0.67   not enough
  t4   prob=0.10   cumsum=0.77   not enough
  t5   prob=0.08   cumsum=0.85   not enough
  t6   prob=0.06   cumsum=0.91   ← cutoff at 0.90
  t7   prob=0.05   cumsum=0.96   ░░░░ masked
  t8   prob=0.04   cumsum=1.00   ░░░░ masked

  Nucleus size = 7 tokens  (same top_p=0.9, more tokens needed)
  ─────────────────────────────────────────────────────

  This adaptive behavior is top-p's key advantage over top-k.
```

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `top_p` | float | `(0, 1]` | Minimum cumulative mass to preserve |
| `filter_value` | float | any | Score written to masked positions (default: `-inf`) |
| `min_tokens_to_keep` | int | `[1, ∞)` | Safety floor (default: `1`) |

### Interactions

- Highly sensitive to temperature. High temperature flattens the distribution → larger
  nucleus for the same `top_p`. Low temperature sharpens → smaller nucleus.
- Running top-p and top-k together is common: top-k bounds the absolute maximum size;
  top-p adapts within that bound.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `top_p <= 0` or `top_p > 1` |
| Near-greedy behavior | `top_p` very close to 0 |
| Nucleus mass slightly under threshold | Off-by-one in shift logic (built-in is correct) |

---

## RepetitionPenaltyLogitsProcessor

### Formula

```
  For each token i that appears anywhere in input_ids[b, :]:

      z'_i = z_i / penalty     if z_i > 0   (positive logit pushed down)
      z'_i = z_i * penalty     if z_i < 0   (negative logit pushed further down)
      z'_i = z_i               if z_i = 0   (zero is unchanged)
```

This asymmetric rule ensures that both positive and negative logits for a repeated
token are penalized in the same direction (toward lower probability), regardless of
their sign.

### What it does

```
  Prefix contains: ["The", "cat", "sat", "on", "the"]
  Repeated tokens: "The" (id=464), "the" (id=262), "cat" (id=3797) ...

  Raw logits for a few tokens:
  ──────────────────────────────────────────────────
  "The"   ████████████████████  4.20  (seen)
  "cat"   ████████████          2.80  (seen)
  "fox"   █████████████████     3.60  (unseen)
  "on"    █████                 1.10  (seen)
  "end"   ████                  0.70  (unseen)

  After RepetitionPenaltyLogitsProcessor(penalty=1.3):
  ──────────────────────────────────────────────────
  "The"   ████████████████      3.23  4.20/1.3  ↓ penalized
  "cat"   █████████             2.15  2.80/1.3  ↓ penalized
  "fox"   █████████████████     3.60  unchanged (unseen)
  "on"    ████                  0.85  1.10/1.3  ↓ penalized
  "end"   ████                  0.70  unchanged (unseen)

  Effect: "fox" and "end" become relatively more likely. Repeated tokens
  are still possible — just downweighted.
```

### Why the sign-dependent rule?

```
  Suppose z_i = -2.0  (token already disfavored by the model)

  Dividing:   -2.0 / 1.3 = -1.54   → logit increases → probability increases! Wrong.
  Multiplying: -2.0 * 1.3 = -2.60  → logit decreases → probability decreases. Correct.

  That is why positive logits are divided and negative logits are multiplied.
  Both directions push the token toward being less likely.
```

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `penalty` | float | `[1.0, ∞)` | `= 1.0` is no-op; higher values suppress repetition more |

### Interactions

- Strongest when applied before top-k or top-p: the penalty can push a repeated token
  below the truncation threshold, removing it entirely.
- Multiple occurrences of the same token do **not** compound — a token seen 10 times
  receives the same penalty as a token seen once. Use `FrequencyPenaltyLogitsProcessor`
  if compounding is desired.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `penalty < 1.0` |
| Named entities, code identifiers suppressed | Penalty too high (`> 1.5`) |
| Repetition still present | Using after top-k; repeated token already in kept set |

---

## TypicalDecodingLogitsWarper

### Formula

```
  log_p_i  = log_softmax(z)_i                  (log-probability of each token)
  H(p)     = -Σ_i p_i * log_p_i                (distribution entropy)
  d_i      = | -log_p_i - H(p) |               (deviation from typical information)

  Sort tokens by increasing d_i (most typical first).
  Keep the smallest prefix whose cumulative probability ≥ mass.
  Mask the rest with filter_value.
```

### What it does

Typical decoding (Meister et al., 2023) keeps tokens whose **information content**
`-log p(x)` is close to the distribution's entropy `H(p)`. It removes both extremes:
tokens that are too predictable (information content far below entropy) and tokens
that are too surprising (information content far above entropy).

```
  Example distribution:
  ─────────────────────────────────────────────────────────────────────────────
  token   prob    -log(p)   |(-log p) - H|   typicality rank   kept (mass=0.9)?
  ──────  ──────  ────────  ─────────────    ─────────────────  ───────────────
  t0      0.50     0.69        0.92          most atypical (too predictable)   ✗
  t1      0.22     1.51        0.10          2nd most typical                  ✓
  t2      0.15     1.90        0.29          3rd most typical                  ✓
  t3      0.08     2.53        0.92          most atypical (too surprising)    ✗
  t4      0.05     2.99        1.38          most atypical (too surprising)    ✗
  ──────────────────────────────────────────────────────────────────────────────
  H(p) = 1.61 nats   (from -Σ p log p)
  ─────────────────────────────────────────────────────────────────────────────

  Ranked by deviation d_i (ascending):
  t1  d=0.10  cumsum=0.22  ✓ keep
  t2  d=0.29  cumsum=0.37  ✓ keep
  t0  d=0.92  cumsum=0.87  ✓ keep  (cumsum crosses 0.9 — include this one, stop after)
  t3  d=0.92  cumsum=0.95  ✗ masked
  t4  d=1.38  cumsum=1.00  ✗ masked

  After masking:
  t0  ████████████████████  (kept, though least typical of survivors)
  t1  ██████████████████    (kept — most typical)
  t2  ██████████████        (kept — second most typical)
  t3  ░░░░░░░░░░░░░░░░░░░░  masked — too surprising
  t4  ░░░░░░░░░░░░░░░░░░░░  masked — too surprising
```

**Contrast with top-p:** top-p ranks by raw probability (keep the most likely tokens).
Typical decoding ranks by deviation from entropy (keep the "normally surprising"
tokens). On a highly confident distribution, top-p keeps only the single top token;
typical decoding may also drop it if it is too predictable.

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `mass` | float | `(0, 1]` | Cumulative mass to preserve after typical sorting |
| `filter_value` | float | any | Score written to masked positions (default: `-inf`) |
| `min_tokens_to_keep` | int | `[1, ∞)` | Safety floor (default: `1`) |

### Interactions

- Sensitive to temperature: high temperature flattens the distribution → entropy
  increases → deviation scores shift → different tokens survive.
- If the distribution is uniform (all tokens equally likely), every token has the same
  deviation (0). The warper returns all tokens unchanged rather than creating arbitrary
  asymmetry.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `mass <= 0` or `mass > 1` |
| Near-greedy behavior | `mass` very close to 0 |
| All tokens kept | Uniform distribution detected |

---

## FrequencyPenaltyLogitsProcessor

### Formula

```
  c_i   = count of token i in input_ids[b, :]   (per batch row b)

  z'_i  = z_i - penalty * c_i
```

This is a purely additive, count-proportional penalty. Each additional occurrence of
a token subtracts another `penalty` from its logit.

### What it does

```
  Prefix counts:  "the" appears 3 times, "cat" 1 time, "fox" 0 times

  Raw logits and frequency penalty (penalty=0.4):
  ──────────────────────────────────────────────────────────────────
  token   raw logit   count   penalty applied   final logit
  ──────  ─────────   ─────   ───────────────   ───────────
  "the"     3.50        3      -0.4 × 3 = -1.2    2.30
  "cat"     2.80        1      -0.4 × 1 = -0.4    2.40
  "fox"     2.60        0      -0.4 × 0 = 0.0     2.60
  ──────────────────────────────────────────────────────────────────

  Before penalty:                  After penalty (penalty=0.4):
  "the"  █████████████████  3.50   "the"  ████████████      2.30  ↓ -1.20
  "cat"  ██████████████     2.80   "cat"  █████████████     2.40  ↓ -0.40
  "fox"  █████████████      2.60   "fox"  █████████████     2.60  unchanged

  Effect: "fox" (unseen) becomes most likely. "the" (3× seen) falls behind
  even "cat" (1× seen). Compounding makes very frequent tokens progressively
  less likely.
```

### Frequency vs repetition penalty

```
  ┌──────────────────────┬──────────────────────┬──────────────────────────┐
  │                      │ RepetitionPenalty     │ FrequencyPenalty         │
  ├──────────────────────┼──────────────────────┼──────────────────────────┤
  │ Operation            │ ratio (÷ or ×)        │ subtraction              │
  │ Count sensitivity    │ No — seen once = seen │ Yes — 3× seen = 3× hit  │
  │                      │  many times           │                          │
  │ Valid range          │ penalty ≥ 1.0         │ penalty ≥ 0.0            │
  │ Effect on scale      │ Relative (adapts to   │ Absolute (depends on     │
  │                      │ logit magnitude)      │ logit scale)             │
  │ Good for             │ Hard suppression of   │ Gradual diversity nudge  │
  │                      │ any repetition        │ proportional to count    │
  └──────────────────────┴──────────────────────┴──────────────────────────┘
```

### Parameters

| Parameter | Type | Range | Effect |
|---|---|---|---|
| `penalty` | float | `[0, ∞)` | `= 0` is no-op; higher values discourage frequent tokens more |

### Interactions

- Works well combined with top-p because it gently redistributes mass rather than
  hard-masking. The nucleus remains adaptive.
- Because the penalty is additive, its effective strength depends on the logit scale.
  If temperature has been applied, logits are already compressed or expanded — the
  frequency penalty's bite changes accordingly.

### Failure modes

| Symptom | Cause |
|---|---|
| `ValueError` at construction | `penalty < 0` |
| Topic words suppressed | Penalty too high; common domain words appear often |
| No visible effect | Penalty too low relative to logit magnitude |

---

## Processor interaction matrix

This table shows the primary interaction between pairs of processors when both are
active in the same pipeline. "Order sensitive" means the order they appear in
`LogitsProcessorList` changes the output distribution.

```
  ┌─────────────────────────────┬──────────────────────────────────────────────────────┐
  │  Pair                        │  Interaction                                         │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  Temperature + TopK          │  Order sensitive. T before TopK changes which tokens  │
  │                              │  survive. T after TopK scales only the kept set.      │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  Temperature + TopP          │  Order sensitive. T changes cumulative mass shape.    │
  │                              │  T before TopP is standard (calibrate first).         │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  RepetitionPenalty + TopK    │  Order sensitive (see architecture doc). Penalty      │
  │                              │  before TopK can exclude repeated tokens entirely.    │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  RepetitionPenalty + TopP    │  Less critical than TopK. Penalty before TopP shifts  │
  │                              │  cumulative mass so nucleus size changes.             │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  RepPenalty + FreqPenalty    │  Additive in effect. RepPenalty uses ratio; FreqPen   │
  │                              │  uses subtraction. Running both compounds suppression. │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  TopK + TopP                 │  TopK bounds maximum nucleus size. TopP adapts within  │
  │                              │  that bound. Order: TopK then TopP is conventional.   │
  ├─────────────────────────────┼──────────────────────────────────────────────────────┤
  │  Temperature + TypicalDecod. │  Significant. Temperature changes H(p) and deviation   │
  │                              │  scores. High T makes more tokens "typical".           │
  └─────────────────────────────┴──────────────────────────────────────────────────────┘
```
