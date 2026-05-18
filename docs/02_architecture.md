# Architecture

`logit_lens` is built around a single minimal protocol: every processor implements
`__call__(input_ids, scores) -> scores`. That signature matches the decoding boundary
in transformer generation exactly, and it is the same protocol HuggingFace uses, so
processors drop directly into `generate()` without adapters.

---

## The processor protocol

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  def __call__(                                                               │
  │      self,                                                                   │
  │      input_ids: torch.LongTensor,    # shape: (batch_size, seq_len)         │
  │      scores:    torch.FloatTensor,   # shape: (batch_size, vocab_size)      │
  │  ) -> torch.FloatTensor:             # shape: (batch_size, vocab_size)      │
  │      ...                                                                     │
  └──────────────────────────────────────────────────────────────────────────────┘

  input_ids  — the already-generated token ids for each sequence in the batch.
               Any context-sensitive processor (repetition penalty, frequency
               penalty, no-repeat n-gram) depends on this.

  scores     — raw next-token logits from the LM head. NOT probabilities.
               A processor may rescale, bias, or mask these values, but must
               return a tensor with the same (batch_size, vocab_size) shape.

  Invariant: token index j in the output still means vocabulary token j.
             Processors must not reorder vocabulary positions.
```

---

## Class hierarchy

```
  LogitsProcessor  (ABC)
  │   The root abstraction. Defines the __call__ protocol and the
  │   to_config() serialization hook.
  │
  ├── LogitsWarper  (ABC)
  │   │   A semantic subtype for processors that reshape the *distribution*
  │   │   itself — by scaling, truncating, or masking support. Same call
  │   │   signature; the distinction is conceptual, not behavioral.
  │   │
  │   ├── TemperatureLogitsWarper    z' = z / T
  │   ├── TopKLogitsWarper           keep top-k, mask rest with -inf
  │   ├── TopPLogitsWarper           keep smallest nucleus with mass ≥ p
  │   └── TypicalDecodingLogitsWarper  keep tokens near entropy H(p)
  │
  ├── RepetitionPenaltyLogitsProcessor   z'_i = z_i / pen  (z_i > 0)
  │                                      z'_i = z_i * pen  (z_i < 0)
  ├── FrequencyPenaltyLogitsProcessor    z'_i = z_i - pen * count(i)
  │
  └── StatefulLogitsProcessor  (ABC, mixin)
          Carries per-batch tensors between calls within a single generation.
          Provides reset_state() so independent generations don't leak state.
          └── (user-defined stateful processors)

  LogitsProcessorList  (not a subclass of LogitsProcessor)
      An ordered chain that runs processors in sequence and owns serialization
      of the whole pipeline as a JSON document.
```

---

## How `LogitsProcessorList` chains processors

Each processor in the list receives the output of the previous one as its `scores`
input. `input_ids` is passed unchanged to every processor because each may need the
full generated prefix independently.

```
  initial state
  ─────────────
  input_ids : LongTensor  (batch=1, seq_len=12)
  scores₀   : FloatTensor (batch=1, vocab=50257)  ← raw model output

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  LogitsProcessorList                                                     │
  │                                                                          │
  │  scores₀  ──►  [TemperatureLogitsWarper(0.9)]  ──►  scores₁             │
  │                                                                          │
  │  scores₁  ──►  [RepetitionPenaltyProcessor(1.1)]  ──►  scores₂          │
  │                                                                          │
  │  scores₂  ──►  [TopPLogitsWarper(0.92)]  ──►  scores₃                   │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘

  scores₃ returned to generate() → softmax → sample → next token

  Key detail: input_ids flows into every box unchanged.
              scores flows forward, mutated at each step.
```

The list executes unconditionally in order. There is no short-circuit. A processor
that wants to be a no-op should return `scores` unchanged — the list will still call
the next processor.

---

## Ordering matters: a concrete example

Ordering is a real semantic choice. The same processors in different orders can produce
different output distributions.

### Scenario: "the" has appeared twice in the prefix; top-k=2

```
  Raw logits
  ───────────────────────────────
  "the"   ██████████████  4.20   (repeated token — high raw score)
  "fox"   ████████████    3.80
  "cat"   ████████        2.10
  "and"   ███             1.50
```

**Ordering A — RepetitionPenalty(1.4) before TopK(2)**

```
  After RepetitionPenalty(1.4):    After TopK(2):
  ─────────────────────────────    ─────────────────────────────
  "the"   █████████       3.00     "the"   ░░░░░░░░░░░░  -inf  ✗ excluded!
  "fox"   ████████████    3.80     "fox"   ████████████  3.80  ✓
  "cat"   ████████        2.10     "cat"   ░░░░░░░░░░░░  -inf  ✗
  "and"   ███             1.50     "and"   ░░░░░░░░░░░░  -inf  ✗
                                   ─────────────────────────────
                                   "the" cannot appear in output.
```

**Ordering B — TopK(2) before RepetitionPenalty(1.4)**

```
  After TopK(2):                   After RepetitionPenalty(1.4):
  ─────────────────────────────    ─────────────────────────────
  "the"   ██████████████  4.20  ✓  "the"   █████████      3.00  still alive
  "fox"   ████████████    3.80  ✓  "fox"   ████████████   3.80
  "cat"   ░░░░░░░░░░░░   -inf  ✗  "cat"   ░░░░░░░░░░░░  -inf
  "and"   ░░░░░░░░░░░░   -inf  ✗  "and"   ░░░░░░░░░░░░  -inf
                                   ─────────────────────────────
                                   "the" survives with reduced score.
```

**Summary:** penalty-before-truncation can push repeated tokens out of the kept set
entirely. Truncation-before-penalty lets them survive with a lower score. Neither is
universally correct — the right choice depends on how hard you want to suppress
repetition.

`logit_lens` makes ordering explicit. The standard pipeline uses:

```
  Temperature → Penalties → Truncation
```

---

## Stateless vs stateful processors

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Stateless processors                                                  │
  │                                                                        │
  │  • Pure functions of (input_ids, scores).                              │
  │  • Safe to reuse across requests without resetting.                    │
  │  • All six built-in processors are stateless.                          │
  │                                                                        │
  │  def __call__(self, input_ids, scores):                                │
  │      return scores / self.temperature   ← no self-mutation             │
  └────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │  Stateful processors  (StatefulLogitsProcessor mixin)                  │
  │                                                                        │
  │  • Accumulate tensors in self._state between calls within one         │
  │    generation (e.g., a running n-gram history or a ban list).          │
  │  • reset_state() must be called between independent generation calls   │
  │    to avoid leaking state from one request to the next.                │
  │                                                                        │
  │  def __call__(self, input_ids, scores):                                │
  │      self._state["seen"].update(input_ids[0].tolist())  ← mutates     │
  │      ...                                                               │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## Design decisions

### Why `__call__` instead of a named method?

Using `__call__` keeps processors directly passable to any callable with the same
signature. The object still carries configuration, validation, and optional state —
but execution is lightweight and HuggingFace-compatible without any adapter layer.

### Why `-inf` masking instead of zeroing?

```
  Intuition: "zero the logit of a token I don't want"
  ─────────────────────────────────────────────────────

  Token logits before masking: [3.0, 1.5, 0.1, 2.0]  ← "on" is the 4th token
  After zeroing index 3:        [3.0, 1.5, 0.1, 0.0]
  After softmax:                [0.55, 0.14, 0.03, 0.28]
      ─── "on" still has 28% probability! ───

  After setting index 3 to -inf: [3.0, 1.5, 0.1, -inf]
  After softmax:                  [0.78, 0.17, 0.04, 0.000]
      ─── "on" has exactly 0% probability ───
```

Zero is a valid logit. Setting a token to zero does not remove it from the
distribution — it just gives it a score of 0. Only `-inf` guarantees that
`exp(-inf) = 0`, which means zero probability after softmax.

### Why temperature before other processors?

Temperature is a global rescaling of the model's raw preferences. Applying it first
calibrates the score landscape before context-sensitive penalties and truncation act
on it. If temperature runs after top-k, it can no longer change which tokens survive
truncation. If it runs after repetition penalty, the penalty's effective magnitude
changes depending on temperature scale.

The standard pipeline applies temperature first so that every downstream processor
sees a consistently scaled distribution. This is a convention, not a law — you can
change it.

### Why `log_softmax` in typical sampling?

Typical decoding computes `−log p(x)` for each token (information content) and
`H(p) = −Σ p log p` (entropy of the distribution). Taking `log` of a softmax output
computed from probabilities risks underflow when probabilities are tiny. Running the
entire computation in log-space with a numerically stable `log_softmax` avoids this.

---

## Batch dimension

All processors operate on the batch dimension independently. A batch of size B means
B independent sequences, each with its own generated prefix and its own row in `scores`.
Processors that penalize repeated tokens (repetition penalty, frequency penalty) process
each row against its own `input_ids` row — repetition in sequence 0 does not affect
sequence 1's logits.

```
  scores  shape: (batch_size, vocab_size)
  ─────────────────────────────────────────────────────────
  row 0:  [ 2.4, 1.2, 0.1, -0.7, ... ]  ← sequence 0's logits
  row 1:  [ 0.8, 3.1, -0.5, 1.9, ... ]  ← sequence 1's logits (independent)
  row 2:  [ 1.1, 0.3, 2.7, -1.2, ... ]  ← sequence 2's logits (independent)
  ─────────────────────────────────────────────────────────
  Every processor must preserve this shape. Shape changes break all downstream
  processors and the sampler.
```
