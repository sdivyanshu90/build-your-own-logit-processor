# Usage Guide

`logit_lens` is designed to plug directly into HuggingFace `generate()`. Processors
implement the same callable protocol, so you pass a `LogitsProcessorList` and
`generate()` calls it at each decoding step without any adapter.

---

## Choosing a processor recipe

Use this decision tree to pick a starting configuration:

```
  What is your generation goal?
  │
  ├─► Exact reproducibility (same output every run)
  │       └─► TemperatureLogitsWarper(1e-8)   [greedy approximation]
  │
  ├─► Factual question answering or code completion
  │       └─► Low temperature (0.3–0.5) + TopK(20)
  │           build_standard_pipeline(temperature=0.4, top_k=20)
  │
  ├─► General balanced generation (default)
  │       └─► Temperature(0.9) + TopK(40) + TopP(0.92) + RepPenalty(1.1)
  │           build_standard_pipeline()
  │
  ├─► Creative writing / storytelling
  │       └─► High temperature (1.1–1.3) + TopP(0.92) + RepPenalty(1.3)
  │           build_standard_pipeline(temperature=1.2, top_p=0.92,
  │                                   repetition_penalty=1.3)
  │
  ├─► Diverse output without incoherence
  │       └─► Temperature(0.95) + TypicalDecoding(0.9)
  │           (manual composition — see recipe 4 below)
  │
  └─► Strong repetition avoidance
          └─► RepPenalty(1.3–1.5) + FrequencyPenalty(0.3) + TopP(0.92)
              (manual composition — see recipe 5 below)
```

---

## Pipeline assembly at a glance

```
  Build once before generation:

  processors = LogitsProcessorList([
      TemperatureLogitsWarper(temperature=0.9),      ← step 1: global rescale
      RepetitionPenaltyLogitsProcessor(penalty=1.1), ← step 2: context-sensitive
      TopPLogitsWarper(top_p=0.92),                  ← step 3: truncation
  ])

  Pass to generate():

  model.generate(
      **inputs,
      max_new_tokens=32,
      do_sample=True,          ← required; greedy ignores logits_processor
      logits_processor=processors,
  )

  At each decoding step the list is called automatically:

  step t=0:  processors(input_ids=[101, 234], scores=[...])
  step t=1:  processors(input_ids=[101, 234, 819], scores=[...])
  step t=2:  processors(input_ids=[101, 234, 819, 42], scores=[...])
  ...
  (input_ids grows by one token per step; scores is always vocab_size wide)
```

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import build_standard_pipeline

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

inputs     = tokenizer("A logit processor library can", return_tensors="pt")
processors = build_standard_pipeline(
    temperature=0.9,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.1,
)
output_ids = model.generate(
    **inputs,
    max_new_tokens=24,
    do_sample=True,
    logits_processor=processors,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

`build_standard_pipeline` wires processors in the recommended order:
Temperature → RepetitionPenalty → TopK → TopP → TypicalDecoding (when configured).
Parameters left at their neutral values (`temperature=1.0`, `top_k=None`, etc.)
produce no processor for that slot.

---

## Cookbook

### Recipe 1 — Greedy decoding (deterministic)

Temperature zero is singular (division by zero). Use `1e-8` as an approximation:
the library masks all non-maximum logits, which is effectively argmax.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import TemperatureLogitsWarper

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")
inputs    = tokenizer("The formal proof begins with", return_tensors="pt")

processors = LogitsProcessorList([
    TemperatureLogitsWarper(temperature=1e-8),
])
output_ids = model.generate(
    **inputs, max_new_tokens=20, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# Effect on distribution:
# before:  [3.0, 1.5, 0.1, -0.7, ...]
# after:   [3.0, -inf, -inf, -inf, ...]   ← only argmax survives
```

---

### Recipe 2 — Creative writing

High temperature flattens the distribution; top-p cuts the low-probability tail;
repetition penalty keeps the text from looping.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import build_standard_pipeline

tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
inputs     = tokenizer("On the edge of the city there was", return_tensors="pt")
processors = build_standard_pipeline(
    temperature=1.2,          # flatten — more varied word choice
    top_p=0.92,               # cut extreme tail
    repetition_penalty=1.3,   # discourage loops
)
output_ids = model.generate(
    **inputs, max_new_tokens=40, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# Distribution shape:
# raw:        [0.67, 0.20, 0.07, 0.03, 0.02, 0.01]   (T=1.0)
# after T=1.2:[0.56, 0.19, 0.09, 0.06, 0.05, 0.05]   (flatter, more variety)
# after TopP: nucleus covers top ~92% → low-prob tokens masked
```

---

### Recipe 3 — Factual or constrained generation

Lower temperature + tight top-k keeps the model on confident, high-probability
continuations. Good for question answering, code, or structured output.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import build_standard_pipeline

tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
inputs     = tokenizer("The capital of France is", return_tensors="pt")
processors = build_standard_pipeline(
    temperature=0.3,   # sharpen — fewer, more predictable choices
    top_k=20,          # hard limit to top 20 tokens
)
output_ids = model.generate(
    **inputs, max_new_tokens=12, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# Distribution shape:
# raw:         [0.67, 0.20, 0.07, 0.03, 0.02, ...]  (T=1.0)
# after T=0.3: [0.97, 0.02, 0.01, 0.00, 0.00, ...]  (very sharp)
# after TopK:  top 20 tokens kept, rest masked
```

---

### Recipe 4 — Diverse output without incoherence (typical decoding)

Typical decoding removes both the most-predictable tokens and the most-surprising
ones, targeting continuations that feel locally natural.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import TemperatureLogitsWarper, TypicalDecodingLogitsWarper

tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
inputs     = tokenizer("A rigorous explanation should", return_tensors="pt")
processors = LogitsProcessorList([
    TemperatureLogitsWarper(temperature=0.95),
    TypicalDecodingLogitsWarper(mass=0.9),
])
output_ids = model.generate(
    **inputs, max_new_tokens=30, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# What typical decoding does differently from top-p:
# top-p keeps highest-probability tokens
# typical keeps tokens closest to the distribution entropy H(p)
# → removes ultra-predictable tokens AND ultra-surprising tokens
```

---

### Recipe 5 — Aggressive repetition control

Combine repetition penalty (binary: seen or not) with frequency penalty (counts
matter) for the strongest repetition suppression.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import (
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    FrequencyPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
inputs     = tokenizer("The system design review concluded that", return_tensors="pt")
processors = LogitsProcessorList([
    TemperatureLogitsWarper(temperature=0.9),
    RepetitionPenaltyLogitsProcessor(penalty=1.3),   # any repeat: ÷1.3
    FrequencyPenaltyLogitsProcessor(penalty=0.3),    # each additional: -0.3
    TopPLogitsWarper(top_p=0.92),
])
output_ids = model.generate(
    **inputs, max_new_tokens=40, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# A token seen 4 times gets:
#   RepetitionPenalty: logit / 1.3  (same regardless of count)
#   FrequencyPenalty:  logit - 0.3 * 4 = logit - 1.2
# Combined effect is substantially stronger than either alone.
```

---

### Recipe 6 — Custom no-repeat n-gram processor

This builds a fully custom processor from `LogitsProcessor`. It blocks any token
that would complete an n-gram already seen in the current sequence.

```python
from __future__ import annotations
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_lens.base import LogitsProcessor
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import TopPLogitsWarper


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int) -> None:
        if ngram_size < 2:
            raise ValueError("ngram_size must be >= 2.")
        self.ngram_size = ngram_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Not enough context yet to form a complete n-gram.
        if input_ids.shape[-1] + 1 < self.ngram_size:
            return scores

        output      = scores.float().clone()
        prefix_size = self.ngram_size - 1

        for batch_index, sequence in enumerate(input_ids.tolist()):
            # Build a map from each (n-1)-gram prefix → set of next tokens seen.
            seen: dict[tuple[int, ...], set[int]] = defaultdict(set)
            for start in range(len(sequence) - self.ngram_size + 1):
                ngram = sequence[start : start + self.ngram_size]
                seen[tuple(ngram[:-1])].add(ngram[-1])

            # Ban any token that would repeat a seen n-gram from the current prefix.
            current_prefix = tuple(sequence[-prefix_size:])
            banned = seen.get(current_prefix, set())
            if banned:
                output[batch_index, list(banned)] = float("-inf")

        return output.to(dtype=scores.dtype)

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {"ngram_size": self.ngram_size}}


tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
inputs     = tokenizer("The system design review concluded that", return_tensors="pt")
processors = LogitsProcessorList([
    NoRepeatNGramLogitsProcessor(ngram_size=3),
    TopPLogitsWarper(top_p=0.92),
])
output_ids = model.generate(
    **inputs, max_new_tokens=32, do_sample=True, logits_processor=processors
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Saving and loading a pipeline

Pipelines serialize to plain JSON. Every built-in processor implements `to_config()`.

```python
from logit_lens.pipeline import build_standard_pipeline
from logit_lens.serialization import save_pipeline, load_pipeline

pipeline = build_standard_pipeline(temperature=0.9, top_k=40, top_p=0.92,
                                   repetition_penalty=1.1)
save_pipeline(pipeline, "pipeline.json")

# pipeline.json looks like:
# {
#   "processors": [
#     {"type": "TemperatureLogitsWarper",          "params": {"temperature": 0.9}},
#     {"type": "RepetitionPenaltyLogitsProcessor", "params": {"penalty": 1.1}},
#     {"type": "TopKLogitsWarper",                 "params": {"top_k": 40, ...}},
#     {"type": "TopPLogitsWarper",                 "params": {"top_p": 0.92, ...}}
#   ]
# }

restored = load_pipeline("pipeline.json")
# restored is a new LogitsProcessorList with the same configuration
```

JSON configs are plain text: easy to version in git, diff across experiments, or
generate programmatically.

---

## Preset comparison

| Setting | Temperature | top_k | top_p | rep_penalty | Best for |
|---|---|---|---|---|---|
| Ultra-conservative | 0.2 | 10 | — | 1.0 | Structured / templated output |
| Factual | 0.4 | 20 | — | 1.0 | Q&A, summarization |
| Balanced | 0.9 | 40 | 0.92 | 1.1 | General chat / default |
| Creative | 1.2 | — | 0.92 | 1.3 | Stories, poems |
| Exploratory | 1.4 | — | 0.95 | 1.5 | Brainstorming |

---

## Troubleshooting

### NaN in outputs

```
  Symptom: output contains nan tokens or model raises NaN errors.
  Diagnosis: check scores before and after each processor.

  import torch
  for i, p in enumerate(processors):
      scores = p(input_ids, scores)
      if torch.isnan(scores).any():
          print(f"NaN introduced by processor {i}: {p}")
          break

  Common causes:
  • Custom processors that call exp() on large positive logits before masking.
  • float16 overflow in custom arithmetic (always upcast to float32 first).
  • Upstream model instability (check raw model output before processors).
```

### All-inf row (distribution collapse)

```
  Symptom: generate() raises an error about all -inf logits, or samples garbage.
  Cause: a processor masked every token in at least one batch row.

  Prevention:
  • Use min_tokens_to_keep >= 1 in TopK, TopP, and TypicalDecoding.
  • Custom processors must ensure at least one finite logit per row.

  Diagnosis:
  filtered = processor(input_ids, scores)
  all_inf  = (filtered == float("-inf")).all(dim=-1)   # True for problem rows
  print(f"All-inf rows: {all_inf.nonzero()}")
```

### Dtype mismatch

```
  Symptom: RuntimeError about dtype incompatibility downstream.
  Cause: custom processor returns a different dtype than it received.

  Correct pattern:
  def __call__(self, input_ids, scores):
      original_dtype = scores.dtype
      work = scores.float()          # upcast to float32 for safe math
      ...                            # do your processing
      return work.to(original_dtype) # cast back before returning
```

### `do_sample=False` ignores processors

```
  Symptom: processors are passed to generate() but have no visible effect.
  Cause: do_sample=False uses greedy / beam search, which bypasses the processor list.
  Fix: always set do_sample=True when using logit processors.
```

### Top-k rejects `top_k=0`

```
  Symptom: ValueError at construction.
  Cause: TopKLogitsWarper(top_k=0) is invalid. Zero tokens to keep is undefined.
  Fix: omit TopKLogitsWarper entirely from the list if you want no top-k filter.
```
