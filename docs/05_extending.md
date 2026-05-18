# Extending logit_lens

The simplest extension is to subclass `LogitsProcessor`, implement `__call__`, and
provide a `to_config()` method for JSON serialization. This tutorial builds a
`BannedTokensLogitsProcessor` that hard-blocks a configurable list of token IDs.

---

## Extension lifecycle

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │   1. Subclass         2. Validate         3. Implement        4. Serialize  │
  │      LogitsProcessor     in __init__          __call__           to_config  │
  │          │                   │                   │                  │       │
  │          ▼                   ▼                   ▼                  ▼       │
  │   class MyProc(LP):  if param < 0:       output = scores    return {        │
  │       def __init__:     raise ValueError     .float()           "type": ... │
  │           ...            (...)               .clone()           "params":.. │
  │                                          ...                  }             │
  │                                          return output                      │
  │                                              .to(scores.dtype)              │
  └─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │   5. (Optional) Register for deserialization                                │
  │                                                                             │
  │   from logit_lens.serialization import PROCESSOR_REGISTRY                  │
  │   PROCESSOR_REGISTRY["MyProc"] = MyProc                                    │
  │                                                                             │
  │   Required only if you want save_pipeline / load_pipeline to round-trip    │
  │   your custom class from disk.                                              │
  └─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │   6. Use in a LogitsProcessorList                                           │
  │                                                                             │
  │   pipeline = LogitsProcessorList([                                          │
  │       MyProc(param=value),                                                  │
  │       TopPLogitsWarper(top_p=0.9),                                          │
  │   ])                                                                        │
  │   model.generate(..., logits_processor=pipeline)                            │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Implement the processor

```python
from __future__ import annotations  # postponed evaluation for annotations

from typing import Any

import torch

from logit_lens.base import LogitsProcessor


class BannedTokensLogitsProcessor(LogitsProcessor):
    """Hard-block a fixed list of token ids at every decoding step."""

    def __init__(self, banned_token_ids: list[int]) -> None:
        # Reject empty list — it would silently do nothing.
        if not banned_token_ids:
            raise ValueError("banned_token_ids must contain at least one token id.")
        # Token ids are non-negative vocabulary indices.
        if any(token_id < 0 for token_id in banned_token_ids):
            raise ValueError("banned_token_ids must all be >= 0.")
        # Deduplicate and sort for stable, predictable behavior.
        self.banned_token_ids = sorted(set(banned_token_ids))

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids                          # this processor ignores the prefix
        output = scores.float().clone()        # upcast to float32 for safe masking
        output[:, self.banned_token_ids] = float("-inf")  # ban across all batch rows
        return output.to(dtype=scores.dtype)   # cast back to caller's dtype

    def to_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "params": {"banned_token_ids": self.banned_token_ids},
        }
```

### Design checklist

```
  □  Upcast to float32 before arithmetic or masking.
  □  Clone scores before writing — never mutate the input tensor in place.
  □  Return output.to(dtype=scores.dtype) — preserve the caller's dtype.
  □  Validate all constructor arguments and raise ValueError on bad input.
  □  Implement to_config() if you want JSON serialization.
  □  Delete or ignore input_ids if your processor is context-free.
  □  Ensure at least one token per batch row remains finite after masking.
  □  Preserve (batch_size, vocab_size) shape — do not reorder vocab positions.
```

---

## Step 2 — Register for serialization

Registration is only needed for `save_pipeline` / `load_pipeline` round-trips.
In-memory use in a `LogitsProcessorList` works without registration.

```python
from logit_lens.serialization import PROCESSOR_REGISTRY

PROCESSOR_REGISTRY["BannedTokensLogitsProcessor"] = BannedTokensLogitsProcessor
```

The registry is a plain dict mapping string type names → Python classes. When
`load_pipeline()` reads a JSON config, it looks up each `"type"` key in this dict and
calls the class with the stored `"params"`.

```
  save_pipeline writes:                load_pipeline reads:
  ──────────────────────────────────   ────────────────────────────────────────────
  {                                    "BannedTokensLogitsProcessor"
    "type": "BannedTokensLogitsProc",    → PROCESSOR_REGISTRY["BannedTokens..."]
    "params": {                           → BannedTokensLogitsProcessor(
      "banned_token_ids": [198, 50256]       banned_token_ids=[198, 50256]
    }                                    )
  }
```

---

## Step 3 — Compose into a pipeline

```python
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import TopPLogitsWarper

pipeline = LogitsProcessorList([
    BannedTokensLogitsProcessor([198, 50256]),  # ban newline (198) and EOS (50256)
    TopPLogitsWarper(top_p=0.9),
])
```

Ordering still matters. Banning tokens before top-p means the nucleus is computed
over the already-filtered vocabulary — the banned tokens are fully excluded from
cumulative mass. Banning after top-p means the nucleus may have included those tokens
before they were masked out, which changes which remaining tokens survive.

For hard constraints (banned tokens, forced tokens) put them **before** truncation
warpers.

---

## Step 4 — Save and load the configuration

```python
from logit_lens.serialization import save_pipeline, load_pipeline

save_pipeline(pipeline, "pipeline.json")
```

The resulting JSON is explicit and human-readable:

```json
{
  "processors": [
    {
      "type": "BannedTokensLogitsProcessor",
      "params": {
        "banned_token_ids": [198, 50256]
      }
    },
    {
      "type": "TopPLogitsWarper",
      "params": {
        "top_p": 0.9,
        "filter_value": "-inf",
        "min_tokens_to_keep": 1
      }
    }
  ]
}
```

Restore the pipeline on any machine (as long as the class is imported and registered):

```python
restored = load_pipeline("pipeline.json")
```

JSON serialization round-trip:

```
  Python objects          JSON file               Python objects (restored)
  ──────────────────────  ──────────────────────  ──────────────────────────
  BannedTokensProcessor   {"type":"Banned...",    BannedTokensProcessor
  banned=[198,50256]       "params":{...}}        banned=[198,50256]
         │                       │                       │
         └──► to_config() ──────►│                       │
                                 │──► from_config() ─────┘
```

---

## Building a stateful processor

Use `StatefulLogitsProcessor` when you need to accumulate information across decoding
steps that cannot be trivially recovered from `input_ids` alone.

```python
from __future__ import annotations
from typing import Any

import torch

from logit_lens.base import StatefulLogitsProcessor


class BudgetedRepetitionProcessor(StatefulLogitsProcessor):
    """Ban any token that has been sampled more than `budget` times."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        if budget < 1:
            raise ValueError("budget must be >= 1.")
        self.budget = budget

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        # Initialize count tensor on first call.
        if "counts" not in self._state:
            self._state["counts"] = torch.zeros(
                batch_size, vocab_size, dtype=torch.long, device=scores.device
            )

        counts = self._state["counts"]
        output = scores.float().clone()

        # Mask any token whose count has reached the budget.
        over_budget = counts >= self.budget
        output[over_budget] = float("-inf")

        # Update counts from the last generated token (last column of input_ids).
        if input_ids.shape[-1] > 0:
            last_tokens = input_ids[:, -1]           # shape: (batch_size,)
            for b in range(batch_size):
                counts[b, last_tokens[b]] += 1

        return output.to(dtype=scores.dtype)

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {"budget": self.budget}}
```

**Key pattern:** call `reset_state()` between independent generation runs so that
count tensors from one generation do not bleed into the next:

```python
processor = BudgetedRepetitionProcessor(budget=3)

# First generation
output1 = model.generate(..., logits_processor=LogitsProcessorList([processor]))

# Must reset before the second independent generation
processor.reset_state()

# Second generation (fresh counts)
output2 = model.generate(..., logits_processor=LogitsProcessorList([processor]))
```

---

## Common extension patterns

```
  ┌─────────────────────┬──────────────────────────────────────────────────────┐
  │  Pattern            │  Implementation notes                                │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Hard ban           │  Set banned positions to -inf. Context-free: del     │
  │                     │  input_ids in __call__. Register to serialize.        │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Additive bias      │  scores + bias_tensor. Useful for domain steering.   │
  │                     │  Bias can be a fixed vector loaded from a file.       │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Grammar / FSM      │  Read input_ids to track parser state. Ban tokens    │
  │                     │  that would produce an invalid continuation.          │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Forced prefix      │  Force specific tokens in early steps by setting      │
  │                     │  all-but-one positions to -inf based on step count.   │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Watermarking       │  Add a small bias to a pseudo-random subset of        │
  │                     │  tokens keyed on the generation context.              │
  ├─────────────────────┼──────────────────────────────────────────────────────┤
  │  Stateful n-gram    │  Use StatefulLogitsProcessor to cache n-gram          │
  │  cache              │  prefixes so repeated prefix lookups are O(1).        │
  └─────────────────────┴──────────────────────────────────────────────────────┘
```
