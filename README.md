# logit_lens

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](#)
[![Coverage](https://img.shields.io/badge/Coverage-pending-lightgrey)](#)
[![PyPI](https://img.shields.io/badge/PyPI-pending-lightgrey)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)

`logit_lens` is a production-ready Python library for composable logit processing
during autoregressive text generation. It provides numerically stable decoding
primitives, HuggingFace-compatible processor composition, JSON serialization, and
a complete pytest suite.

---

## The one-sentence idea

Between the transformer and the sampler there is a gap. `logit_lens` fills it:

```
  model ──► raw logits ──► LogitsProcessorList ──► softmax ──► sampler ──► token
                           └──────────────────┘
                                your code lives here
```

---

## How one token gets chosen

Every token your language model emits passes through five steps. Here is the full
picture for a toy five-token vocabulary:

```
  ┌─ Step 1: model forward pass ────────────────────────────────────────────────┐
  │                                                                             │
  │  One real-valued score (logit) is produced per vocabulary token.            │
  │                                                                             │
  │  "The"   ████████████████████  2.40                                         │
  │  "cat"   █████████████         1.20                                         │
  │  "sat"   ████                  0.10                                         │
  │  "on"    ██                   -0.70                                         │
  │  "."     █                    -1.00                                         │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─ Step 2: TemperatureLogitsWarper(temperature=0.8) ──────────────────────────┐
  │                                                                             │
  │  Divide every logit by 0.8. Gaps widen — top token becomes relatively       │
  │  more dominant.                                                             │
  │                                                                             │
  │  "The"   ██████████████████████████  3.00  (+0.60)                          │
  │  "cat"   █████████████████           1.50  (+0.30)                          │
  │  "sat"   ████                        0.13  (+0.03)                          │
  │  "on"    ██                         -0.88  (-0.18)                          │
  │  "."     █                          -1.25  (-0.25)                          │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─ Step 3: TopKLogitsWarper(top_k=3) ─────────────────────────────────────────┐
  │                                                                             │
  │  Keep only the 3 highest-scoring tokens. Set the rest to -∞.               │
  │                                                                             │
  │  "The"   ██████████████████████████  3.00   ✓  kept                        │
  │  "cat"   █████████████████           1.50   ✓  kept                        │
  │  "sat"   ████                        0.13   ✓  kept                        │
  │  "on"    ░░░░░░░░░░░░░░░░░░░░░░░░░░  -inf   ✗  masked                      │
  │  "."     ░░░░░░░░░░░░░░░░░░░░░░░░░░  -inf   ✗  masked                      │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─ Step 4: softmax ───────────────────────────────────────────────────────────┐
  │                                                                             │
  │  exp(3.00)=20.09  exp(1.50)=4.48  exp(0.13)=1.14  sum=25.71               │
  │                                                                             │
  │  "The"   ██████████████████████████████  0.782                              │
  │  "cat"   ██████                          0.174                              │
  │  "sat"   ██                              0.044                              │
  │  "on"    (0.000)                                                            │
  │  "."     (0.000)                                                            │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─ Step 5: multinomial sample ────────────────────────────────────────────────┐
  │                                                                             │
  │  Draw one token at random from the distribution.                            │
  │  → "The"  (sampled — most probable but not certain)                        │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick install

```bash
python -m pip install -e ".[dev]"
```

## 5-line quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logit_lens.pipeline import build_standard_pipeline

tokenizer  = AutoTokenizer.from_pretrained("gpt2")
model      = AutoModelForCausalLM.from_pretrained("gpt2")
processors = build_standard_pipeline(temperature=0.9, top_k=40, top_p=0.92,
                                     repetition_penalty=1.1)
output_ids = model.generate(
    **tokenizer("Logits matter because", return_tensors="pt"),
    max_new_tokens=24, do_sample=True, logits_processor=processors,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Choose a recipe

| Goal | Temperature | Warper | Penalty | Code |
|---|---|---|---|---|
| Deterministic (greedy) | ~0 (`1e-8`) | — | — | `LogitsProcessorList([TemperatureLogitsWarper(1e-8)])` |
| Factual / focused | 0.3–0.5 | top-k 20 | — | `build_standard_pipeline(temperature=0.4, top_k=20)` |
| Balanced default | 0.9 | top-k 40, top-p 0.92 | rep 1.1 | `build_standard_pipeline()` |
| Creative writing | 1.1–1.3 | top-p 0.92 | rep 1.3 | `build_standard_pipeline(temperature=1.2, top_p=0.92, repetition_penalty=1.3)` |
| Diverse & coherent | 0.95 | typical 0.9 | — | manual (see usage guide) |

---

## Processor overview

| Class | Kind | Uses prefix? | Effect |
|---|---|---|---|
| `TemperatureLogitsWarper` | Warper | No | Multiplies logit gaps — sharpens or flattens |
| `TopKLogitsWarper` | Warper | No | Hard-masks every token outside the top k |
| `TopPLogitsWarper` | Warper | No | Masks tail until nucleus mass ≥ top_p |
| `RepetitionPenaltyLogitsProcessor` | Processor | Yes | Ratio-penalizes logits of already-seen tokens |
| `TypicalDecodingLogitsWarper` | Warper | No | Masks tokens far from distribution entropy |
| `FrequencyPenaltyLogitsProcessor` | Processor | Yes | Subtracts penalty × occurrence count |

---

## Standard pipeline ordering

```
  Temperature  ──►  RepetitionPenalty / FrequencyPenalty  ──►  TopK / TopP / Typical
  (global        (context-sensitive downweighting)             (hard truncation)
   rescale)
```

Rescale first so penalties act on a calibrated scale. Truncate last so masking sees
final adjusted scores. Ordering is explicit and composable — you can change it.

---

## Documentation

- [What Are Logits?](docs/01_what_are_logits.md) — raw scores, softmax, and why log-space is the right home for decoding
- [Architecture](docs/02_architecture.md) — class hierarchy, `LogitsProcessorList`, and why ordering matters
- [Processor Reference](docs/03_processor_reference.md) — math, parameters, interactions, and step-by-step visualizations for all six processors
- [Usage Guide](docs/04_usage_guide.md) — annotated recipes from greedy to creative, plus a troubleshooting section
- [Extending](docs/05_extending.md) — building, registering, and serializing a custom processor end to end

---

## Project layout

```
logit_lens/
├── base.py          ← LogitsProcessor (ABC), LogitsWarper (ABC), StatefulLogitsProcessor
├── processors.py    ← six concrete processors
├── pipeline.py      ← LogitsProcessorList, build_standard_pipeline
├── serialization.py ← PROCESSOR_REGISTRY, save_pipeline, load_pipeline
└── utils.py         ← log_softmax_stable, entropy, assert_valid_scores

tests/               ← unit + integration test suite
docs/                ← five prose documents
scripts/demo.py      ← executable GPT-2 end-to-end demo
```

---

## Highlights

- Six built-in processors: temperature, top-k, top-p, repetition penalty, typical decoding, frequency penalty.
- Direct HuggingFace `generate()` compatibility through `LogitsProcessorList`.
- Numerically safe: sensitive math runs in `float32`; output is cast back to the original dtype.
- JSON round-trip: `save_pipeline()` / `load_pipeline()` serialize and restore full pipelines.
- Strict typing, black formatting, and unit plus integration coverage.
