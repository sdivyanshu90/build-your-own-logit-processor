"""Demonstrate end-to-end use of logit_lens with GPT-2."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from logit_lens.pipeline import LogitsProcessorList, build_standard_pipeline
from logit_lens.processors import (
    FrequencyPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TypicalDecodingLogitsWarper,
)
from logit_lens.serialization import load_pipeline, save_pipeline


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    processors: LogitsProcessorList,
    *,
    seed: int,
    max_new_tokens: int = 20,
) -> tuple[torch.LongTensor, str]:
    """Generate text with a deterministic seed and a custom processor pipeline.

    Args:
        model: Loaded causal language model.
        tokenizer: Matching tokenizer.
        prompt: Prompt text.
        processors: Custom logit processor chain.
        seed: Random seed for sampling.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        A tuple of generated token ids and decoded text.
    """

    encoded = tokenizer(prompt, return_tensors="pt")
    set_seed(seed)
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_ids, text


def main() -> None:
    """Run three GPT-2 decoding examples and verify serialization."""

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "In a quiet laboratory, the decoding algorithm"
    creative_pipeline = build_standard_pipeline(
        temperature=1.2,
        top_p=0.92,
        repetition_penalty=1.3,
    )
    constrained_pipeline = build_standard_pipeline(temperature=0.3, top_k=20)
    typical_pipeline = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=0.95),
            TypicalDecodingLogitsWarper(mass=0.9),
            FrequencyPenaltyLogitsProcessor(penalty=0.15),
        ]
    )

    configs: list[tuple[str, LogitsProcessorList, int]] = [
        ("creative", creative_pipeline, 11),
        ("constrained", constrained_pipeline, 13),
        ("typical", typical_pipeline, 17),
    ]

    for name, pipeline, seed in configs:
        _, text = generate_text(model, tokenizer, prompt, pipeline, seed=seed)
        print(f"[{name}] {pipeline}")
        print(text)
        print()

    with TemporaryDirectory() as directory:
        path = Path(directory) / "creative_pipeline.json"
        save_pipeline(creative_pipeline, path)
        restored = load_pipeline(path)
        first_ids, _ = generate_text(
            model, tokenizer, prompt, creative_pipeline, seed=23
        )
        second_ids, _ = generate_text(model, tokenizer, prompt, restored, seed=23)
        if not torch.equal(first_ids, second_ids):
            raise AssertionError(
                "Serialized pipeline did not reproduce the same generation output."
            )

    print("Serialization round-trip check passed.")


if __name__ == "__main__":
    main()
