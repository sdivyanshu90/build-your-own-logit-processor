"""Tests for processor composition and serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import logit_lens.pipeline as pipeline_module
from logit_lens.base import LogitsProcessor
from logit_lens.pipeline import LogitsProcessorList, build_standard_pipeline
from logit_lens.processors import (
    FrequencyPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalDecodingLogitsWarper,
)
from logit_lens.serialization import load_pipeline, save_pipeline


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


class AdditiveProcessor(LogitsProcessor):
    """Processor used to validate list ordering."""

    def __init__(self, delta: float, record: list[float]) -> None:
        self.delta = delta
        self.record = record

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        self.record.append(self.delta)
        return cast(
            torch.FloatTensor, (scores.float() + self.delta).to(dtype=scores.dtype)
        )

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {"delta": self.delta}}


class MarkerProcessor(LogitsProcessor):
    """Processor placeholder used to inspect factory ordering."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        return scores

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {"name": self.name}}


def test_list_applies_in_order() -> None:
    record: list[float] = []
    pipeline = LogitsProcessorList(
        [AdditiveProcessor(1.0, record), AdditiveProcessor(2.0, record)]
    )
    scores = torch.zeros((1, 3), dtype=torch.float32)
    output = pipeline(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert record == [1.0, 2.0], "Processors should be invoked in insertion order."
    assert torch.equal(
        output, torch.full((1, 3), 3.0)
    ), "Ordered processors should accumulate sequentially."


def test_empty_list_is_identity() -> None:
    pipeline = LogitsProcessorList()
    scores = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    output = pipeline(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert torch.equal(
        output, scores
    ), "An empty processor list should be the identity map."


def test_serialization_roundtrip(tmp_path: Path) -> None:
    pipeline = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=0.8),
            TopKLogitsWarper(top_k=3),
            TopPLogitsWarper(top_p=0.9),
            TypicalDecodingLogitsWarper(mass=0.95),
            FrequencyPenaltyLogitsProcessor(penalty=0.2),
        ]
    )
    path = tmp_path / "pipeline.json"
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    input_ids = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)

    save_pipeline(pipeline, path)
    restored = load_pipeline(path)

    assert torch.equal(
        pipeline(_as_long_tensor(input_ids), _as_float_tensor(scores)),
        restored(_as_long_tensor(input_ids), _as_float_tensor(scores)),
    ), "Serialized and deserialized pipelines should produce identical outputs."


def test_from_config_creates_correct_types() -> None:
    config = {
        "processors": [
            {"type": "TemperatureLogitsWarper", "params": {"temperature": 0.7}},
            {
                "type": "TopKLogitsWarper",
                "params": {"top_k": 4, "filter_value": "-inf", "min_tokens_to_keep": 1},
            },
            {
                "type": "TopPLogitsWarper",
                "params": {
                    "top_p": 0.9,
                    "filter_value": "-inf",
                    "min_tokens_to_keep": 1,
                },
            },
            {
                "type": "TypicalDecodingLogitsWarper",
                "params": {
                    "mass": 0.95,
                    "filter_value": "-inf",
                    "min_tokens_to_keep": 1,
                },
            },
            {"type": "FrequencyPenaltyLogitsProcessor", "params": {"penalty": 0.3}},
        ]
    }
    pipeline = LogitsProcessorList.from_config(config)

    assert [type(processor).__name__ for processor in pipeline] == [
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalDecodingLogitsWarper",
        "FrequencyPenaltyLogitsProcessor",
    ], "from_config() should instantiate the requested processor classes in order."


def test_build_standard_pipeline_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    creation_order: list[str] = []

    def make_factory(name: str) -> Any:
        def factory(*args: Any, **kwargs: Any) -> MarkerProcessor:
            del args, kwargs
            creation_order.append(name)
            return MarkerProcessor(name)

        return factory

    monkeypatch.setattr(
        pipeline_module, "TemperatureLogitsWarper", make_factory("temperature")
    )
    monkeypatch.setattr(
        pipeline_module,
        "RepetitionPenaltyLogitsProcessor",
        make_factory("repetition"),
    )
    monkeypatch.setattr(pipeline_module, "TopKLogitsWarper", make_factory("top_k"))
    monkeypatch.setattr(pipeline_module, "TopPLogitsWarper", make_factory("top_p"))

    pipeline = build_standard_pipeline(
        temperature=0.7, top_k=5, top_p=0.9, repetition_penalty=1.1
    )

    assert creation_order == [
        "temperature",
        "repetition",
        "top_k",
        "top_p",
    ], "build_standard_pipeline() should instantiate processors in decoding order."
    assert [
        getattr(processor, "name") for processor in pipeline
    ] == creation_order, "The returned pipeline order should match construction order."


@pytest.mark.slow
@pytest.mark.parametrize(
    "processor_list",
    [
        build_standard_pipeline(temperature=0.7, top_k=20, repetition_penalty=1.1),
        build_standard_pipeline(temperature=1.1, top_p=0.9),
        LogitsProcessorList(
            [
                TypicalDecodingLogitsWarper(mass=0.9),
                FrequencyPenaltyLogitsProcessor(penalty=0.2),
            ]
        ),
    ],
)
def test_huggingface_generate_is_deterministic(
    processor_list: LogitsProcessorList,
) -> None:
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    prompt = "Logit processors help decoding by"
    encoded = tokenizer(prompt, return_tensors="pt")

    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": 8,
        "pad_token_id": tokenizer.eos_token_id,
        "logits_processor": processor_list,
    }

    set_seed(1234)
    first = model.generate(**encoded, **generation_kwargs)
    set_seed(1234)
    second = model.generate(**encoded, **generation_kwargs)

    assert torch.equal(
        first, second
    ), "Generation with a fixed seed and identical processors should be deterministic."
