"""Tests for the base processor abstractions."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from logit_lens.base import LogitsProcessor, StatefulLogitsProcessor


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


class IdentityProcessor(LogitsProcessor):
    """Minimal concrete processor used for testing."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        return scores

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {}}


class CountingStatefulProcessor(StatefulLogitsProcessor):
    """Simple stateful processor used for exercising reset behavior."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        previous = self._state.get("calls")
        if previous is None:
            self._state["calls"] = torch.ones(scores.shape[0], dtype=torch.long)
        else:
            self._state["calls"] = previous + 1
        return scores

    def to_config(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "params": {}}


def test_logits_processor_is_abstract() -> None:
    with pytest.raises(TypeError):
        cast(Any, LogitsProcessor)()


def test_minimal_concrete_subclass_is_usable() -> None:
    processor: LogitsProcessor = IdentityProcessor()
    input_ids = _as_long_tensor(torch.zeros((2, 3), dtype=torch.long))
    scores = _as_float_tensor(torch.randn(2, 5))
    output = processor(input_ids, scores)
    assert torch.equal(
        output, scores
    ), "IdentityProcessor should pass scores through unchanged."


def test_stateful_processor_reset_state_clears_tensors() -> None:
    processor = CountingStatefulProcessor(
        state={"cache": torch.ones(2, 3, dtype=torch.float32)}
    )
    input_ids = _as_long_tensor(torch.zeros((2, 1), dtype=torch.long))
    scores = _as_float_tensor(torch.randn(2, 4))

    processor(input_ids, scores)
    assert processor.state, "State should be populated after calling the processor."

    processor.reset_state()
    assert not processor.state, "reset_state() should clear all tracked state tensors."


@settings(max_examples=30)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=0, max_value=5),
    vocab=st.integers(min_value=1, max_value=8),
    use_half=st.booleans(),
)
def test_identity_subclass_preserves_shape_and_dtype(
    batch: int, seq: int, vocab: int, use_half: bool
) -> None:
    processor: LogitsProcessor = IdentityProcessor()
    dtype = torch.float16 if use_half else torch.float32
    input_ids = _as_long_tensor(torch.zeros((batch, seq), dtype=torch.long))
    scores = _as_float_tensor(torch.randn(batch, vocab, dtype=dtype))
    output = processor(input_ids, scores)

    assert (
        output.shape == scores.shape
    ), "Identity processor must preserve the score tensor shape."
    assert (
        output.dtype == scores.dtype
    ), "Identity processor must preserve the input dtype."
