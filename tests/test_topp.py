"""Tests for nucleus sampling."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from logit_lens.processors import TopPLogitsWarper


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def test_topp_cumulative_probability_threshold() -> None:
    scores = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
    warped = TopPLogitsWarper(top_p=0.8)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    original_probs = torch.softmax(scores, dim=-1)
    kept_mass = original_probs[torch.isfinite(warped)].sum()

    assert (
        kept_mass >= 0.8
    ), "Top-p should keep tokens whose original probability mass reaches the threshold."


def test_topp_one_keeps_all() -> None:
    scores = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float32)
    warped = TopPLogitsWarper(top_p=1.0)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert torch.equal(
        warped, scores
    ), "top_p=1.0 should preserve the full support of the distribution."


def test_topp_near_zero_keeps_one() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    warped = TopPLogitsWarper(top_p=1e-6)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    kept = torch.isfinite(warped).sum(dim=-1)

    assert torch.equal(
        kept, torch.tensor([1])
    ), "Very small top_p should keep only the most likely token."


def test_topp_invalid_raises() -> None:
    with pytest.raises(ValueError, match="top_p must be in"):
        TopPLogitsWarper(top_p=0.0)

    assert True, "Invalid top_p values must raise during construction."


def test_topp_min_tokens_respected() -> None:
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    warped = TopPLogitsWarper(top_p=0.01, min_tokens_to_keep=3)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    kept = torch.isfinite(warped).sum(dim=-1)

    assert torch.equal(
        kept, torch.tensor([3])
    ), "Top-p must keep at least min_tokens_to_keep tokens."


@settings(max_examples=25)
@given(
    top_p=st.floats(
        min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
    )
)
def test_topp_uniform_logits_remain_valid_distribution(top_p: float) -> None:
    scores = torch.zeros((2, 8), dtype=torch.float32)
    warped = TopPLogitsWarper(top_p=top_p)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    probs = torch.softmax(warped.float(), dim=-1)

    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(2), atol=1e-6
    ), "Top-p over uniform logits should still yield a normalized distribution after softmax."
