"""Tests for top-k filtering."""

from __future__ import annotations

from typing import cast

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from logit_lens.processors import TopKLogitsWarper


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def test_topk_zeroes_correct_number() -> None:
    scores = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0, 5.0], [1.0, 3.0, 5.0, 7.0, 9.0]], dtype=torch.float32
    )
    warped = TopKLogitsWarper(top_k=2)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    removed = (torch.isinf(warped) & (warped < 0)).sum(dim=-1)

    assert torch.equal(
        removed, torch.tensor([3, 3])
    ), "Top-k should mask exactly vocab_size - k tokens in each row."


def test_topk_preserves_top_values() -> None:
    scores = torch.tensor([[0.2, 0.9, -0.5, 1.2, 0.1]], dtype=torch.float32)
    warped = TopKLogitsWarper(top_k=2)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    top_indices = torch.topk(scores, k=2, dim=-1).indices

    assert torch.equal(
        warped[0, top_indices[0]], scores[0, top_indices[0]]
    ), "Top-k must preserve the original values of retained logits."


def test_topk_larger_than_vocab_clamps() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    warped = TopKLogitsWarper(top_k=10)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert torch.equal(
        warped, scores
    ), "top_k larger than vocab size should act like identity."


def test_min_tokens_to_keep() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    warped = TopKLogitsWarper(top_k=1, min_tokens_to_keep=3)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    kept = torch.isfinite(warped).sum(dim=-1)

    assert torch.equal(
        kept, torch.tensor([3])
    ), "min_tokens_to_keep must override an overly aggressive top_k choice."


def test_batch_independence() -> None:
    scores = torch.tensor(
        [[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 8.0, 9.0]], dtype=torch.float32
    )
    warped = TopKLogitsWarper(top_k=2)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert torch.isfinite(
        warped[0, :2]
    ).all(), "The first batch item should keep its own top-2 tokens."
    assert torch.isfinite(
        warped[1, 2:]
    ).all(), "The second batch item should keep its own top-2 tokens."


@settings(max_examples=30)
@given(
    top_k=st.integers(min_value=1, max_value=8),
    values=st.lists(
        st.floats(min_value=-6.0, max_value=6.0, allow_nan=False, allow_infinity=False),
        min_size=24,
        max_size=24,
    ),
)
def test_topk_never_masks_entire_row(top_k: int, values: list[float]) -> None:
    scores = torch.tensor(values, dtype=torch.float32).reshape(3, 8)
    warped = TopKLogitsWarper(top_k=top_k)(
        _as_long_tensor(torch.zeros((3, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert (
        not torch.isinf(warped).all(dim=-1).any()
    ), "Top-k must leave at least one valid token per row."
