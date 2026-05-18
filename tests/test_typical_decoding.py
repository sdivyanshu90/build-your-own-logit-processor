"""Tests for typical decoding."""

from __future__ import annotations

from typing import cast

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from logit_lens.processors import TypicalDecodingLogitsWarper
from logit_lens.utils import entropy, log_softmax_stable


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def test_typical_keeps_mass_fraction() -> None:
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    warped = TypicalDecodingLogitsWarper(mass=0.6)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    original_probs = torch.softmax(scores, dim=-1)
    kept_mass = original_probs[torch.isfinite(warped)].sum()

    assert (
        kept_mass >= 0.6
    ), "Typical decoding should preserve at least the requested probability mass."


def test_uniform_distribution_keeps_all() -> None:
    scores = torch.zeros((2, 6), dtype=torch.float32)
    warped = TypicalDecodingLogitsWarper(mass=0.5)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert torch.equal(
        warped, scores
    ), "Uniform logits should remain untouched because all tokens are equally typical."


def test_peaked_distribution_filters_aggressively() -> None:
    scores = torch.tensor([[12.0, -5.0, -6.0, -7.0, -8.0]], dtype=torch.float32)
    warped = TypicalDecodingLogitsWarper(mass=0.2)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    kept = torch.isfinite(warped).sum(dim=-1)

    assert (
        kept.item() <= 2
    ), "A sharply peaked distribution should keep only a small typical set."


def test_entropy_computation_stable() -> None:
    scores = torch.tensor([[0.0, -80.0, -90.0, -100.0]], dtype=torch.float32)
    log_probs = log_softmax_stable(scores)
    value = entropy(log_probs)

    assert torch.isfinite(
        value
    ).all(), "Entropy must stay finite for distributions with tiny probabilities."


@settings(max_examples=25)
@given(
    mass=st.floats(
        min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    values=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=12,
    ),
)
def test_typical_kept_set_is_non_empty(mass: float, values: list[float]) -> None:
    scores = torch.tensor(values, dtype=torch.float32).reshape(2, 6)
    warped = TypicalDecodingLogitsWarper(mass=mass)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )

    assert (
        torch.isfinite(warped).any(dim=-1).all()
    ), "Typical decoding must keep at least one token per row."
