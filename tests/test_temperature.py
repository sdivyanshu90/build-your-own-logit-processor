"""Tests for temperature scaling."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from logit_lens.processors import TemperatureLogitsWarper
from logit_lens.utils import entropy, log_softmax_stable


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def _entropy_from_scores(scores: torch.Tensor) -> torch.Tensor:
    return entropy(log_softmax_stable(scores.float()))


def test_temperature_scaling_increases_entropy(random_logits: torch.Tensor) -> None:
    baseline = _entropy_from_scores(random_logits)
    warped = TemperatureLogitsWarper(temperature=2.0)(
        _as_long_tensor(torch.zeros((random_logits.shape[0], 0), dtype=torch.long)),
        _as_float_tensor(random_logits),
    )
    warped_entropy = _entropy_from_scores(warped)

    assert torch.all(
        warped_entropy >= baseline
    ), "Higher temperature should not decrease entropy for the tested logits."


def test_temperature_below_one_decreases_entropy(random_logits: torch.Tensor) -> None:
    baseline = _entropy_from_scores(random_logits)
    warped = TemperatureLogitsWarper(temperature=0.5)(
        _as_long_tensor(torch.zeros((random_logits.shape[0], 0), dtype=torch.long)),
        _as_float_tensor(random_logits),
    )
    warped_entropy = _entropy_from_scores(warped)

    assert torch.all(
        warped_entropy <= baseline
    ), "Temperature below one should not increase entropy for the tested logits."


def test_temperature_near_zero_approximates_argmax() -> None:
    scores = torch.tensor([[1.0, 4.0, 2.0, -3.0]], dtype=torch.float32)
    warped = TemperatureLogitsWarper(temperature=1e-8)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    probs = torch.softmax(warped.float(), dim=-1)

    assert torch.isclose(
        probs[0, 1], torch.tensor(1.0)
    ), "Near-zero temperature should collapse probability mass onto the argmax token."


def test_temperature_one_is_identity() -> None:
    scores = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
    warped = TemperatureLogitsWarper(temperature=1.0)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    assert torch.equal(warped, scores), "Temperature 1.0 should leave logits unchanged."


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        TemperatureLogitsWarper(temperature=0.0)

    assert True, "Invalid temperature values must raise at initialization time."


def test_fp16_safe() -> None:
    scores = torch.tensor([[0.2, -0.1, 1.7, -2.5]], dtype=torch.float16)
    warped = TemperatureLogitsWarper(temperature=0.8)(
        _as_long_tensor(torch.zeros((1, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    probs = torch.softmax(warped.float(), dim=-1)

    assert torch.isfinite(
        probs
    ).all(), "Temperature warping should remain finite for fp16 inputs."


@settings(max_examples=25)
@given(
    temperature=st.floats(
        min_value=0.01,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    values=st.lists(
        st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=10,
    ),
)
def test_temperature_softmax_remains_normalized(
    temperature: float, values: list[float]
) -> None:
    scores = torch.tensor(values, dtype=torch.float32).reshape(2, 5)
    warped = TemperatureLogitsWarper(temperature=temperature)(
        _as_long_tensor(torch.zeros((2, 0), dtype=torch.long)),
        _as_float_tensor(scores),
    )
    probs = torch.softmax(warped.float(), dim=-1)

    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(2), atol=1e-6
    ), "Softmax of temperature-warped logits must sum to one within tolerance."
