"""Tests for numerical helper utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from logit_lens.utils import assert_valid_scores, entropy, log_softmax_stable, safe_log


def test_safe_log_no_inf_from_zero() -> None:
    value = safe_log(torch.tensor([0.0]))

    assert torch.isfinite(
        value
    ).all(), "safe_log should clamp zero before taking the logarithm."


def test_log_softmax_stable_matches_torch() -> None:
    scores = torch.tensor([[4.0, 1.0, -3.0]], dtype=torch.float32)
    actual = log_softmax_stable(scores)
    expected = F.log_softmax(scores, dim=-1)

    assert torch.allclose(
        actual, expected, atol=1e-6
    ), "Stable log-softmax should match PyTorch's implementation."


def test_assert_valid_scores_raises_on_nan() -> None:
    scores = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="NaN"):
        assert_valid_scores(scores, context="nan_test")

    assert True, "assert_valid_scores should reject NaN entries."


def test_assert_valid_scores_raises_on_all_inf_row() -> None:
    scores = torch.tensor([[float("-inf"), float("-inf")]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="all-inf"):
        assert_valid_scores(scores, context="inf_test")

    assert True, "assert_valid_scores should reject rows with no finite logits."


def test_entropy_zero_for_delta() -> None:
    log_probs = torch.log(torch.tensor([[1.0 - 1e-8, 1e-8]], dtype=torch.float32))
    value = entropy(log_probs)

    assert (
        value.item() < 1e-5
    ), "Entropy should be near zero for an almost one-hot distribution."


def test_entropy_maximum_for_uniform() -> None:
    vocab_size = 5
    probs = torch.full((1, vocab_size), 1.0 / vocab_size, dtype=torch.float32)
    value = entropy(torch.log(probs))
    expected = torch.tensor([torch.log(torch.tensor(float(vocab_size))).item()])

    assert torch.allclose(
        value, expected, atol=1e-6
    ), "Uniform distributions should achieve the maximum entropy log(vocab_size)."
