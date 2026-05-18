"""Tests for frequency penalties."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from logit_lens.processors import FrequencyPenaltyLogitsProcessor


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def test_penalty_scales_with_count() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 0, 0, 2]], dtype=torch.long))
    warped = FrequencyPenaltyLogitsProcessor(penalty=0.5)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.isclose(
        warped[0, 0], torch.tensor(1.5)
    ), "Frequency penalty should subtract penalty multiplied by the token count."


def test_unseen_tokens_unaffected() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 0, 0, 2]], dtype=torch.long))
    warped = FrequencyPenaltyLogitsProcessor(penalty=0.5)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.isclose(
        warped[0, 1], scores[0, 1]
    ), "Tokens that never appeared in the prefix should not be changed."


def test_zero_penalty_is_identity() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 1, 2]], dtype=torch.long))
    warped = FrequencyPenaltyLogitsProcessor(penalty=0.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.equal(
        warped, scores
    ), "Penalty 0.0 should act as the identity processor."


def test_negative_penalty_raises() -> None:
    with pytest.raises(ValueError, match="penalty must be >= 0.0"):
        FrequencyPenaltyLogitsProcessor(penalty=-0.1)

    assert True, "Negative frequency penalties must be rejected at initialization time."


def test_batch_counts_are_independent() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 0, 1], [2, 2, 2]], dtype=torch.long))
    warped = FrequencyPenaltyLogitsProcessor(penalty=0.5)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.isclose(
        warped[0, 0], torch.tensor(2.0)
    ), "The first batch row should use only its own token counts."
    assert torch.isclose(
        warped[1, 2], torch.tensor(-0.5)
    ), "The second batch row should use only its own token counts."
