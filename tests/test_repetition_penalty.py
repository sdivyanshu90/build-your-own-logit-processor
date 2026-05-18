"""Tests for repetition penalties."""

from __future__ import annotations

from typing import cast

import torch

from logit_lens.processors import RepetitionPenaltyLogitsProcessor


def _as_long_tensor(tensor: torch.Tensor) -> torch.LongTensor:
    return cast(torch.LongTensor, tensor)


def _as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, tensor)


def test_penalty_reduces_logit_of_seen_tokens() -> None:
    scores = torch.tensor([[2.0, -1.5, 0.5, 0.1]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 1]], dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=2.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert (
        warped[0, 0] < scores[0, 0]
    ), "Seen positive logits should be reduced by the repetition penalty."
    assert (
        warped[0, 1] < scores[0, 1]
    ), "Seen negative logits should become more negative under the penalty."


def test_penalty_does_not_affect_unseen_tokens() -> None:
    scores = torch.tensor([[2.0, -1.5, 0.5, 0.1]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 1]], dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=2.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.equal(
        warped[0, 2:], scores[0, 2:]
    ), "Tokens absent from input_ids must remain unchanged."


def test_empty_input_ids_is_identity() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.zeros((1, 0), dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=1.5)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.equal(warped, scores), "An empty prompt should leave logits unchanged."


def test_penalty_one_is_identity() -> None:
    scores = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0, 1]], dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=1.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.equal(
        warped, scores
    ), "Penalty 1.0 should behave like the identity transformation."


def test_positive_logit_is_divided() -> None:
    scores = torch.tensor([[4.0, 1.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0]], dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=2.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.isclose(
        warped[0, 0], torch.tensor(2.0)
    ), "Positive seen logits should be divided by the penalty factor."


def test_negative_logit_is_multiplied() -> None:
    scores = torch.tensor([[-3.0, 1.0]], dtype=torch.float32)
    input_ids = _as_long_tensor(torch.tensor([[0]], dtype=torch.long))
    warped = RepetitionPenaltyLogitsProcessor(penalty=2.0)(
        input_ids, _as_float_tensor(scores)
    )

    assert torch.isclose(
        warped[0, 0], torch.tensor(-6.0)
    ), "Negative seen logits should be multiplied by the penalty factor."


def test_duplicate_tokens_in_input() -> None:
    scores = torch.tensor([[2.0, 1.0, -0.5]], dtype=torch.float32)
    once = _as_long_tensor(torch.tensor([[1]], dtype=torch.long))
    repeated = _as_long_tensor(torch.tensor([[1, 1, 1]], dtype=torch.long))
    processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    once_warped = processor(once, _as_float_tensor(scores))
    repeated_warped = processor(repeated, _as_float_tensor(scores))

    assert torch.equal(
        once_warped, repeated_warped
    ), "Repeated appearances of the same token should not stack the repetition penalty."
