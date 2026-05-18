"""Concrete logit processors and warpers."""

from __future__ import annotations

from typing import Any, cast

import torch

from logit_lens.base import LogitsProcessor, LogitsWarper
from logit_lens.utils import assert_valid_scores, entropy, log_softmax_stable


def _serialize_float(value: float) -> float | str:
    """Convert non-finite floats into JSON-compatible string markers."""

    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return value


def _restore_scores_dtype(
    scores: torch.Tensor, dtype: torch.dtype
) -> torch.FloatTensor:
    """Cast an intermediate tensor back to the caller's dtype."""

    return cast(torch.FloatTensor, scores.to(dtype=dtype))


class TemperatureLogitsWarper(LogitsWarper):
    """Scale logits by a positive temperature.

    Args:
        temperature: Positive scaling factor. Values below one sharpen the distribution,
            values above one flatten it.
    """

    def __init__(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.temperature = float(temperature)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        scores_float = scores.float()

        if self.temperature < 1e-7:
            max_scores = scores_float.max(dim=-1, keepdim=True).values
            output = scores_float.masked_fill(scores_float < max_scores, float("-inf"))
        elif self.temperature == 1.0:
            output = scores_float.clone()
        else:
            output = scores_float / self.temperature

        assert_valid_scores(output, context=self.__class__.__name__)
        return _restore_scores_dtype(output, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the warper configuration."""

        return {
            "type": self.__class__.__name__,
            "params": {"temperature": self.temperature},
        }

    def __repr__(self) -> str:
        return f"TemperatureLogitsWarper(temperature={self.temperature})"


class TopKLogitsWarper(LogitsWarper):
    """Keep only the ``k`` highest-scoring tokens in each batch row.

    Args:
        top_k: Number of tokens to retain before masking the remainder.
        filter_value: Value written into masked positions.
        min_tokens_to_keep: Lower bound on the number of tokens preserved per row.
    """

    def __init__(
        self,
        top_k: int,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}."
            )
        self.top_k = int(top_k)
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = int(min_tokens_to_keep)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        scores_float = scores.float()
        vocab_size = scores_float.shape[-1]
        effective_k = min(max(self.top_k, self.min_tokens_to_keep), vocab_size)

        if effective_k >= vocab_size:
            return _restore_scores_dtype(scores_float.clone(), scores.dtype)

        _, topk_indices = torch.topk(scores_float, k=effective_k, dim=-1)
        keep_mask = torch.zeros_like(scores_float, dtype=torch.bool)
        keep_mask.scatter_(1, topk_indices, True)
        filtered = scores_float.masked_fill(~keep_mask, self.filter_value)

        assert_valid_scores(filtered, context=self.__class__.__name__)
        return _restore_scores_dtype(filtered, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the warper configuration."""

        return {
            "type": self.__class__.__name__,
            "params": {
                "top_k": self.top_k,
                "filter_value": _serialize_float(self.filter_value),
                "min_tokens_to_keep": self.min_tokens_to_keep,
            },
        }

    def __repr__(self) -> str:
        return (
            "TopKLogitsWarper("
            f"top_k={self.top_k}, filter_value={self.filter_value}, "
            f"min_tokens_to_keep={self.min_tokens_to_keep})"
        )


class TopPLogitsWarper(LogitsWarper):
    """Keep the smallest prefix of tokens whose cumulative probability exceeds ``top_p``.

    Args:
        top_p: Probability mass to preserve.
        filter_value: Value written into masked positions.
        min_tokens_to_keep: Lower bound on the number of tokens preserved per row.
    """

    def __init__(
        self,
        top_p: float,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1,
    ) -> None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}.")
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}."
            )
        self.top_p = float(top_p)
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = int(min_tokens_to_keep)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        scores_float = scores.float()
        vocab_size = scores_float.shape[-1]

        if self.top_p == 1.0 and self.min_tokens_to_keep <= 1:
            return _restore_scores_dtype(scores_float.clone(), scores.dtype)

        sorted_logits, sorted_indices = torch.sort(
            scores_float, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p

        # Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020),
        # nucleus sampling keeps the token that first crosses the cumulative mass threshold.
        shifted_mask = sorted_indices_to_remove.clone()
        shifted_mask[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        shifted_mask[..., 0] = False

        keep_count = min(self.min_tokens_to_keep, vocab_size)
        shifted_mask[..., :keep_count] = False

        indices_to_remove = torch.zeros_like(shifted_mask)
        indices_to_remove.scatter_(1, sorted_indices, shifted_mask)
        filtered = scores_float.masked_fill(indices_to_remove, self.filter_value)

        assert_valid_scores(filtered, context=self.__class__.__name__)
        return _restore_scores_dtype(filtered, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the warper configuration."""

        return {
            "type": self.__class__.__name__,
            "params": {
                "top_p": self.top_p,
                "filter_value": _serialize_float(self.filter_value),
                "min_tokens_to_keep": self.min_tokens_to_keep,
            },
        }

    def __repr__(self) -> str:
        return (
            "TopPLogitsWarper("
            f"top_p={self.top_p}, filter_value={self.filter_value}, "
            f"min_tokens_to_keep={self.min_tokens_to_keep})"
        )


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Penalize tokens that have already appeared in the generated context.

    Args:
        penalty: Multiplicative penalty applied using the Hugging Face formulation.
    """

    def __init__(self, penalty: float) -> None:
        if penalty < 1.0:
            raise ValueError(f"penalty must be >= 1.0, got {penalty}.")
        self.penalty = float(penalty)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_float = scores.float()
        if input_ids.numel() == 0 or input_ids.shape[-1] == 0 or self.penalty == 1.0:
            return _restore_scores_dtype(scores_float.clone(), scores.dtype)

        gathered = torch.gather(scores_float, dim=1, index=input_ids)
        penalized = torch.where(
            gathered < 0.0, gathered * self.penalty, gathered / self.penalty
        )
        updated = scores_float.clone()
        updated.scatter_(1, input_ids, penalized)

        assert_valid_scores(updated, context=self.__class__.__name__)
        return _restore_scores_dtype(updated, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the processor configuration."""

        return {"type": self.__class__.__name__, "params": {"penalty": self.penalty}}

    def __repr__(self) -> str:
        return f"RepetitionPenaltyLogitsProcessor(penalty={self.penalty})"


class TypicalDecodingLogitsWarper(LogitsWarper):
    """Filter tokens by how typical their information content is.

    Args:
        mass: Probability mass to preserve after sorting by typical deviation.
        filter_value: Value written into masked positions.
        min_tokens_to_keep: Lower bound on the number of tokens preserved per row.
    """

    def __init__(
        self,
        mass: float = 0.9,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1,
    ) -> None:
        if not 0.0 < mass <= 1.0:
            raise ValueError(f"mass must be in (0.0, 1.0], got {mass}.")
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}."
            )
        self.mass = float(mass)
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = int(min_tokens_to_keep)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        scores_float = scores.float()
        vocab_size = scores_float.shape[-1]

        log_probs = log_softmax_stable(scores_float)
        probs = log_probs.exp()
        row_entropy = entropy(log_probs).unsqueeze(-1)

        # Meister et al., "Typical Decoding for Natural Language Generation" (2023),
        # Eq. 2 defines information content as -log p(x), and Eq. 3 ranks tokens by
        # the deviation | -log p(x) - H(p) | from the distribution entropy.
        information_content = -log_probs
        typical_deviation = torch.abs(information_content - row_entropy)

        deviation_span = typical_deviation.amax(dim=-1) - typical_deviation.amin(dim=-1)
        uniform_rows = deviation_span <= 1e-6
        if torch.all(uniform_rows):
            return _restore_scores_dtype(scores_float.clone(), scores.dtype)

        sorted_deviation, sorted_indices = torch.sort(
            typical_deviation, descending=False, dim=-1
        )
        del sorted_deviation
        sorted_probs = torch.gather(probs, dim=1, index=sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.mass

        shifted_mask = sorted_indices_to_remove.clone()
        shifted_mask[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        shifted_mask[..., 0] = False

        keep_count = min(self.min_tokens_to_keep, vocab_size)
        shifted_mask[..., :keep_count] = False
        if torch.any(uniform_rows):
            shifted_mask[uniform_rows] = False

        indices_to_remove = torch.zeros_like(shifted_mask)
        indices_to_remove.scatter_(1, sorted_indices, shifted_mask)
        filtered = scores_float.masked_fill(indices_to_remove, self.filter_value)

        assert_valid_scores(filtered, context=self.__class__.__name__)
        return _restore_scores_dtype(filtered, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the warper configuration."""

        return {
            "type": self.__class__.__name__,
            "params": {
                "mass": self.mass,
                "filter_value": _serialize_float(self.filter_value),
                "min_tokens_to_keep": self.min_tokens_to_keep,
            },
        }

    def __repr__(self) -> str:
        return (
            "TypicalDecodingLogitsWarper("
            f"mass={self.mass}, filter_value={self.filter_value}, "
            f"min_tokens_to_keep={self.min_tokens_to_keep})"
        )


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    """Subtract a penalty proportional to the count of each previously seen token.

    Args:
        penalty: Non-negative coefficient multiplied by token occurrence counts.
    """

    def __init__(self, penalty: float) -> None:
        if penalty < 0.0:
            raise ValueError(f"penalty must be >= 0.0, got {penalty}.")
        self.penalty = float(penalty)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_float = scores.float()
        if input_ids.numel() == 0 or input_ids.shape[-1] == 0 or self.penalty == 0.0:
            return _restore_scores_dtype(scores_float.clone(), scores.dtype)

        vocab_size = scores_float.shape[-1]
        count_rows = [
            torch.bincount(batch_ids, minlength=vocab_size) for batch_ids in input_ids
        ]
        counts = torch.stack(count_rows, dim=0).to(
            device=scores_float.device, dtype=scores_float.dtype
        )
        updated = scores_float - (self.penalty * counts)

        assert_valid_scores(updated, context=self.__class__.__name__)
        return _restore_scores_dtype(updated, scores.dtype)

    def to_config(self) -> dict[str, Any]:
        """Serialize the processor configuration."""

        return {"type": self.__class__.__name__, "params": {"penalty": self.penalty}}

    def __repr__(self) -> str:
        return f"FrequencyPenaltyLogitsProcessor(penalty={self.penalty})"


__all__ = [
    "FrequencyPenaltyLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
    "TypicalDecodingLogitsWarper",
]
