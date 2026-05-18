"""Base abstractions for composing logit processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:

    class TransformersLogitsProcessor:
        """Typed fallback for mypy when transformers stubs are unavailable."""

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            raise NotImplementedError

else:
    from transformers import LogitsProcessor as TransformersLogitsProcessor


class LogitsProcessor(TransformersLogitsProcessor, ABC):
    """Abstract interface for transforming next-token logits.

    Implementations receive the already generated token ids and the model's raw next-token
    logits. The processor returns a tensor with the same shape as ``scores`` and is expected
    to preserve the batch dimension.
    """

    @abstractmethod
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Transform raw next-token logits.

        Args:
            input_ids: Token ids with shape ``(batch_size, sequence_length)``.
            scores: Raw model logits with shape ``(batch_size, vocab_size)``. These are
                logits, not probabilities.

        Returns:
            A tensor of transformed logits with shape ``(batch_size, vocab_size)``.
        """

    def to_config(self) -> dict[str, Any]:
        """Serialize the processor into a JSON-compatible configuration payload.

        Returns:
            A dictionary with the processor type name and constructor parameters.

        Raises:
            NotImplementedError: Raised when a concrete processor does not implement
                serialization.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement to_config() for serialization."
        )


class LogitsWarper(LogitsProcessor, ABC):
    """Semantic subtype for processors that reshape a distribution.

    Warpers typically scale, truncate, or mask a next-token distribution so that the
    support or sharpness changes. This is distinct from processors that apply context-
    dependent additive or multiplicative penalties without necessarily removing support.
    """


class StatefulLogitsProcessor(LogitsProcessor, ABC):
    """Mixin for processors that carry per-generation state.

    Stateful processors may keep tensors keyed by state name, usually shaped with the batch
    dimension first so each generation stream has isolated state.

    Args:
        state: Optional initial state tensors keyed by name.
    """

    def __init__(self, state: Mapping[str, torch.Tensor] | None = None) -> None:
        self._state: dict[str, torch.Tensor] = {}
        if state is not None:
            self._state = {name: tensor.clone() for name, tensor in state.items()}

    @property
    def state(self) -> Mapping[str, torch.Tensor]:
        """Return a read-only view of the current processor state."""

        return self._state

    def reset_state(self) -> None:
        """Clear all stored state between independent generation calls."""

        self._state.clear()
