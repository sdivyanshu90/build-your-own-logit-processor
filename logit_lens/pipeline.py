"""Pipeline composition for logit processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from logit_lens.base import LogitsProcessor
from logit_lens.processors import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

if TYPE_CHECKING:

    class TransformersLogitsProcessorList(list[LogitsProcessor]):
        """Typed fallback for mypy when transformers stubs are unavailable."""

else:
    from transformers import LogitsProcessorList as TransformersLogitsProcessorList


def _deserialize_config_value(value: object) -> object:
    """Restore JSON-friendly non-finite markers into Python values."""

    if value == "-inf":
        return float("-inf")
    if value == "inf":
        return float("inf")
    return value


class LogitsProcessorList(TransformersLogitsProcessorList):
    """Ordered chain of logit processors.

    Instances apply each processor in insertion order, feeding the output scores of one
    stage into the next stage.
    """

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply every processor in order to the provided scores."""

        processed_scores = scores
        for processor in self:
            processed_scores = processor(input_ids, processed_scores)
        return processed_scores

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LogitsProcessorList":
        """Instantiate a processor list from a serialized configuration.

        Args:
            config: Dictionary containing a ``processors`` list.

        Returns:
            A fully constructed processor pipeline.

        Raises:
            KeyError: Raised when a processor type is not registered.
            ValueError: Raised when the configuration shape is invalid.
        """

        if "processors" not in config or not isinstance(config["processors"], list):
            raise ValueError("config must contain a 'processors' list.")

        from logit_lens.serialization import PROCESSOR_REGISTRY

        processors: list[LogitsProcessor] = []
        for raw_entry in config["processors"]:
            if not isinstance(raw_entry, dict):
                raise ValueError("each processor config entry must be a dictionary.")
            raw_type = raw_entry.get("type")
            raw_params = raw_entry.get("params", {})
            if not isinstance(raw_type, str):
                raise ValueError(
                    "processor config entries must include a string 'type'."
                )
            if not isinstance(raw_params, dict):
                raise ValueError(
                    "processor config entries must include a dict 'params'."
                )

            processor_cls = PROCESSOR_REGISTRY[raw_type]
            decoded_params = {
                key: _deserialize_config_value(value)
                for key, value in raw_params.items()
            }
            processor_factory = cast(Callable[..., LogitsProcessor], processor_cls)
            processor = processor_factory(**decoded_params)
            processors.append(processor)
        return cls(processors)

    def to_config(self) -> dict[str, Any]:
        """Serialize the processor chain into a JSON-compatible dictionary."""

        return {"processors": [processor.to_config() for processor in self]}

    def __repr__(self) -> str:
        processor_chain = " -> ".join(repr(processor) for processor in self)
        return f"LogitsProcessorList([{processor_chain}])"


def build_standard_pipeline(
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
) -> LogitsProcessorList:
    """Build a common sampling pipeline with stable default ordering.

    Args:
        temperature: Optional temperature warper.
        top_k: Optional top-k truncation threshold.
        top_p: Optional nucleus sampling mass.
        repetition_penalty: Optional repetition penalty.

    Returns:
        A processor list ordered for autoregressive decoding.
    """

    processors: list[LogitsProcessor] = []

    if temperature is not None:
        # Temperature is applied first because it globally rescales the score landscape
        # without changing token identity, while context-dependent penalties must happen
        # before truncating warpers so top-k/top-p see the already-penalized logits.
        processors.append(TemperatureLogitsWarper(temperature=temperature))
    if repetition_penalty is not None:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if top_k is not None:
        processors.append(TopKLogitsWarper(top_k=top_k))
    if top_p is not None:
        processors.append(TopPLogitsWarper(top_p=top_p))

    return LogitsProcessorList(processors)


__all__ = ["LogitsProcessorList", "build_standard_pipeline"]
