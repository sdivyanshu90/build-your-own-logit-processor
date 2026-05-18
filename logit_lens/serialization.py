"""Serialization helpers for logit processor pipelines.

The JSON format is a single object with a ``processors`` field containing an ordered list
of processor specifications. Each processor specification has a ``type`` string matching a
registered class name and a ``params`` object containing constructor keyword arguments.

Example:
{
  "processors": [
    {
      "type": "TemperatureLogitsWarper",
      "params": {
        "temperature": 0.8
      }
    },
    {
      "type": "TopPLogitsWarper",
      "params": {
        "top_p": 0.92,
        "filter_value": "-inf",
        "min_tokens_to_keep": 1
      }
    }
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

from logit_lens.base import LogitsProcessor
from logit_lens.pipeline import LogitsProcessorList
from logit_lens.processors import (
    FrequencyPenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalDecodingLogitsWarper,
)

PROCESSOR_REGISTRY: dict[str, type[LogitsProcessor]] = {
    "TemperatureLogitsWarper": TemperatureLogitsWarper,
    "TopKLogitsWarper": TopKLogitsWarper,
    "TopPLogitsWarper": TopPLogitsWarper,
    "RepetitionPenaltyLogitsProcessor": RepetitionPenaltyLogitsProcessor,
    "TypicalDecodingLogitsWarper": TypicalDecodingLogitsWarper,
    "FrequencyPenaltyLogitsProcessor": FrequencyPenaltyLogitsProcessor,
}


def save_pipeline(pipeline: LogitsProcessorList, path: str | Path) -> None:
    """Save a processor pipeline to JSON.

    Args:
        pipeline: Pipeline to serialize.
        path: Output JSON path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pipeline.to_config(), indent=2) + "\n", encoding="utf-8"
    )


def load_pipeline(path: str | Path) -> LogitsProcessorList:
    """Load a processor pipeline from JSON.

    Args:
        path: Input JSON path.

    Returns:
        The deserialized processor pipeline.
    """

    input_path = Path(path)
    config = json.loads(input_path.read_text(encoding="utf-8"))
    return LogitsProcessorList.from_config(config)


__all__ = ["PROCESSOR_REGISTRY", "load_pipeline", "save_pipeline"]
