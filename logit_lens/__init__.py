"""Public package interface for logit_lens."""

from logit_lens.base import LogitsProcessor, LogitsWarper, StatefulLogitsProcessor
from logit_lens.pipeline import LogitsProcessorList, build_standard_pipeline
from logit_lens.processors import (
    FrequencyPenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalDecodingLogitsWarper,
)
from logit_lens.serialization import PROCESSOR_REGISTRY, load_pipeline, save_pipeline
from logit_lens.utils import (
    assert_valid_scores,
    entropy,
    log_softmax_stable,
    safe_log,
)

__version__ = "0.1.0"

__all__ = [
    "PROCESSOR_REGISTRY",
    "FrequencyPenaltyLogitsProcessor",
    "LogitsProcessor",
    "LogitsProcessorList",
    "LogitsWarper",
    "RepetitionPenaltyLogitsProcessor",
    "StatefulLogitsProcessor",
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
    "TypicalDecodingLogitsWarper",
    "__version__",
    "assert_valid_scores",
    "build_standard_pipeline",
    "entropy",
    "load_pipeline",
    "log_softmax_stable",
    "safe_log",
    "save_pipeline",
]
