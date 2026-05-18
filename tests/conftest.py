"""Shared pytest fixtures for logit_lens."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from logit_lens.base import LogitsProcessor
from logit_lens.processors import (
    FrequencyPenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalDecodingLogitsWarper,
)


@pytest.fixture
def vocab_size() -> int:
    return 50257


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def seq_len() -> int:
    return 32


@pytest.fixture
def random_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(13)
    return torch.randn(batch_size, vocab_size, generator=generator)


@pytest.fixture
def peaked_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    logits = torch.full((batch_size, vocab_size), -8.0)
    logits[:, 7] = 12.0
    return logits


@pytest.fixture
def input_ids(batch_size: int, seq_len: int, vocab_size: int) -> torch.LongTensor:
    generator = torch.Generator().manual_seed(29)
    return cast(
        torch.LongTensor,
        torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator),
    )


@pytest.fixture
def all_processors() -> list[LogitsProcessor]:
    return [
        TemperatureLogitsWarper(temperature=1.0),
        TopKLogitsWarper(top_k=50),
        TopPLogitsWarper(top_p=0.9),
        RepetitionPenaltyLogitsProcessor(penalty=1.2),
        TypicalDecodingLogitsWarper(mass=0.9),
        FrequencyPenaltyLogitsProcessor(penalty=0.5),
    ]
