"""Utility helpers for numerically stable logit processing."""

from __future__ import annotations

import torch


def safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute a clamped logarithm.

    Args:
        x: Input tensor.
        eps: Lower clamp used to avoid ``log(0)``.

    Returns:
        The elementwise logarithm of ``x`` after clamping to ``eps``.

    Raises:
        ValueError: Raised when ``eps`` is not strictly positive.
    """

    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    x_float = x.float()
    return torch.log(torch.clamp(x_float, min=eps))


def log_softmax_stable(scores: torch.Tensor) -> torch.Tensor:
    """Compute a numerically stable log-softmax.

    Args:
        scores: Logits with shape ``(..., vocab_size)``.

    Returns:
        Float32 log probabilities with the same shape as ``scores``.
    """

    scores_float = scores.float()
    row_max = scores_float.max(dim=-1, keepdim=True).values
    shifted = scores_float - row_max
    return shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)


def assert_valid_scores(scores: torch.Tensor, context: str = "") -> None:
    """Validate that a score tensor is usable for generation.

    Args:
        scores: Logit tensor with shape ``(batch_size, vocab_size)``.
        context: Optional label describing the caller for clearer error messages.

    Raises:
        AssertionError: Raised if ``scores`` contains NaNs or a row where every element is
            infinite.
    """

    prefix = f"{context}: " if context else ""
    if torch.isnan(scores).any():
        nan_locations = torch.nonzero(torch.isnan(scores), as_tuple=False)
        first_location = nan_locations[0].tolist()
        raise AssertionError(
            f"{prefix}scores contain NaN at batch index {first_location[0]} and "
            f"token index {first_location[1]}."
        )

    if scores.ndim == 0:
        return

    all_inf_rows = torch.isinf(scores).all(dim=-1)
    if all_inf_rows.any():
        rows = torch.nonzero(all_inf_rows, as_tuple=False).flatten().tolist()
        raise AssertionError(
            f"{prefix}scores contain all-inf rows at batch indices {rows}."
        )


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy from log probabilities.

    Args:
        log_probs: Log probabilities with shape ``(..., vocab_size)``.

    Returns:
        The entropy for each row, with shape ``log_probs.shape[:-1]``.
    """

    log_probs_float = log_probs.float()
    probs = log_probs_float.exp()
    return -(probs * log_probs_float).sum(dim=-1)
