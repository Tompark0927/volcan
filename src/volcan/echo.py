"""Echo detection — Pillar 2.

Maintains a rolling buffer of recent NCA states and computes echo_k = cosine
similarity between s_t and s_{t-k} for k in {1, 2, 3, 4}.

Interpretation (matches architecture.md §5 Pillar 2):
  - echo_1 ≈ 1                               → fixed point, model confident
  - echo_1 ≈ 0, echo_2 ≈ 1                   → period-2 (two competing rules)
  - echo_1 ≈ 0, echo_2 ≈ 0, echo_3 ≈ 1       → period-3 (rare)
  - all echoes low                           → chaos

This module is differentiable end-to-end so it can be used both as an inference-
time regime detector and as a training-time loss term.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Regime = Literal["stable", "period_2", "period_3", "chaos"]


@dataclass
class EchoReading:
    """Result of computing echoes at a specific step."""

    echo_1: torch.Tensor  # (B,)
    echo_2: torch.Tensor  # (B,)
    echo_3: torch.Tensor  # (B,)
    echo_4: torch.Tensor  # (B,)

    def regime(self, threshold: float = 0.9) -> list[Regime]:
        """Per-batch regime classification using a similarity threshold."""
        out: list[Regime] = []
        e1, e2, e3, _ = self.echo_1, self.echo_2, self.echo_3, self.echo_4
        for i in range(e1.shape[0]):
            if e1[i] >= threshold:
                out.append("stable")
            elif e2[i] >= threshold:
                out.append("period_2")
            elif e3[i] >= threshold:
                out.append("period_3")
            else:
                out.append("chaos")
        return out

    def best_echo(self) -> torch.Tensor:
        """Per-batch max over (echo_1, echo_2, echo_3) — strongest clean periodicity."""
        return torch.stack([self.echo_1, self.echo_2, self.echo_3], dim=-1).max(dim=-1).values


def cosine_similarity_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two batched tensors, flattened per sample.

    Args:
        a, b: (B, ...) — same shape.

    Returns:
        (B,) similarity in [-1, 1].
    """
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(flat_a, flat_b, dim=-1, eps=1e-8)


class EchoDetector:
    """Rolling buffer of recent states; computes echo_k on demand.

    Note: this is a *plain* (non-nn.Module) class because it has no parameters.
    It's used during training and inference as an observer.
    """

    def __init__(self, max_lag: int = 4):
        self.max_lag = max_lag
        self._buffer: deque[torch.Tensor] = deque(maxlen=max_lag + 1)

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, state: torch.Tensor) -> None:
        """Append a new state. Detached so the buffer doesn't hold gradients."""
        self._buffer.append(state.detach())

    def push_with_grad(self, state: torch.Tensor) -> None:
        """Append a new state, keeping gradients (for use inside loss terms)."""
        self._buffer.append(state)

    def has_lag(self, k: int) -> bool:
        """Whether the buffer holds at least k+1 entries (so echo_k is defined)."""
        return len(self._buffer) > k

    def echo(self, current: torch.Tensor) -> EchoReading:
        """Compute echo_1 through echo_4 against the buffered history.

        If insufficient history is available, the missing echoes are returned
        as zeros (treated as "no signal yet").
        """
        b = current.shape[0]
        device = current.device
        zero = torch.zeros(b, device=device)
        history = list(self._buffer)
        n = len(history)

        def _echo(k: int) -> torch.Tensor:
            if n < k:
                return zero
            past = history[-k]
            return cosine_similarity_per_sample(current, past)

        return EchoReading(
            echo_1=_echo(1),
            echo_2=_echo(2),
            echo_3=_echo(3),
            echo_4=_echo(4),
        )
