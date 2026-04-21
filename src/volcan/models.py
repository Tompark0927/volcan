"""NCA models for Volcan.

Week 1 ships `BasicNCA` only — an isotropic Mordvintsev-style Neural Cellular
Automaton. This is the sanity-check baseline; the five-pillar Volcan cell
(anisotropic forces, mycelial bypass, spectral tension, ghost field, apoptosis)
comes in Week 2.

BasicNCA is intentionally simple:
  - Per-cell state of C channels: the first VOCAB_SIZE are color logits, the
    rest are "hidden" scratch.
  - Perception: three parallel 3×3 sensors (identity, Sobel-x, Sobel-y) —
    fixed, not learned. This matches the Growing NCA convention and gives the
    network gradient access to local spatial structure without any learned
    convolution parameters.
  - Update: a small MLP on the concatenated perception vector, producing a
    residual delta.
  - Stochastic fire: only a fraction of cells update per step (Mordvintsev's
    stability trick). This is important for training — without it, BPTT through
    long rollouts is much less stable.

The model is deliberately kept small (~15k params) so it trains in seconds on
CPU or MPS for the smoke test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arc import VOCAB_SIZE


def _sobel_kernels() -> torch.Tensor:
    """Return a (3, 1, 3, 3) kernel stack: identity, Sobel-x, Sobel-y.

    These are the fixed perception filters used by the original Growing NCA
    (Mordvintsev et al. 2020). They give each cell a view of (a) its own state
    and (b) local gradients in x and y.
    """
    identity = torch.tensor(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
    )
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ) / 8.0
    sobel_y = sobel_x.t().clone()
    return torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]  # (3, 1, 3, 3)


class BasicNCA(nn.Module):
    """Isotropic NCA, Mordvintsev-style. Week 1 sanity baseline."""

    def __init__(
        self,
        channels: int = 24,
        hidden: int = 64,
        vocab_size: int = VOCAB_SIZE,
        fire_rate: float = 0.5,
    ):
        """
        Args:
            channels: total per-cell state dimensionality. First `vocab_size`
                channels are color logits (softmax-able), the rest are hidden.
            hidden: MLP hidden layer width.
            vocab_size: number of color classes (10 ARC colors + 1 "outside").
            fire_rate: fraction of cells that update on each step (stochastic
                fire, Mordvintsev trick).
        """
        super().__init__()
        if channels < vocab_size:
            raise ValueError(
                f"channels ({channels}) must be >= vocab_size ({vocab_size})"
            )
        self.channels = channels
        self.vocab_size = vocab_size
        self.fire_rate = fire_rate

        # Fixed perception kernels (identity + Sobel-x + Sobel-y), per channel.
        # Total perception dim per cell = 3 × channels.
        self.register_buffer("perception_kernel", _sobel_kernels())  # (3, 1, 3, 3)

        # Update MLP: (3 * channels) -> hidden -> channels (delta)
        self.update = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Mordvintsev initializes the final layer to zero so the initial
        # update is a no-op and the residual dynamics start neutral.
        nn.init.zeros_(self.update[-1].weight)
        if self.update[-1].bias is not None:
            nn.init.zeros_(self.update[-1].bias)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the fixed perception filters to every channel independently.

        Input:  (B, channels, H, W)
        Output: (B, 3*channels, H, W)
        """
        b, c, h, w = state.shape
        # Repeat the 3 kernels for every channel (depthwise).
        k = self.perception_kernel.repeat(c, 1, 1, 1)  # (3*c, 1, 3, 3)
        return F.conv2d(state, k, padding=1, groups=c)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """One NCA update step with stochastic fire."""
        perception = self.perceive(state)
        delta = self.update(perception)

        # Stochastic fire: mask out some fraction of cells.
        if self.fire_rate < 1.0:
            b, _, h, w = state.shape
            mask = (torch.rand(b, 1, h, w, device=state.device) < self.fire_rate).float()
            delta = delta * mask

        return state + delta

    def forward(
        self,
        state: torch.Tensor,
        steps: int = 32,
    ) -> torch.Tensor:
        """Run the NCA for `steps` iterations, returning the final state."""
        for _ in range(steps):
            state = self.step(state)
        return state

    def init_state(
        self,
        input_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Build an initial state from a one-hot input grid.

        Input:  (B, vocab_size, H, W) one-hot
        Output: (B, channels, H, W) — first vocab_size channels = input, rest 0.
        """
        b, v, h, w = input_onehot.shape
        if v != self.vocab_size:
            raise ValueError(
                f"input_onehot has {v} channels, expected {self.vocab_size}"
            )
        state = torch.zeros(b, self.channels, h, w, device=input_onehot.device)
        state[:, : self.vocab_size] = input_onehot
        return state

    def color_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Extract color logits from the first vocab_size channels."""
        return state[:, : self.vocab_size]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
