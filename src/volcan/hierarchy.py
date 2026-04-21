"""Hierarchical Macro-Cells (CEO directive, post-ceiling experiment).

The measured bottleneck: 5 experiments all landed at 3/31 = 9.7% on ARC-AGI-2.
Varying data, capacity, diversity, and adapter rank all failed to unlock more
tasks. The remaining unexplored axis is the **information bottleneck**: pure
local 3×3 NCA updates propagate signal at 1 cell/step, which means "global
rules" (move everything to the corner, count total cells, etc.) require
O(grid_size) iterations to compose.

HRM (Hierarchical Reasoning Model, 2025) showed that multi-timescale loops
with macro-level aggregation let small models solve problems that otherwise
need much larger flat models. This module ports that idea to an NCA substrate.

Design:
  - **Macro-grid**: 10×10 cells overlaid on the base 30×30 grid (block size 3)
  - **Macro-state**: 16 channels per macro-cell
  - **Aggregation**: base state (B, 59, 30, 30) is 3×3 avg-pooled to macro grid
    (B, 59, 10, 10), then projected to (B, 16, 10, 10) via a 1x1 conv
  - **Macro update**: a small MLP (1x1 convs) updates the macro state
  - **Broadcast**: macro state upsampled to (B, 16, 30, 30) via nearest-neighbor
    and read by every base cell in its parent block
  - **Timing**: every step (simpler than multi-timescale for v1; can add
    frequency gating if empirical results demand it)

Parameter budget: ~15-20K, well inside our efficiency envelope.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroCells(nn.Module):
    """Hierarchical macro-cell layer over a base NCA grid.

    Maintains a state tensor at (B, macro_channels, macro_H, macro_W) where
    macro_H = base_H // block_size. On each call:
      1. Pool base state down to the macro grid via avg-pool
      2. Project pooled features into macro channel space
      3. Update the macro state with a small MLP over (pooled, prev_macro)
      4. Upsample to base resolution and return the broadcast field

    The macro state is maintained by the caller across NCA steps; this module
    is stateless (except its weights). The caller initializes with
    `init_macro_state(...)`.
    """

    def __init__(
        self,
        base_channels: int = 59,
        macro_channels: int = 16,
        macro_hidden: int = 32,
        block_size: int = 3,
        base_grid_size: int = 30,
    ):
        super().__init__()
        if base_grid_size % block_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must divide base_grid_size ({base_grid_size}); "
                f"use 3 for 30×30 (→ 10×10 macro grid) or 2 for even sizes."
            )
        self.base_channels = base_channels
        self.macro_channels = macro_channels
        self.block_size = block_size
        self.base_grid_size = base_grid_size
        self.macro_grid_size = base_grid_size // block_size

        # Pool-then-project: (B, base_channels, macro_H, macro_W) → (B, macro_channels, ...)
        self.pool_project = nn.Conv2d(base_channels, macro_channels, kernel_size=1)

        # Macro update MLP: takes (pooled_features, prev_macro_state) and outputs delta.
        # Input dim: macro_channels (pooled) + macro_channels (prev state) = 2 * macro_channels
        # Output dim: macro_channels (residual delta)
        self.macro_update = nn.Sequential(
            nn.Conv2d(2 * macro_channels, macro_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(macro_hidden, macro_channels, kernel_size=1),
        )
        # Zero-init the delta head so early training dynamics are neutral.
        nn.init.zeros_(self.macro_update[-1].weight)
        if self.macro_update[-1].bias is not None:
            nn.init.zeros_(self.macro_update[-1].bias)

    def init_macro_state(self, base_state: torch.Tensor) -> torch.Tensor:
        """Zero-initialized macro state sized for the given base batch."""
        b = base_state.shape[0]
        return torch.zeros(
            b,
            self.macro_channels,
            self.macro_grid_size,
            self.macro_grid_size,
            device=base_state.device,
            dtype=base_state.dtype,
        )

    def forward(
        self,
        base_state: torch.Tensor,
        macro_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the macro state and produce a broadcast field for the base.

        Args:
            base_state: (B, base_channels, base_H, base_W)
            macro_state: (B, macro_channels, macro_H, macro_W)

        Returns:
            new_macro_state: (B, macro_channels, macro_H, macro_W)
            broadcast_field: (B, macro_channels, base_H, base_W) — each base cell
                gets its parent macro-cell's state, via nearest-neighbor upsample.
        """
        # 1. Pool base to macro grid
        pooled = F.avg_pool2d(base_state, kernel_size=self.block_size, stride=self.block_size)
        pooled_proj = self.pool_project(pooled)  # (B, macro_channels, macro_H, macro_W)

        # 2. Update macro state via residual MLP
        macro_input = torch.cat([pooled_proj, macro_state], dim=1)
        delta = self.macro_update(macro_input)
        new_macro_state = macro_state + delta

        # 3. Broadcast back to base grid via nearest-neighbor upsample
        broadcast = F.interpolate(
            new_macro_state,
            scale_factor=self.block_size,
            mode="nearest",
        )

        return new_macro_state, broadcast

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
