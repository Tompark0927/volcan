"""D8 dihedral symmetries — used for D8-Ensemble Resonance inference (Week 6).

The D8 group has 8 elements: identity, three rotations (90/180/270 CCW), and
four reflections (horizontal, vertical, both diagonals). Each element is a
function that operates on a (..., H, W) tensor and has a known inverse.

The D8-Ensemble inference protocol (architecture.md Pillar 2 + CEO's idea):
  1. For each test task, generate 8 symmetric versions of (demos, query)
  2. Run Phase A on each version
  3. Use Pillar 2 echo detection to measure how confidently each version
     converges (echo_1 of the final Phase A state)
  4. Pick the symmetry with the highest echo_1 — the model is "most sure"
     about its rule encoding under that orientation
  5. Run Phase B on that symmetry, then apply the inverse to get back to
     the original orientation

We define each element by name and provide both forward and inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def _identity(t: torch.Tensor) -> torch.Tensor:
    return t


def _rot90(t: torch.Tensor) -> torch.Tensor:
    """Rotate 90° CCW (in image coordinates: top-right corner moves to top-left)."""
    return torch.rot90(t, k=1, dims=(-2, -1))


def _rot180(t: torch.Tensor) -> torch.Tensor:
    return torch.rot90(t, k=2, dims=(-2, -1))


def _rot270(t: torch.Tensor) -> torch.Tensor:
    return torch.rot90(t, k=3, dims=(-2, -1))


def _flip_h(t: torch.Tensor) -> torch.Tensor:
    """Mirror left-right (flip across the vertical axis)."""
    return torch.flip(t, dims=(-1,))


def _flip_v(t: torch.Tensor) -> torch.Tensor:
    """Mirror top-bottom (flip across the horizontal axis)."""
    return torch.flip(t, dims=(-2,))


def _transpose(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(-2, -1).contiguous()


def _anti_transpose(t: torch.Tensor) -> torch.Tensor:
    """Flip across the anti-diagonal: (H, W) → (W, H) but reversed both ways."""
    return torch.flip(_transpose(t), dims=(-2, -1))


@dataclass(frozen=True)
class D8Element:
    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


# All 8 elements of the dihedral group D8.
D8: tuple[D8Element, ...] = (
    D8Element("identity",       _identity,       _identity),
    D8Element("rot90",          _rot90,          _rot270),
    D8Element("rot180",         _rot180,         _rot180),
    D8Element("rot270",         _rot270,         _rot90),
    D8Element("flip_h",         _flip_h,         _flip_h),
    D8Element("flip_v",         _flip_v,         _flip_v),
    D8Element("transpose",      _transpose,      _transpose),
    D8Element("anti_transpose", _anti_transpose, _anti_transpose),
)
