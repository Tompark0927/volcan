"""Loss functions for Volcan — implements the six terms in architecture.md §12.

  - L_denoise: masked-denoising / score-matching on color channels (primary)
  - L_ghost_stability: end-of-Phase-A ghost field convergence (Pillar 3)
  - L_regime: reward strongest clean periodicity from echoes (Pillar 2)
  - L_cycle: forward/backward consistency on demo pairs (Pillar 2 helper)
  - L_mdl: ghost field sparsity + transition smoothness (Pillar 4 supporting)
  - L_apoptosis: incoherent cells biased toward background (Pillar 5)

For Week 2 smoke test we ship:
  - L_denoise (full)
  - L_ghost_stability (full)
  - L_regime (full)
  - L_apoptosis (lightweight, training-time pressure)
  - L_mdl (lightweight)

L_cycle requires the reverse-mode update flag and is deferred to Week 2.5 once
the simpler losses are validated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .echo import EchoReading


# -----------------------------------------------------------------------------
# L_denoise — masked denoising / score matching on color channels
# -----------------------------------------------------------------------------


def masked_denoising_loss(
    predicted_color: torch.Tensor,
    target_index: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    masked_weight: float = 3.0,
    valid_mask: torch.Tensor | None = None,
    pad_weight: float = 0.05,
    deeply_supervised_steps: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute the denoising loss on the predicted color channels.

    The loss has TWO different mask concepts:
      - `valid_mask`: cells that are part of the actual ARC content (non-pad).
        Pad cells contribute much less (weight = `pad_weight`) so the model is
        graded on the puzzle, not on its padding-detection ability. Without
        this, the ~890 pad cells in a 30×30 padded grid drown out the ~10
        content cells and the model converges to "predict outside everywhere."
      - `mask`: per-cell denoising mask (the cell was corrupted by the noise
        schedule). These cells get `masked_weight`× the loss of unmasked cells,
        matching modern masked-diffusion LM practice.

    Both masks compose multiplicatively.
    """
    def _step_loss(logits: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target_index, reduction="none")  # (B, H, W)
        weight = torch.ones_like(ce)
        if mask is not None:
            weight = torch.where(mask, weight * masked_weight, weight)
        if valid_mask is not None:
            weight = torch.where(valid_mask, weight, weight * pad_weight)
        return (ce * weight).sum() / weight.sum().clamp_min(1.0)

    total = _step_loss(predicted_color)
    if deeply_supervised_steps:
        for s in deeply_supervised_steps:
            total = total + _step_loss(s)
        total = total / (1 + len(deeply_supervised_steps))
    return total


def corrupt_target(
    target_onehot: torch.Tensor,
    noise_level: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inhale: corrupt a target one-hot grid with masked noise.

    Each cell is independently replaced (with probability `noise_level`) by a
    uniform distribution over the V colors.

    Returns:
        corrupted: (B, V, H, W) — the noised input for Phase B.
        mask: (B, H, W) — True where the cell was corrupted (these are the
            cells the loss should focus on).
    """
    b, v, h, w = target_onehot.shape
    device = target_onehot.device
    if generator is not None:
        rand = torch.rand(b, h, w, device=device, generator=generator)
    else:
        rand = torch.rand(b, h, w, device=device)
    mask = rand < noise_level
    uniform = torch.full_like(target_onehot, 1.0 / v)
    mask_expanded = mask.unsqueeze(1).expand_as(target_onehot)
    corrupted = torch.where(mask_expanded, uniform, target_onehot)
    return corrupted, mask


# -----------------------------------------------------------------------------
# L_ghost_stability — end-of-Phase-A ghost convergence (Pillar 3)
# -----------------------------------------------------------------------------


def ghost_stability_loss(
    ghost_now: torch.Tensor,
    ghost_past: torch.Tensor,
) -> torch.Tensor:
    """L2 distance between current and past ghost states; lower = more stable."""
    return F.mse_loss(ghost_now, ghost_past)


# -----------------------------------------------------------------------------
# L_regime — reward clean periodicity (Pillar 2)
# -----------------------------------------------------------------------------


def regime_loss(
    final_echoes: EchoReading,
    chaos_threshold: float = 0.5,
    chaos_penalty: float = 0.5,
) -> torch.Tensor:
    """Reward strong clean periodicity, penalize chaos.

    Args:
        final_echoes: EchoReading at the end of Phase B.
        chaos_threshold: if max(echoes) < this, the run is "chaotic."
        chaos_penalty: extra constant penalty added when chaotic.

    Returns:
        per-batch averaged scalar loss.
    """
    best = torch.stack(
        [final_echoes.echo_1, final_echoes.echo_2, final_echoes.echo_3],
        dim=-1,
    ).max(dim=-1).values  # (B,)
    chaos = (best < chaos_threshold).float() * chaos_penalty  # (B,)
    return (-best + chaos).mean()


# -----------------------------------------------------------------------------
# L_mdl — ghost field sparsity + smoothness (Pillar 4 helper)
# -----------------------------------------------------------------------------


def mdl_loss(
    ghost_trajectory: list[torch.Tensor],
) -> torch.Tensor:
    """L1 sparsity on ghost states + L1 transition smoothness.

    Args:
        ghost_trajectory: list of (B, G, H, W) ghost tensors over Phase A.
    """
    if not ghost_trajectory:
        return torch.tensor(0.0)
    sparsity = sum(g.abs().mean() for g in ghost_trajectory) / len(ghost_trajectory)
    if len(ghost_trajectory) >= 2:
        diffs = [
            (ghost_trajectory[i] - ghost_trajectory[i - 1]).abs().mean()
            for i in range(1, len(ghost_trajectory))
        ]
        transitions = sum(diffs) / len(diffs)
    else:
        transitions = torch.tensor(0.0, device=ghost_trajectory[0].device)
    return sparsity + 0.5 * transitions


# -----------------------------------------------------------------------------
# L_apoptosis — coherence-driven background bias (Pillar 5)
# -----------------------------------------------------------------------------


def compute_vitality(
    color_now: torch.Tensor,
    color_past: torch.Tensor,
) -> torch.Tensor:
    """Per-cell vitality = how stable each cell's color distribution has been.

    Args:
        color_now, color_past: (B, V, H, W).

    Returns:
        vitality: (B, 1, H, W) in [0, 1].
    """
    # Per-cell cosine similarity between successive states.
    sim = F.cosine_similarity(color_now, color_past, dim=1, eps=1e-8)  # (B, H, W)
    # Map [-1, 1] → [0, 1] (we expect roughly [0, 1] in practice for distributions).
    return ((sim + 1) / 2).unsqueeze(1)


def apoptosis_loss(
    color_now: torch.Tensor,
    vitality: torch.Tensor,
    background_index: int = 0,
) -> torch.Tensor:
    """Encourage incoherent cells (low vitality) to commit to background color.

    Args:
        color_now: (B, V, H, W) color logits or probabilities.
        vitality: (B, 1, H, W) per-cell vitality in [0, 1].
        background_index: the color index to push toward (default 0 = black).

    Returns:
        scalar loss.
    """
    b, v, h, w = color_now.shape
    target = torch.zeros(b, v, h, w, device=color_now.device, dtype=color_now.dtype)
    target[:, background_index] = 1.0
    # Weight by (1 - vitality): only incoherent cells contribute.
    weight = (1.0 - vitality)
    diff = (color_now - target) ** 2
    return (diff * weight).mean()


def soft_apoptosis(
    color_now: torch.Tensor,
    vitality: torch.Tensor,
    pressure_scale: float,
    background_index: int = 0,
) -> torch.Tensor:
    """Inference-time apoptosis: bias the color distribution toward background
    by an amount proportional to (1 - vitality).

    This is the inference-side counterpart to `apoptosis_loss`. Returns a new
    color tensor (does not mutate input).
    """
    b, v, h, w = color_now.shape
    bg = torch.zeros(b, v, h, w, device=color_now.device, dtype=color_now.dtype)
    bg[:, background_index] = 1.0
    pressure = (1.0 - vitality) * pressure_scale  # (B, 1, H, W)
    return (1.0 - pressure) * color_now + pressure * bg


# -----------------------------------------------------------------------------
# Activity penalty — "Metabolic Cost" / Free Energy (Week 6 — CEO addition)
# -----------------------------------------------------------------------------


def activity_penalty(deltas: list[torch.Tensor]) -> torch.Tensor:
    """Sum of absolute state changes across NCA iteration steps.

    L = Σ_t || s_{t+1} − s_t ||_1

    Penalizes "unnecessary cellular activity" — biological cells don't move
    unless they have to, and the most efficient path to a fixed point is the
    one that doesn't move much. This is mathematical Occam's razor at the
    dynamics level: the simplest configuration that satisfies the loss is
    rewarded.

    Args:
        deltas: list of (B, C, H, W) tensors of per-cell state deltas at each step.

    Returns:
        scalar — average absolute delta per cell across the trajectory.
    """
    if not deltas:
        return torch.tensor(0.0)
    total = sum(d.abs().mean() for d in deltas)
    return total / len(deltas)
