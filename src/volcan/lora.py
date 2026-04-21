"""LoRA (Low-Rank Adaptation) for Volcan (Phase 1 Step 3).

LoRA freezes the base model's weights and trains only a small low-rank
"delta" on top. Forward pass: `y = W·x + B·A·x` where W is frozen and A, B
are small trainable matrices with rank r (default 16). Introduced by Hu et al.
2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)); the MIT TTT
paper (Akyürek et al. 2024) showed a +7 pp delta on ARC-1 from per-task
LoRA TTT vs full-weight TTT, because LoRA preserves the pretrained priors
instead of overwriting them on 2-3 demos.

For Volcan we only target the update MLP's 1×1 convolutions — that's where
the interesting adaptation happens. The mycelial, spectral, and Phase A
Ghost Dream mechanisms stay fully frozen during TTT.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn


class LoRAConv2dAdapter(nn.Module):
    """LoRA adapter for an nn.Conv2d with kernel_size=1 (i.e., a 1x1 conv,
    which is equivalent to nn.Linear applied per spatial location).

    The adapter computes `y = base(x) + B(A(x))` where A is (in → r) and B
    is (r → out), both 1×1 convs. B is initialized to zero so the adapter
    starts as an identity operation (no effect on the frozen base).
    """

    def __init__(self, base: nn.Conv2d, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if base.kernel_size != (1, 1):
            raise ValueError(
                f"LoRAConv2dAdapter only supports 1x1 convs, got kernel {base.kernel_size}"
            )
        self.base = base  # frozen
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_ch = base.in_channels
        out_ch = base.out_channels

        # A: (in_ch → rank), B: (rank → out_ch). Both 1x1 convs.
        self.lora_A = nn.Conv2d(in_ch, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_ch, kernel_size=1, bias=False)

        # Kaiming for A, zero for B so the adapter starts as a no-op.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


def _wrap_sequential_with_lora(
    seq: nn.Sequential, rank: int, alpha: float
) -> list[nn.Parameter]:
    """Replace every 1×1 Conv2d in a Sequential with a LoRA-wrapped version.
    Returns the list of adapter parameters."""
    lora_params: list[nn.Parameter] = []
    for i, layer in enumerate(seq):
        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
            adapter = LoRAConv2dAdapter(layer, rank=rank, alpha=alpha)
            adapter = adapter.to(next(layer.parameters()).device)
            seq[i] = adapter
            lora_params.extend(adapter.lora_parameters())
    return lora_params


def attach_lora_to_update_mlp(
    volcan_cell: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
) -> list[nn.Parameter]:
    """Wrap each 1×1 Conv2d in `volcan_cell.update` with a LoRA adapter.

    Handles both the dense Sequential update and the MoEUpdateMLP variant:
      - Dense: wrap every Conv2d in the Sequential
      - MoE:   wrap every Conv2d in every expert's Sequential (the router stays
               frozen — we don't want TTT changing which expert handles which
               pixel, only how each expert behaves)

    Freezes ALL base parameters of the cell. Returns the list of LoRA
    adapter parameters — pass these to your optimizer for TTT.
    """
    # Deferred import to avoid a circular dependency with volcan_cell.
    from .moe import MoEUpdateMLP

    # Step 1: freeze everything.
    for p in volcan_cell.parameters():
        p.requires_grad = False

    # Step 2: find and wrap.
    update = volcan_cell.update
    lora_params: list[nn.Parameter] = []
    if isinstance(update, MoEUpdateMLP):
        # Wrap each expert independently. Router stays frozen.
        for expert in update.experts:
            lora_params.extend(_wrap_sequential_with_lora(expert, rank, alpha))
    else:
        # Dense Sequential case.
        lora_params.extend(_wrap_sequential_with_lora(update, rank, alpha))
    return lora_params


def detach_lora_from_update_mlp(volcan_cell: nn.Module) -> None:
    """Undo `attach_lora_to_update_mlp` — restore the original 1×1 Convs and
    re-enable gradients on the base parameters. Used to clean up the model
    after TTT if we want to reuse the same cell for another task.
    """
    update_seq = volcan_cell.update
    for i, layer in enumerate(update_seq):
        if isinstance(layer, LoRAConv2dAdapter):
            update_seq[i] = layer.base
    for p in volcan_cell.parameters():
        p.requires_grad = True


@contextmanager
def lora_ttt(volcan_cell: nn.Module, rank: int = 16, alpha: float = 16.0):
    """Context manager: attach LoRA, yield the adapter params, detach on exit."""
    lora_params = attach_lora_to_update_mlp(volcan_cell, rank=rank, alpha=alpha)
    try:
        yield lora_params
    finally:
        detach_lora_from_update_mlp(volcan_cell)
