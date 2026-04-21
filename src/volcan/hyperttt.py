"""Hyper-TTT: HyperNetwork for LoRA meta-initialization.

The CEO directive after six ceiling-confirming experiments: instead of starting
every TTT with random LoRA weights, train a HyperNetwork that predicts "good"
LoRA init directly from the task's demos. TTT then starts ~90% of the way to
the solution instead of from scratch.

Design:
  - **Encoder.** Use the base pretrained VolcanCell itself as the feature
    extractor. Run Phase A ICL on the 3 demo pairs → get the ghost state
    (which already encodes "the rule" from pretraining). Pool spatially to a
    fixed-size task embedding.
  - **Decoder.** A small MLP maps the task embedding → a flat vector of
    LoRA A/B matrix entries, one pair per 1×1 conv in the update MLP.
  - **Training.** Two-stage procedure:
      1. Stage 1 ("target collection"): for each task in a held-out corpus
         (dream325, NOT dream_wide — we want distribution-shifted training),
         run the standard D8+LoRA TTT. Save the final LoRA weights as targets.
      2. Stage 2 ("regression"): supervised MSE loss between the HyperNet's
         predicted LoRA weights and the Stage-1 targets.
  - **Inference.** Given a new task's demos, run the HyperNet to get LoRA init,
    attach those weights as the initial adapter, continue TTT for 150 steps,
    predict.

This is the "cheap-first" version (Option D from the design notes). A full
MAML/second-order version would also train the base weights to be easy-to-adapt;
this version just learns a good init distribution. Both are valid, the former
is higher-risk higher-reward. We ship the former.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRAConv2dAdapter
from .moe import MoEUpdateMLP
from .volcan_cell import VolcanCell


# -----------------------------------------------------------------------------
# LoRA weight schema — what the HyperNet predicts
# -----------------------------------------------------------------------------


@dataclass
class LoRASchema:
    """Shape spec for the LoRA weights attached to an update MLP.

    For a Sequential of 1×1 Conv2d layers, each with (in_ch, out_ch), we attach
    a LoRA adapter with matrices A: (in_ch → rank) and B: (rank → out_ch).
    """

    # Per conv: (in_channels, out_channels). Ordered by position in the Sequential.
    conv_shapes: list[tuple[int, int]]
    rank: int

    @property
    def total_params(self) -> int:
        """Total scalar count of all LoRA A+B matrices across all convs."""
        n = 0
        for in_ch, out_ch in self.conv_shapes:
            n += in_ch * self.rank + self.rank * out_ch
        return n

    def slice_spec(self) -> list[tuple[int, int, int, int]]:
        """Return per-layer (a_start, a_end, b_start, b_end) offsets in a flat
        concatenated tensor of all A matrices followed by all B matrices."""
        specs = []
        offset = 0
        for in_ch, out_ch in self.conv_shapes:
            a_start = offset
            a_end = a_start + in_ch * self.rank
            b_start = a_end
            b_end = b_start + self.rank * out_ch
            specs.append((a_start, a_end, b_start, b_end))
            offset = b_end
        return specs


def infer_lora_schema(model: VolcanCell, rank: int = 16) -> LoRASchema:
    """Walk the update MLP and extract (in_ch, out_ch) for each 1×1 Conv2d.

    For MoE models we only walk the FIRST expert and use its shapes — all
    experts have identical shapes by construction.
    """
    update = model.update
    shapes: list[tuple[int, int]] = []

    if isinstance(update, MoEUpdateMLP):
        # Experts share shape; use expert 0 as the template.
        target = update.experts[0]
    else:
        target = update

    for layer in target:
        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
            shapes.append((layer.in_channels, layer.out_channels))

    return LoRASchema(conv_shapes=shapes, rank=rank)


# -----------------------------------------------------------------------------
# HyperNetwork — demos → LoRA weights
# -----------------------------------------------------------------------------


class HyperNetwork(nn.Module):
    """Maps (demos, query input) → LoRA weights for TTT initialization.

    The encoder reuses the base VolcanCell: running Phase A ICL produces a
    ghost state that encodes "what the rule is" given the demos. We pool
    spatially to get a fixed-size task embedding, then decode via a small MLP
    to the flat LoRA parameter vector.
    """

    def __init__(
        self,
        base_model: VolcanCell,
        schema: LoRASchema,
        *,
        task_embed_dim: int = 64,
        decoder_hidden: int = 256,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.schema = schema
        self.task_embed_dim = task_embed_dim

        if freeze_base:
            # The base Volcan stays frozen — HyperNetwork only trains its own
            # encoder-pool + decoder. This is critical: we don't want HyperNet
            # training to contaminate the pretrained base weights.
            for p in self.base.parameters():
                p.requires_grad = False

        # Encoder head: the base Volcan's ghost state is (B, 32, 30, 30).
        # Pool to (B, task_embed_dim) via adaptive avg pool + a linear head.
        # Output must divide 30 for MPS AdaptiveAvgPool2d (5: 30/5=6).
        ghost_dim = self.base.cfg.ghost_channels  # 32
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ghost_dim * 25, task_embed_dim * 2),
            nn.GELU(),
            nn.Linear(task_embed_dim * 2, task_embed_dim),
        )

        # Decoder: task_embed → flat LoRA parameters.
        self.decoder = nn.Sequential(
            nn.Linear(task_embed_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, schema.total_params),
        )
        # Initialize last layer with zero so HyperNet starts as identity
        # (returns zeros = standard LoRA init).
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @torch.no_grad()
    def _compute_task_embedding(
        self,
        demo_inputs: torch.Tensor,  # (B, K, V, H, W)
        demo_outputs: torch.Tensor,  # (B, K, V, H, W)
        query_input: torch.Tensor,  # (B, V, H, W)
        *,
        icl_steps_per_clamp: int = 4,
    ) -> torch.Tensor:
        """Run Phase A ICL on the base model, return pooled ghost state (no grad)."""
        self.base.eval()
        state, _forces, _obj, _steps = self.base.phase_a_icl(
            demo_inputs, demo_outputs, query_input,
            steps_per_clamp=icl_steps_per_clamp,
        )
        ghost = self.base.ghost(state)  # (B, 32, H, W)
        return ghost

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        query_input: torch.Tensor,
        *,
        icl_steps_per_clamp: int = 4,
    ) -> torch.Tensor:
        """Predict a flat LoRA parameter vector of shape (B, schema.total_params)."""
        # Compute task embedding from the base model's ghost state (frozen base).
        ghost = self._compute_task_embedding(
            demo_inputs, demo_outputs, query_input,
            icl_steps_per_clamp=icl_steps_per_clamp,
        )
        pooled = self.pool(ghost)  # (B, 32, 4, 4)
        task_embed = self.encoder_head(pooled)  # (B, task_embed_dim)
        flat_lora = self.decoder(task_embed)  # (B, schema.total_params)
        return flat_lora

    def num_params(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )


# -----------------------------------------------------------------------------
# Apply predicted LoRA weights to a VolcanCell's update MLP
# -----------------------------------------------------------------------------


def flat_lora_to_adapters(
    flat_lora: torch.Tensor,
    schema: LoRASchema,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Unpack a flat LoRA vector into per-layer (A_weight, B_weight) tuples.

    Args:
        flat_lora: (schema.total_params,) — a single task's LoRA prediction.
                   (Not batched — call this per task at inference time.)
        schema: the LoRASchema used to predict it.

    Returns:
        List of (A_weight, B_weight) where:
          A_weight has shape (rank, in_ch, 1, 1) — ready to assign to a Conv2d
          B_weight has shape (out_ch, rank, 1, 1)
    """
    if flat_lora.ndim != 1:
        raise ValueError(f"expected 1D flat_lora, got shape {tuple(flat_lora.shape)}")

    specs = schema.slice_spec()
    results = []
    for (in_ch, out_ch), (a_s, a_e, b_s, b_e) in zip(schema.conv_shapes, specs):
        a_flat = flat_lora[a_s:a_e]
        b_flat = flat_lora[b_s:b_e]
        a = a_flat.reshape(schema.rank, in_ch, 1, 1)
        b = b_flat.reshape(out_ch, schema.rank, 1, 1)
        results.append((a, b))
    return results


def attach_hypernet_lora(
    volcan_cell: VolcanCell,
    hypernet_output: torch.Tensor,
    schema: LoRASchema,
    *,
    alpha: float | None = None,
) -> list[nn.Parameter]:
    """Attach LoRA adapters initialized from a HyperNetwork prediction.

    Works like `attach_lora_to_update_mlp` but uses the hypernet's predicted
    weights instead of Kaiming init. Returns the trainable LoRA parameters.
    """
    if alpha is None:
        alpha = float(schema.rank)

    # Freeze base
    for p in volcan_cell.parameters():
        p.requires_grad = False

    per_layer = flat_lora_to_adapters(hypernet_output, schema)

    update = volcan_cell.update
    lora_params: list[nn.Parameter] = []

    def _wrap(seq: nn.Sequential, adapters: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        a_idx = 0
        for i, layer in enumerate(seq):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
                if a_idx >= len(adapters):
                    return
                A_w, B_w = adapters[a_idx]
                adapter = LoRAConv2dAdapter(layer, rank=schema.rank, alpha=alpha)
                adapter = adapter.to(next(layer.parameters()).device)
                # Overwrite the default Kaiming init with the hypernet's prediction.
                with torch.no_grad():
                    adapter.lora_A.weight.copy_(A_w.to(adapter.lora_A.weight.device))
                    adapter.lora_B.weight.copy_(B_w.to(adapter.lora_B.weight.device))
                seq[i] = adapter
                lora_params.extend(adapter.lora_parameters())
                a_idx += 1

    if isinstance(update, MoEUpdateMLP):
        # For MoE, we wrap EVERY expert with the SAME predicted LoRA weights.
        # (One hypernet prediction → all experts. A more expressive variant
        # would predict per-expert LoRA, but multiplies the hypernet output
        # dim by num_experts.)
        for expert in update.experts:
            _wrap(expert, per_layer)
    else:
        _wrap(update, per_layer)

    return lora_params
