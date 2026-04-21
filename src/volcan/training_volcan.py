"""Volcan training loop with two-phase iteration and masked denoising.

Week 2 ships `overfit_volcan_single_task` — the smoke-test counterpart to
Week 1's `overfit_single_task`. It trains a `VolcanCell` to overfit one ARC
task using:

  - Phase A (Ghost Dream) on each demo input
  - Phase B (Crystallization) starting from a noisy color initialization
  - Masked denoising loss on the final color
  - Light ghost stability + regime + apoptosis regularizers

Goal: prove all five pillars compose into something trainable end-to-end on
real ARC data, even if the architecture is many times bigger and slower than
BasicNCA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import optim

from .arc import Example, OUTSIDE_TOKEN, VOCAB_SIZE, Task, grid_to_onehot
from .hyperttt import LoRASchema, attach_hypernet_lora
from .lora import attach_lora_to_update_mlp, detach_lora_from_update_mlp
from .losses import (
    apoptosis_loss,
    compute_vitality,
    corrupt_target,
    ghost_stability_loss,
    masked_denoising_loss,
    mdl_loss,
    regime_loss,
)
from .symmetry import D8
from .volcan_cell import VolcanCell


@dataclass
class VolcanTrainLog:
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    denoise_losses: list[float] = field(default_factory=list)
    regime_losses: list[float] = field(default_factory=list)

    def add(
        self,
        step: int,
        loss: float,
        accuracy: float,
        denoise: float,
        regime: float,
    ) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.denoise_losses.append(denoise)
        self.regime_losses.append(regime)


def _augment_task_d8(task: Task) -> Task:
    """Expand a task's demos with all 8 D8 symmetries.

    Each original demo (input, output) generates 8 augmented pairs by applying
    the same symmetry to both input and output. This teaches the model that
    rules are equivariant under rotation/reflection — a critical inductive bias
    that the MIT TTT paper showed roughly DOUBLES accuracy vs. unaugmented TTT.

    A task with 4 demos becomes 32 augmented demos (4 × 8).
    """
    aug_train: list[Example] = []
    for ex in task.train:
        inp_t = torch.tensor(ex.input, dtype=torch.long)
        out_t = torch.tensor(ex.output, dtype=torch.long)
        for elem in D8:
            inp_aug = elem.forward(inp_t).tolist()
            out_aug = elem.forward(out_t).tolist()
            aug_train.append(Example(input=inp_aug, output=out_aug))
    return Task(task_id=task.task_id + "_d8", train=aug_train, test=task.test)


def _stack_demos(
    task: Task,
    pad_to: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a task's demo pairs into (input_onehot, target_onehot, target_idx)."""
    inputs = torch.stack([grid_to_onehot(ex.input, pad_to=pad_to) for ex in task.train])
    targets_oh = torch.stack(
        [grid_to_onehot(ex.output, pad_to=pad_to) for ex in task.train]
    )
    targets_idx = targets_oh.argmax(dim=1)
    return inputs, targets_oh, targets_idx


def overfit_volcan_single_task(
    model: VolcanCell,
    task: Task,
    *,
    num_steps: int = 400,
    phase_a_max: int = 24,
    phase_a_min: int = 8,
    phase_b_steps: int = 24,
    lr: float = 2e-3,
    device: str = "cpu",
    log_every: int = 25,
    pad_to: int = 30,
    on_log: Callable[[int, float, float, float, float], None] | None = None,
    lambda_stab: float = 0.05,
    lambda_regime: float = 0.02,
    lambda_mdl: float = 0.005,
    lambda_apop: float = 0.005,
    noise_min: float = 0.2,
    noise_max: float = 0.95,
    d8_augment: bool = False,
    loo_validation: bool = False,
    loo_check_every: int = 25,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_lr_multiplier: float = 5.0,
    hypernet_lora_flat: torch.Tensor | None = None,
    hypernet_lora_schema: LoRASchema | None = None,
) -> VolcanTrainLog:
    """Train Volcan to overfit a single task's demo pairs.

    Each gradient step:
      1. Run Phase A on each demo input (with the input color clamped).
      2. Sample a noise level σ ~ Uniform[noise_min, noise_max].
      3. Corrupt the demo target with that noise level.
      4. Run Phase B starting from the corrupted target.
      5. Compute the masked-denoising loss + regularizers.
      6. Backprop, step optimizer.

    If d8_augment is True, the task's demos are expanded with all 8 D8
    symmetries before training.

    If loo_validation is True AND the task has ≥ 3 demos, one demo is held
    out before training. Every `loo_check_every` steps, the model is tested
    on the held-out demo. If it predicts correctly → early stop (the model
    has learned the rule, not just memorized the demos). This is the MIT
    TTT paper's second key trick.
    """
    model = model.to(device)
    model.train()

    # --- LOO setup: hold out one demo for validation ---
    val_demo = None
    train_task = task
    if loo_validation and task.num_train >= 3:
        import random as _rng
        val_idx = _rng.randint(0, task.num_train - 1)
        val_demo = task.train[val_idx]
        train_demos = [ex for i, ex in enumerate(task.train) if i != val_idx]
        train_task = Task(task_id=task.task_id, train=train_demos, test=task.test)

    if d8_augment:
        train_task = _augment_task_d8(train_task)

    inputs, targets_oh, targets_idx = _stack_demos(train_task, pad_to=pad_to)
    inputs = inputs.to(device)
    targets_oh = targets_oh.to(device)
    targets_idx = targets_idx.to(device)

    # --- LoRA setup: freeze base params, attach low-rank adapters ---
    # With LoRA, only the adapter weights train; pretrained priors stay intact.
    # MIT TTT showed +7 pp on ARC-1 from per-task LoRA vs full-weight TTT.
    lora_attached = False
    if use_lora:
        if hypernet_lora_flat is not None and hypernet_lora_schema is not None:
            # Hyper-TTT: initialize LoRA with a hypernet's prediction instead of Kaiming.
            lora_params = attach_hypernet_lora(
                model, hypernet_lora_flat, hypernet_lora_schema, alpha=lora_alpha
            )
        else:
            lora_params = attach_lora_to_update_mlp(model, rank=lora_rank, alpha=lora_alpha)
        lora_attached = True
        # LoRA LR should be higher than pretraining LR because we're adapting a
        # small subspace (~1-3% of the base params).
        trainable_params = lora_params
        effective_lr = lr * lora_lr_multiplier
    else:
        trainable_params = list(model.parameters())
        effective_lr = lr

    # Build the "valid" mask: cells that are content in either the input or
    # the target. Pad cells (where both are OUTSIDE) get downweighted in the
    # loss so they don't drown out the actual ARC content.
    inputs_idx = inputs.argmax(dim=1)  # (N, H, W)
    valid_mask = (inputs_idx != OUTSIDE_TOKEN) | (targets_idx != OUTSIDE_TOKEN)

    optimizer = optim.Adam(trainable_params, lr=effective_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    log = VolcanTrainLog()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    for step in range(1, num_steps + 1):
        # ---------- Phase A: Ghost Dream (color clamped to input) ----------
        ghost_state, forces, object_field, _steps_a = model.phase_a(
            inputs,
            max_steps=phase_a_max,
            min_steps=phase_a_min,
        )
        ghost_now = model.ghost(ghost_state)

        # ---------- Phase B: Crystallization (color denoising) ----------
        # Masked-diffusion training schedule:
        #   1. Start the color from the demo INPUT (this is what inference uses).
        #   2. Sample a noise level σ.
        #   3. Replace fraction σ of cells with uniform — this gives us a mix of
        #      "preserve the input where you should" and "denoise the rest toward
        #      the target." The unmasked init = input, the masked init = noise.
        #   4. Run Phase B; loss is on ALL cells (with masked cells weighted higher).
        noise_level = float(
            torch.rand(1, generator=rng).item() * (noise_max - noise_min) + noise_min
        )
        # Start from the input — this is the conditioning, it shouldn't be discarded.
        init_color = inputs.clone()
        # Then mask a fraction of cells to uniform (the noise the model has to denoise).
        masked_uniform, mask = corrupt_target(init_color, noise_level=noise_level)

        final_state, _final_forces, echoes, _activity = model.phase_b(
            ghost_state,
            forces,
            object_field=object_field,
            init_color=masked_uniform,
            max_steps=phase_b_steps,
        )
        final_color = model.color_logits(final_state)

        # ---------- Losses ----------
        L_denoise = masked_denoising_loss(
            final_color,
            targets_idx,
            mask=mask,
            valid_mask=valid_mask,
        )

        # Ghost stability: compare end-of-Phase-A ghost to a slightly earlier ghost.
        # We don't keep the full Phase A trajectory in this loop, so we use a
        # cheap proxy: the ghost should have low magnitude variance across cells.
        # (The Phase A early-termination already enforces echo_1 high.)
        L_stab = ghost_now.std(dim=(2, 3)).mean()

        L_regime = regime_loss(echoes[-1])

        # MDL: small L1 on the ghost.
        L_mdl = ghost_now.abs().mean()

        # Apoptosis: vitality from last two color states in the Phase B trajectory.
        if len(echoes) >= 2:
            # We didn't store states; recompute vitality from final color vs background bias.
            # Cheap proxy: penalize final-state cells whose color distribution is uniform.
            vitality = (1.0 - final_color.std(dim=1, keepdim=True))
            L_apop = apoptosis_loss(final_color, vitality)
        else:
            L_apop = torch.tensor(0.0, device=final_color.device)

        loss = (
            L_denoise
            + lambda_stab * L_stab
            + lambda_regime * L_regime
            + lambda_mdl * L_mdl
            + lambda_apop * L_apop
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---------- Metrics ----------
        with torch.no_grad():
            pred = final_color.argmax(dim=1)
            # Accuracy on content cells only — otherwise 99% comes from padding.
            content_correct = ((pred == targets_idx) & valid_mask).sum().item()
            content_total = valid_mask.sum().item()
            accuracy = content_correct / max(content_total, 1)

        if step == 1 or step % log_every == 0 or step == num_steps:
            log.add(
                step,
                loss.item(),
                accuracy,
                L_denoise.item(),
                L_regime.item(),
            )
            if on_log is not None:
                on_log(step, loss.item(), accuracy, L_denoise.item(), L_regime.item())

        # --- LOO early stopping: check if held-out demo is predicted correctly ---
        if (
            val_demo is not None
            and step % loo_check_every == 0
            and step >= loo_check_every * 2  # skip the first check (model is cold)
        ):
            model.eval()
            with torch.no_grad():
                from .arc import grids_equal, onehot_to_grid

                val_inp = grid_to_onehot(val_demo.input, pad_to=pad_to).to(device)
                val_pred = predict_volcan(
                    model, val_inp,
                    phase_a_max=phase_a_max,
                    phase_b_steps=phase_b_steps,
                )
                val_grid = onehot_to_grid(val_pred.cpu())
                if grids_equal(val_grid, val_demo.output):
                    # The model can predict the held-out demo → it learned the
                    # rule, not just memorized the training demos. Stop early.
                    break
            model.train()

    return log


@torch.no_grad()
def predict_volcan(
    model: VolcanCell,
    input_onehot: torch.Tensor,
    *,
    phase_a_max: int = 24,
    phase_b_steps: int = 24,
) -> torch.Tensor:
    """Run the full Phase A + Phase B pipeline and return color logits.

    At inference there is no noise; Phase B starts from the input itself
    (matching the σ=0 endpoint of the training noise schedule).
    """
    model.eval()
    squeeze = False
    if input_onehot.ndim == 3:
        input_onehot = input_onehot.unsqueeze(0)
        squeeze = True

    ghost_state, forces, object_field, _ = model.phase_a(input_onehot, max_steps=phase_a_max)
    final_state, _, _, _ = model.phase_b(
        ghost_state, forces,
        object_field=object_field,
        init_color=input_onehot, max_steps=phase_b_steps,
    )
    color = model.color_logits(final_state)
    return color.squeeze(0) if squeeze else color
