"""Volcan pretraining loop (Week 4).

The Week 2 `overfit_volcan_single_task` trains on a single task's demos. For
pretraining we need a fundamentally different shape:

  - Sample from a CORPUS of many tasks (5K synthetic for Week 4 first run)
  - Each batch element comes from a DIFFERENT task with a DIFFERENT rule
  - The model has to learn general priors over "what kinds of grid
    transformations are plausible" — not memorize any single rule

This module implements:
  - `MultiTaskDataset`: lightweight wrapper that samples random (input, target)
    pairs from a list of tasks, with a fresh random demo per call
  - `pretrain_volcan`: the actual training loop, with checkpointing and
    optional periodic eval

We use the same masked-denoising loss as the smoke test (single-task overfit),
but the source of each batch element is a different task, so the model sees
many different rules per batch.
"""

from __future__ import annotations

import random as _random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import optim

from .arc import OUTSIDE_TOKEN, Task, grid_to_onehot
from .echo import cosine_similarity_per_sample
from .losses import activity_penalty, corrupt_target, masked_denoising_loss
from .symmetry import D8
from .volcan_cell import VolcanCell


# -----------------------------------------------------------------------------
# Multi-task dataset
# -----------------------------------------------------------------------------


class MultiTaskDataset:
    """Samples random (input, target) pairs from a corpus of tasks.

    Each `sample_batch(N)` call returns N independent samples, each from a
    randomly chosen task and a randomly chosen demo within that task.
    """

    def __init__(self, tasks: list[Task], pad_to: int = 30, seed: int = 0):
        if not tasks:
            raise ValueError("MultiTaskDataset requires at least one task")
        self.tasks = tasks
        self.pad_to = pad_to
        self._rng = _random.Random(seed)

    def __len__(self) -> int:
        return len(self.tasks)

    def sample_one(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one (input_onehot, target_onehot, valid_mask) tuple."""
        task = self._rng.choice(self.tasks)
        ex = self._rng.choice(task.train)
        inp_onehot = grid_to_onehot(ex.input, pad_to=self.pad_to)
        tgt_onehot = grid_to_onehot(ex.output, pad_to=self.pad_to)
        # valid_mask: cells that are content in either input or target.
        inp_idx = inp_onehot.argmax(dim=0)
        tgt_idx = tgt_onehot.argmax(dim=0)
        valid = (inp_idx != OUTSIDE_TOKEN) | (tgt_idx != OUTSIDE_TOKEN)
        return inp_onehot, tgt_onehot, valid

    def sample_batch(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of `batch_size` independent training examples.

        Returns:
            inputs: (B, V, H, W) one-hot
            targets: (B, V, H, W) one-hot
            valid_mask: (B, H, W) bool
        """
        inputs, targets, masks = [], [], []
        for _ in range(batch_size):
            i, t, m = self.sample_one()
            inputs.append(i)
            targets.append(t)
            masks.append(m)
        return (
            torch.stack(inputs),
            torch.stack(targets),
            torch.stack(masks),
        )

    def sample_icl_one(
        self,
        num_demos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one ICL training example via leave-one-out from a task.

        Picks a task with at least num_demos+1 demo pairs, takes num_demos as
        ICL context, and holds out one as the query.

        Returns:
            demo_inputs:  (K, V, H, W) — K=num_demos demo input one-hots
            demo_outputs: (K, V, H, W) — corresponding outputs
            query_input:  (V, H, W)
            query_target: (V, H, W)
            valid_mask:   (H, W) bool — content cells of the query
        """
        # Find a task with enough demos.
        for _ in range(50):
            task = self._rng.choice(self.tasks)
            if len(task.train) >= num_demos + 1:
                break
        else:
            # Fallback: use whatever we got, with replacement if needed.
            task = self._rng.choice(self.tasks)

        n_avail = len(task.train)
        if n_avail >= num_demos + 1:
            indices = self._rng.sample(range(n_avail), num_demos + 1)
        else:
            # Sample with replacement; query is always the last index sampled.
            indices = [self._rng.randrange(n_avail) for _ in range(num_demos + 1)]

        demo_idx = indices[:num_demos]
        query_idx = indices[num_demos]

        demo_inputs = torch.stack(
            [grid_to_onehot(task.train[i].input, pad_to=self.pad_to) for i in demo_idx]
        )
        demo_outputs = torch.stack(
            [grid_to_onehot(task.train[i].output, pad_to=self.pad_to) for i in demo_idx]
        )
        query_in = grid_to_onehot(task.train[query_idx].input, pad_to=self.pad_to)
        query_tgt = grid_to_onehot(task.train[query_idx].output, pad_to=self.pad_to)

        q_in_idx = query_in.argmax(dim=0)
        q_tgt_idx = query_tgt.argmax(dim=0)
        valid = (q_in_idx != OUTSIDE_TOKEN) | (q_tgt_idx != OUTSIDE_TOKEN)

        return demo_inputs, demo_outputs, query_in, query_tgt, valid

    def sample_icl_batch(
        self,
        batch_size: int,
        num_demos: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of ICL training examples.

        Returns:
            demo_inputs:  (B, K, V, H, W)
            demo_outputs: (B, K, V, H, W)
            query_inputs: (B, V, H, W)
            query_targets: (B, V, H, W)
            valid_mask:   (B, H, W) bool
        """
        d_in, d_out, q_in, q_tgt, masks = [], [], [], [], []
        for _ in range(batch_size):
            di, do, qi, qt, m = self.sample_icl_one(num_demos)
            d_in.append(di)
            d_out.append(do)
            q_in.append(qi)
            q_tgt.append(qt)
            masks.append(m)
        return (
            torch.stack(d_in),
            torch.stack(d_out),
            torch.stack(q_in),
            torch.stack(q_tgt),
            torch.stack(masks),
        )


# -----------------------------------------------------------------------------
# Pretraining loop
# -----------------------------------------------------------------------------


@dataclass
class PretrainLog:
    """Per-eval metrics from a pretraining run."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    def add(self, step: int, loss: float, acc: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.accuracies.append(acc)


def pretrain_volcan(
    model: VolcanCell,
    dataset: MultiTaskDataset,
    *,
    num_steps: int = 1000,
    batch_size: int = 16,
    lr: float = 1e-3,
    phase_a_max: int = 6,
    phase_b_steps: int = 16,
    device: str = "cpu",
    log_every: int = 50,
    checkpoint_every: int | None = None,
    checkpoint_path: str | Path | None = None,
    on_log: Callable[[int, float, float, float], None] | None = None,
    noise_min: float = 0.2,
    noise_max: float = 0.95,
    seed: int = 0,
) -> PretrainLog:
    """Pretrain Volcan on a multi-task dataset via masked denoising.

    Each gradient step:
      1. Sample a batch of independent (input, target) pairs from random tasks
      2. Run Phase A on the inputs (color clamped, ghost evolves)
      3. Sample a noise level σ, corrupt the targets
      4. Run Phase B starting from the corrupted targets
      5. Compute masked denoising loss + light regularizers
      6. Backprop, step
    """
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    log = PretrainLog()
    rng_noise = torch.Generator(device="cpu")
    rng_noise.manual_seed(seed)

    running_loss = 0.0
    running_acc = 0.0
    running_count = 0
    t0 = time.time()

    for step in range(1, num_steps + 1):
        inputs, targets_oh, valid_mask = dataset.sample_batch(batch_size)
        inputs = inputs.to(device)
        targets_oh = targets_oh.to(device)
        valid_mask = valid_mask.to(device)
        targets_idx = targets_oh.argmax(dim=1)

        # Phase A: Ghost Dream.
        ghost_state, forces, object_field, _ = model.phase_a(
            inputs, max_steps=phase_a_max, min_steps=min(8, phase_a_max)
        )

        # Phase B: corrupt the target, denoise from there.
        noise_level = float(
            torch.rand(1, generator=rng_noise).item() * (noise_max - noise_min)
            + noise_min
        )
        # Start color at the input + masked corruption (matches Week 2 fix).
        masked_init, mask = corrupt_target(inputs.clone(), noise_level=noise_level)

        final_state, _, _, _ = model.phase_b(
            ghost_state, forces,
            object_field=object_field,
            init_color=masked_init, max_steps=phase_b_steps,
        )
        final_color = model.color_logits(final_state)

        loss = masked_denoising_loss(
            final_color,
            targets_idx,
            mask=mask,
            valid_mask=valid_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = final_color.argmax(dim=1)
            content_correct = ((pred == targets_idx) & valid_mask).sum().item()
            content_total = valid_mask.sum().item()
            acc = content_correct / max(content_total, 1)

        running_loss += loss.item()
        running_acc += acc
        running_count += 1

        if step == 1 or step % log_every == 0 or step == num_steps:
            avg_loss = running_loss / running_count
            avg_acc = running_acc / running_count
            elapsed = time.time() - t0
            log.add(step, avg_loss, avg_acc)
            if on_log is not None:
                on_log(step, avg_loss, avg_acc, elapsed)
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0

        if (
            checkpoint_every
            and checkpoint_path
            and step % checkpoint_every == 0
        ):
            _save_checkpoint(model, checkpoint_path, step)

    if checkpoint_path:
        _save_checkpoint(model, checkpoint_path, num_steps)

    return log


def pretrain_volcan_icl(
    model: VolcanCell,
    dataset: MultiTaskDataset,
    *,
    num_steps: int = 1000,
    batch_size: int = 8,
    num_demos: int = 3,
    lr: float = 1e-3,
    icl_steps_per_clamp: int = 4,
    phase_b_steps: int = 16,
    device: str = "cpu",
    log_every: int = 50,
    checkpoint_every: int | None = None,
    checkpoint_path: str | Path | None = None,
    on_log: Callable[[int, float, float, float], None] | None = None,
    noise_min: float = 0.2,
    noise_max: float = 0.95,
    seed: int = 0,
    lambda_activity: float = 0.01,
) -> PretrainLog:
    """Pretrain Volcan with in-context learning (Week 5).

    Each gradient step:
      1. Sample a batch of (K demos, query) tuples — each from a different task
      2. Run Phase A (ICL variant) on the demos + query (sequential clamping)
      3. Sample a noise level σ, corrupt the query target
      4. Run Phase B starting from the corrupted target
      5. Compute masked-denoising loss on the query target
      6. Backprop, step

    The model has to learn how to use the demo context to predict the query
    output — the multi-task incoherence problem from Week 4 is gone because
    each batch element now has self-consistent demo information.
    """
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    log = PretrainLog()
    rng_noise = torch.Generator(device="cpu")
    rng_noise.manual_seed(seed)

    running_loss = 0.0
    running_acc = 0.0
    running_count = 0
    t0 = time.time()

    for step in range(1, num_steps + 1):
        d_in, d_out, q_in, q_tgt, valid_mask = dataset.sample_icl_batch(
            batch_size, num_demos=num_demos
        )
        d_in = d_in.to(device)
        d_out = d_out.to(device)
        q_in = q_in.to(device)
        q_tgt = q_tgt.to(device)
        valid_mask = valid_mask.to(device)
        q_tgt_idx = q_tgt.argmax(dim=1)

        # ICL Phase A: encode demos, then query input.
        ghost_state, forces, object_field, _ = model.phase_a_icl(
            d_in, d_out, q_in, steps_per_clamp=icl_steps_per_clamp
        )

        # Phase B: corrupt query target, denoise from there.
        noise_level = float(
            torch.rand(1, generator=rng_noise).item() * (noise_max - noise_min)
            + noise_min
        )
        masked_init, mask = corrupt_target(q_in.clone(), noise_level=noise_level)

        final_state, _, _, deltas = model.phase_b(
            ghost_state,
            forces,
            object_field=object_field,
            init_color=masked_init,
            max_steps=phase_b_steps,
            collect_activity=(lambda_activity > 0),
        )
        final_color = model.color_logits(final_state)

        loss_denoise = masked_denoising_loss(
            final_color,
            q_tgt_idx,
            mask=mask,
            valid_mask=valid_mask,
        )
        if lambda_activity > 0 and deltas:
            loss_activity = activity_penalty(deltas)
            loss = loss_denoise + lambda_activity * loss_activity
        else:
            loss = loss_denoise

        # Phase 2 Path A: Load-balancing auxiliary loss for MoE (if present).
        # Prevents expert collapse (one expert always wins the routing). Only
        # adds a term if the model has an MoE update; otherwise no-op.
        from .moe import MoEUpdateMLP as _MoE
        if isinstance(model.update, _MoE):
            lb_loss = model.update.load_balancing_loss()
            loss = loss + 0.01 * lb_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = final_color.argmax(dim=1)
            content_correct = ((pred == q_tgt_idx) & valid_mask).sum().item()
            content_total = valid_mask.sum().item()
            acc = content_correct / max(content_total, 1)

        running_loss += loss.item()
        running_acc += acc
        running_count += 1

        if step == 1 or step % log_every == 0 or step == num_steps:
            avg_loss = running_loss / running_count
            avg_acc = running_acc / running_count
            elapsed = time.time() - t0
            log.add(step, avg_loss, avg_acc)
            if on_log is not None:
                on_log(step, avg_loss, avg_acc, elapsed)
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0

        if (
            checkpoint_every
            and checkpoint_path
            and step % checkpoint_every == 0
        ):
            _save_checkpoint(model, checkpoint_path, step)

    if checkpoint_path:
        _save_checkpoint(model, checkpoint_path, num_steps)

    return log


@torch.no_grad()
def predict_volcan_icl(
    model: VolcanCell,
    demo_inputs: list,
    demo_outputs: list,
    query_input: torch.Tensor,
    *,
    pad_to: int = 30,
    icl_steps_per_clamp: int = 4,
    phase_b_steps: int = 16,
    device: str = "cpu",
) -> torch.Tensor:
    """Run ICL inference: feed demos + query to the model and predict the query output.

    Args:
        model: VolcanCell.
        demo_inputs: list of Grids — the demo input grids (raw lists of ints).
        demo_outputs: list of Grids — corresponding output grids.
        query_input: (V, H, W) one-hot tensor for the query input.

    Returns:
        color logits of shape (V, H, W) for the predicted query output.
    """
    model.eval()
    if query_input.ndim == 3:
        query_input_b = query_input.unsqueeze(0).to(device)
    else:
        query_input_b = query_input.to(device)

    d_in_t = torch.stack(
        [grid_to_onehot(g, pad_to=pad_to) for g in demo_inputs]
    ).unsqueeze(0).to(device)
    d_out_t = torch.stack(
        [grid_to_onehot(g, pad_to=pad_to) for g in demo_outputs]
    ).unsqueeze(0).to(device)

    ghost_state, forces, object_field, _ = model.phase_a_icl(
        d_in_t, d_out_t, query_input_b, steps_per_clamp=icl_steps_per_clamp
    )
    # At inference, start Phase B from the query input itself (no noise).
    final_state, _, _, _ = model.phase_b(
        ghost_state, forces,
        object_field=object_field,
        init_color=query_input_b, max_steps=phase_b_steps,
    )
    color = model.color_logits(final_state)
    return color.squeeze(0)


@torch.no_grad()
def predict_volcan_d8_ensemble(
    model: VolcanCell,
    demo_inputs: list,
    demo_outputs: list,
    query_input: torch.Tensor,
    *,
    pad_to: int = 30,
    icl_steps_per_clamp: int = 4,
    phase_b_steps: int = 16,
    device: str = "cpu",
    top_k: int = 2,
) -> list[torch.Tensor]:
    """D8-Ensemble Resonance inference (Week 6, CEO's Pillar 2 idea).

    For each of the 8 D8 symmetries:
      1. Apply the symmetry to the demos and query
      2. Run Phase A (ICL) — the model encodes the rule in this orientation
      3. Measure echo_1 of the final Phase A state vs the previous step
         (the "stability" / "confidence" of the rule encoding)
      4. Run Phase B in this orientation
      5. Apply the inverse symmetry to the predicted output

    Then sort all 8 predictions by their Phase A stability (echo_1) and return
    the top_k most-stable predictions, un-rotated to the original frame.

    The intuition: if the rule is "rotate the shape by 90°", then running the
    model on the original orientation will be unstable (the model has to figure
    out what rotation to apply), but running on the *already-rotated* version
    will be stable (the rule degenerates to identity in the rotated frame).
    The most-stable orientation is where the rule "snaps into place."

    Returns:
        list of (V, H, W) color logit tensors, in the original frame, sorted by
        Phase A confidence (most confident first). Length = top_k.
    """
    model.eval()

    # Convert all inputs to padded one-hot once.
    if query_input.ndim == 3:
        query_input_oh = query_input.unsqueeze(0)
    else:
        query_input_oh = query_input
    query_input_oh = query_input_oh.to(device)

    d_in_t = torch.stack(
        [grid_to_onehot(g, pad_to=pad_to) for g in demo_inputs]
    ).unsqueeze(0).to(device)
    d_out_t = torch.stack(
        [grid_to_onehot(g, pad_to=pad_to) for g in demo_outputs]
    ).unsqueeze(0).to(device)

    candidates: list[tuple[float, torch.Tensor, str]] = []

    for elem in D8:
        # Apply the symmetry to all (B, K, V, H, W) and (B, V, H, W) tensors.
        # Note: D8 ops act on the last two dims (H, W), so they work fine on
        # both 4D and 5D tensors.
        d_in_s = elem.forward(d_in_t)
        d_out_s = elem.forward(d_out_t)
        q_in_s = elem.forward(query_input_oh)

        # Phase A in this orientation, capturing the final two states.
        # We need both to compute echo_1 = cos_sim(s_t, s_{t-1}).
        # phase_a_icl returns the final state but we want stability of the
        # ghost field at the end. Run twice with one extra step to compute echo.
        ghost_state, forces, object_field, _ = model.phase_a_icl(
            d_in_s, d_out_s, q_in_s, steps_per_clamp=icl_steps_per_clamp
        )
        # Stability check: run one extra clamp on the query and measure the
        # difference. A "stable" encoding should change very little.
        ghost_now = model.ghost(ghost_state).clone()
        next_state, _, _ = model.step(
            ghost_state, forces, object_field,
            clamp_color=True, clamped_color=q_in_s,
        )
        ghost_next = model.ghost(next_state)
        stability = float(
            cosine_similarity_per_sample(ghost_now, ghost_next).mean().item()
        )

        # Phase B in this orientation.
        final_state, _, _, _ = model.phase_b(
            ghost_state, forces,
            object_field=object_field,
            init_color=q_in_s, max_steps=phase_b_steps,
        )
        color_s = model.color_logits(final_state)  # (1, V, H, W) in symmetric frame

        # Inverse-rotate back to the original frame.
        color_orig = elem.inverse(color_s)  # (1, V, H, W)

        candidates.append((stability, color_orig.squeeze(0), elem.name))

    # Sort by stability descending — most-confident encoding first.
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [c[1] for c in candidates[:top_k]]


def _save_checkpoint(model: VolcanCell, path: str | Path, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": model.state_dict(),
            "config": model.cfg.__dict__,
        },
        path,
    )


def load_checkpoint(model: VolcanCell, path: str | Path, device: str = "cpu") -> int:
    """Load a checkpoint into `model`. Returns the step number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("step", 0)
