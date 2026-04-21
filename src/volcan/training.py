"""Training loops for Volcan.

Week 1 ships a single entry point: `overfit_single_task`. This is the
sanity-check smoke test — train a `BasicNCA` to overfit the demo pairs of one
ARC task. If loss goes down and we can recover the target output from the
input, the stack (data loader + model + training + viz) is working end-to-end.

Nothing here is the "real" Volcan training loop; that comes in Week 2-3 with
masked denoising, two-phase iteration, echo losses, and apoptosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import optim

from .arc import Task, VOCAB_SIZE, grid_to_onehot
from .models import BasicNCA


@dataclass
class TrainLog:
    """Per-step training metrics."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    def add(self, step: int, loss: float, accuracy: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.accuracies.append(accuracy)


def _stack_demos(
    task: Task,
    pad_to: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a task's demo pairs into (inputs, targets) tensors.

    Returns:
        inputs:  (N, VOCAB_SIZE, pad_to, pad_to) one-hot
        targets: (N, pad_to, pad_to) long, values in [0, VOCAB_SIZE)
    """
    inputs = torch.stack(
        [grid_to_onehot(ex.input, pad_to=pad_to) for ex in task.train]
    )
    targets = torch.stack(
        [grid_to_onehot(ex.output, pad_to=pad_to).argmax(dim=0) for ex in task.train]
    )
    return inputs, targets


def overfit_single_task(
    model: BasicNCA,
    task: Task,
    *,
    num_steps: int = 500,
    nca_steps: int = 32,
    lr: float = 2e-3,
    device: str = "cpu",
    log_every: int = 25,
    pad_to: int = 30,
    on_log: Callable[[int, float, float], None] | None = None,
) -> TrainLog:
    """Train the model to overfit a single task's demo pairs.

    This is the Week 1 sanity check. Success criterion: loss goes monotonically
    down and cell accuracy reaches ~100% on the demo pairs within a few hundred
    gradient steps.
    """
    model = model.to(device)
    model.train()

    inputs, targets = _stack_demos(task, pad_to=pad_to)
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    log = TrainLog()

    for step in range(1, num_steps + 1):
        state = model.init_state(inputs)
        state = model(state, steps=nca_steps)
        logits = model.color_logits(state)  # (N, VOCAB_SIZE, H, W)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: BPTT through 32 steps can spike.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accuracy = (pred == targets).float().mean().item()

        if step == 1 or step % log_every == 0 or step == num_steps:
            log.add(step, loss.item(), accuracy)
            if on_log is not None:
                on_log(step, loss.item(), accuracy)

    return log


@torch.no_grad()
def predict(
    model: BasicNCA,
    input_onehot: torch.Tensor,
    nca_steps: int = 32,
) -> torch.Tensor:
    """Run the model on a single or batched input, return color logits.

    Input:  (B, VOCAB_SIZE, H, W) or (VOCAB_SIZE, H, W)
    Output: (B, VOCAB_SIZE, H, W) or (VOCAB_SIZE, H, W)
    """
    model.eval()
    squeeze = False
    if input_onehot.ndim == 3:
        input_onehot = input_onehot.unsqueeze(0)
        squeeze = True
    state = model.init_state(input_onehot)
    state = model(state, steps=nca_steps)
    logits = model.color_logits(state)
    return logits.squeeze(0) if squeeze else logits
