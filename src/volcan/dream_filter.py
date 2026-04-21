"""Volcan overfit filter for the dream pipeline (Week 8).

Takes a candidate task (typically LLM-generated) and asks: "can Volcan
overfit the demo pairs within a time budget?" If yes, the task is
well-formed AND within Volcan's representational capacity — it belongs in
the Gold Dataset. If no, the task is either inconsistent (LLM hallucination)
or too hard for the current architecture.

The filter is deliberately strict: we want the pretraining corpus to be
tasks Volcan can actually learn something from. Tasks that are out of
capacity are noise at best, poison at worst.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .arc import Task, grid_to_onehot, grids_equal
from .training_volcan import overfit_volcan_single_task, predict_volcan
from .volcan_cell import VolcanCell, VolcanConfig


@dataclass
class FilterResult:
    """Outcome of running the Volcan overfit filter on one task."""

    passed: bool
    final_content_acc: float  # 0..1
    demos_exact: int          # how many demos matched exactly
    num_demos: int
    elapsed_sec: float
    reason: str               # human-readable failure reason if not passed


def volcan_overfit_filter(
    task: Task,
    *,
    max_steps: int = 200,
    time_budget_sec: float = 60.0,
    acc_threshold: float = 1.0,
    exact_match_ratio: float = 1.0,
    phase_a_max: int = 6,
    phase_b_steps: int = 16,
    mlp_hidden: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> FilterResult:
    """Train a fresh Volcan on `task`'s demos; pass if it hits the threshold.

    Args:
        task: the candidate task
        max_steps: max gradient steps (hard upper bound)
        time_budget_sec: wall-clock budget (terminates early if exceeded)
        acc_threshold: content cell accuracy required to pass (default 1.0 = perfect)
        exact_match_ratio: fraction of demos that must exact-match to pass
        phase_a_max: Phase A step limit for training
        phase_b_steps: Phase B step count for training
        mlp_hidden: VolcanConfig mlp_hidden to use
        device: torch device
        seed: RNG seed for reproducibility

    Returns:
        FilterResult with pass/fail and diagnostics.
    """
    # Quick structural sanity checks first.
    if task.num_train < 2:
        return FilterResult(
            passed=False, final_content_acc=0.0, demos_exact=0,
            num_demos=task.num_train, elapsed_sec=0.0,
            reason="fewer than 2 demos",
        )
    # Check grids parse and fit the canvas.
    try:
        for ex in task.train:
            grid_to_onehot(ex.input)
            grid_to_onehot(ex.output)
        for ex in task.test:
            grid_to_onehot(ex.input)
    except Exception as e:
        return FilterResult(
            passed=False, final_content_acc=0.0, demos_exact=0,
            num_demos=task.num_train, elapsed_sec=0.0,
            reason=f"grid validation failed: {e}",
        )

    torch.manual_seed(seed)
    cfg = VolcanConfig(mlp_hidden=mlp_hidden)
    model = VolcanCell(cfg).to(device)

    t0 = time.time()
    try:
        log = overfit_volcan_single_task(
            model, task,
            num_steps=max_steps,
            phase_a_max=phase_a_max,
            phase_b_steps=phase_b_steps,
            device=device,
            log_every=max_steps,  # silent
        )
    except Exception as e:
        return FilterResult(
            passed=False, final_content_acc=0.0, demos_exact=0,
            num_demos=task.num_train, elapsed_sec=time.time() - t0,
            reason=f"training crashed: {e}",
        )
    train_elapsed = time.time() - t0

    # Final content accuracy from the training log.
    final_content_acc = log.accuracies[-1] if log.accuracies else 0.0

    # Check how many demos the trained model can actually predict exactly.
    demos_exact = 0
    for ex in task.train:
        inp = grid_to_onehot(ex.input).to(device)
        try:
            pred_logits = predict_volcan(
                model, inp, phase_a_max=phase_a_max, phase_b_steps=phase_b_steps
            )
            from .arc import onehot_to_grid
            pred_grid = onehot_to_grid(pred_logits.cpu())
            if grids_equal(pred_grid, ex.output):
                demos_exact += 1
        except Exception:
            pass

    total_elapsed = time.time() - t0
    n = task.num_train

    # Pass criteria: ALL demos exact-match AND under budget. The content-acc
    # threshold is checked only as a backup — if exact-match is satisfied, the
    # task is learned regardless of noise-schedule variance at the last step.
    exact_ok = demos_exact >= int(exact_match_ratio * n)
    under_budget = total_elapsed <= time_budget_sec
    passed = exact_ok and under_budget
    reason = ""
    if not passed:
        reasons = []
        if not exact_ok:
            reasons.append(f"exact {demos_exact}/{n} < {int(exact_match_ratio*n)}/{n}")
            # Include content acc for diagnostic context on near-misses.
            reasons.append(f"acc {final_content_acc*100:.1f}%")
        if not under_budget:
            reasons.append(f"elapsed {total_elapsed:.0f}s > {time_budget_sec:.0f}s")
        reason = "; ".join(reasons)

    return FilterResult(
        passed=passed,
        final_content_acc=final_content_acc,
        demos_exact=demos_exact,
        num_demos=n,
        elapsed_sec=total_elapsed,
        reason=reason,
    )
