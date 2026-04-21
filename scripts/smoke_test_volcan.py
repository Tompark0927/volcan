"""Week 2 smoke test — full Volcan with all five pillars.

Same shape as Week 1's smoke_test.py, but uses VolcanCell instead of BasicNCA
and the two-phase + masked-denoising training loop.

Goal: prove the five-pillar architecture is trainable end-to-end on a real
ARC-AGI-2 task. Success criterion is looser than Week 1 because Volcan is
~40× bigger and the loss surface is more complex:

  - Loss drops by ≥ 1 order of magnitude
  - Cell accuracy reaches ≥ 90% on the demo pairs
  - Forward pass completes without numerical issues

We are NOT expecting 100% perfect overfit on Week 2 — that's a tuning problem
once we have the architecture stable. We're proving the plumbing is correct.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.arc import (  # noqa: E402
    Example,
    Task,
    grid_to_onehot,
    grids_equal,
    load_task,
    onehot_to_grid,
)
from volcan.training_volcan import (  # noqa: E402
    overfit_volcan_single_task,
    predict_volcan,
)
from volcan.viz import plot_task_prediction, plot_training_curve  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402


SYNTHETIC_TASK = Task(
    task_id="synthetic_recolor_volcan",
    train=[
        Example(
            input=[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            output=[[2, 0, 2], [0, 2, 0], [2, 0, 2]],
        ),
        Example(
            input=[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            output=[[2, 2, 0, 0], [0, 2, 2, 0], [0, 0, 2, 2]],
        ),
        Example(
            input=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            output=[[0, 0, 2], [0, 2, 0], [2, 0, 0]],
        ),
    ],
    test=[],
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Volcan Week 2 smoke test")
    parser.add_argument("--task-path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--phase-a-max", type=int, default=20)
    parser.add_argument("--phase-b-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=128,
        help="Update MLP hidden width. v0.2 default is 256; we shrink for the smoke test.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.task_path and not args.synthetic:
        task = load_task(args.task_path)
        print(f"loaded task {task.task_id} from {args.task_path}")
    else:
        task = SYNTHETIC_TASK
        print(f"using synthetic task '{task.task_id}'")
    print(f"  {task.num_train} demo pairs")

    device = pick_device()
    print(f"device: {device}")

    cfg = VolcanConfig(mlp_hidden=args.mlp_hidden)
    model = VolcanCell(cfg)
    n_params = model.num_params()
    print(
        f"model: VolcanCell(state_channels={cfg.state_channels}, "
        f"mlp_hidden={cfg.mlp_hidden}) — {n_params:,} params"
    )
    print(
        f"  pillars: forces(8 dirs × {cfg.force_dim}d) + "
        f"mycelial(K={cfg.mycelial_partners}) + "
        f"spectral(modes={cfg.spectral_modes}) + ghost({cfg.ghost_channels}d) + apoptosis"
    )

    def on_log(step: int, loss: float, acc: float, dn: float, reg: float) -> None:
        print(
            f"  step {step:4d}  loss={loss:.4f}  acc={acc*100:5.1f}%  "
            f"L_denoise={dn:.4f}  L_regime={reg:.4f}"
        )

    print(
        f"training for {args.num_steps} steps "
        f"(Phase A: ≤{args.phase_a_max}, Phase B: {args.phase_b_steps})"
    )
    t0 = time.time()
    log = overfit_volcan_single_task(
        model,
        task,
        num_steps=args.num_steps,
        phase_a_max=args.phase_a_max,
        phase_b_steps=args.phase_b_steps,
        device=device,
        on_log=on_log,
    )
    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s")

    # Evaluate.
    demos = []
    exact_match_count = 0
    for ex in task.train:
        inp_onehot = grid_to_onehot(ex.input).to(device)
        pred_logits = predict_volcan(
            model, inp_onehot, phase_a_max=args.phase_a_max, phase_b_steps=args.phase_b_steps
        )
        pred_grid = onehot_to_grid(pred_logits.cpu())
        demos.append((ex.input, ex.output, pred_grid))
        if grids_equal(pred_grid, ex.output):
            exact_match_count += 1

    outputs_dir = PROJECT_ROOT / "outputs" / "week2"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    curve_path = outputs_dir / f"{task.task_id}_curve.png"
    pred_path = outputs_dir / f"{task.task_id}_predictions.png"
    plot_training_curve(log, save_to=curve_path, title=f"{task.task_id} — Volcan v0.2 smoke test")
    plot_task_prediction(task_id=task.task_id, demos=demos, save_to=pred_path)

    initial_loss = log.losses[0]
    final_loss = log.losses[-1]
    final_acc = log.accuracies[-1]
    loss_drop = initial_loss / max(final_loss, 1e-12)

    print("")
    print("=" * 60)
    print(f"Week 2 smoke test report — task {task.task_id}")
    print("=" * 60)
    print(f"  initial loss:        {initial_loss:.4f}")
    print(f"  final loss:          {final_loss:.4f}")
    print(f"  loss drop:           {loss_drop:.1f}×")
    print(f"  final cell accuracy: {final_acc*100:.2f}%")
    print(f"  exact-match demos:   {exact_match_count}/{task.num_train}")
    print(f"  model params:        {n_params:,}")
    print(f"  training time:       {elapsed:.1f}s")
    print(f"  training curve:      {curve_path}")
    print(f"  predictions:         {pred_path}")

    pass_loss = loss_drop >= 10.0
    pass_acc = final_acc >= 0.90
    passed = pass_loss and pass_acc
    print("")
    print(f"  pass criterion — loss drop ≥ 10×:    {'PASS' if pass_loss else 'FAIL'}")
    print(f"  pass criterion — cell accuracy ≥ 90%: {'PASS' if pass_acc else 'FAIL'}")
    print(f"  OVERALL:                               {'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
