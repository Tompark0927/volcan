"""Week 1 smoke test.

Goal: prove the stack (data → model → training → viz) works end-to-end.

Procedure:
  1. Load (or synthesize) one ARC task.
  2. Instantiate a small BasicNCA.
  3. Overfit its demo pairs.
  4. Save a training curve PNG and a before/after PNG.
  5. Print a short report.

Success criterion: loss monotonically decreases by ≥ 1 order of magnitude AND
cell accuracy reaches ≥ 99% on the demo pairs within 500 gradient steps.

This does NOT use any of the five Volcan pillars yet. It's a baseline sanity
check that the plumbing is correct. The full Volcan cell lands in Week 2.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --task-path data/ARC-AGI-2/data/training/00576224.json
    python scripts/smoke_test.py --synthetic   # force the hand-crafted task
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Make `import volcan` work when running from the project root.
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
from volcan.models import BasicNCA  # noqa: E402
from volcan.training import overfit_single_task, predict  # noqa: E402
from volcan.viz import plot_task_prediction, plot_training_curve  # noqa: E402


# -----------------------------------------------------------------------------
# Hand-crafted fallback task (no dataset needed)
# -----------------------------------------------------------------------------
#
# A simple "recolor" task: every blue (1) cell becomes red (2), everything else
# stays the same. Hand-designed to be clearly learnable by a tiny NCA in ~100
# gradient steps.

SYNTHETIC_TASK = Task(
    task_id="synthetic_recolor",
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
    parser = argparse.ArgumentParser(description="Volcan Week 1 smoke test")
    parser.add_argument(
        "--task-path",
        type=str,
        default=None,
        help="Optional path to an ARC task JSON. If omitted, uses the synthetic task.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic task.")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--nca-steps", type=int, default=32)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load the task.
    if args.task_path and not args.synthetic:
        task = load_task(args.task_path)
        print(f"loaded task {task.task_id} from {args.task_path}")
    else:
        task = SYNTHETIC_TASK
        print(f"using synthetic task '{task.task_id}'")
    print(f"  {task.num_train} demo pairs")

    device = pick_device()
    print(f"device: {device}")

    # Build the model.
    model = BasicNCA(channels=args.channels, hidden=args.hidden)
    n_params = model.num_params()
    print(f"model: BasicNCA(channels={args.channels}, hidden={args.hidden}) — {n_params:,} params")

    # Train.
    def on_log(step: int, loss: float, acc: float) -> None:
        print(f"  step {step:4d}  loss={loss:.4f}  acc={acc*100:5.1f}%")

    print(f"training for {args.num_steps} steps, {args.nca_steps} NCA iterations per step")
    t0 = time.time()
    log = overfit_single_task(
        model,
        task,
        num_steps=args.num_steps,
        nca_steps=args.nca_steps,
        device=device,
        on_log=on_log,
    )
    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s")

    # Evaluate and render the final predictions.
    demos = []
    exact_match_count = 0
    for ex in task.train:
        inp_onehot = grid_to_onehot(ex.input).to(device)
        pred_logits = predict(model, inp_onehot, nca_steps=args.nca_steps)
        pred_grid = onehot_to_grid(pred_logits.cpu())
        demos.append((ex.input, ex.output, pred_grid))
        if grids_equal(pred_grid, ex.output):
            exact_match_count += 1

    # Save figures.
    outputs_dir = PROJECT_ROOT / "outputs" / "week1"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    curve_path = outputs_dir / f"{task.task_id}_curve.png"
    pred_path = outputs_dir / f"{task.task_id}_predictions.png"
    plot_training_curve(
        log,
        save_to=curve_path,
        title=f"{task.task_id} — BasicNCA smoke test",
    )
    plot_task_prediction(
        task_id=task.task_id,
        demos=demos,
        save_to=pred_path,
    )

    # Final report.
    final_loss = log.losses[-1]
    final_acc = log.accuracies[-1]
    initial_loss = log.losses[0]
    loss_drop_orders = (initial_loss / max(final_loss, 1e-12))

    print("")
    print("=" * 60)
    print(f"smoke test report — task {task.task_id}")
    print("=" * 60)
    print(f"  initial loss:        {initial_loss:.4f}")
    print(f"  final loss:          {final_loss:.4f}")
    print(f"  loss drop:           {loss_drop_orders:.1f}×")
    print(f"  final cell accuracy: {final_acc*100:.2f}%")
    print(f"  exact-match demos:   {exact_match_count}/{task.num_train}")
    print(f"  model params:        {n_params:,}")
    print(f"  training time:       {elapsed:.1f}s")
    print(f"  training curve:      {curve_path}")
    print(f"  predictions:         {pred_path}")

    # Success criterion.
    pass_loss = loss_drop_orders >= 10.0
    pass_acc = final_acc >= 0.99
    passed = pass_loss and pass_acc
    print("")
    print(f"  pass criterion — loss drop ≥ 10×:    {'PASS' if pass_loss else 'FAIL'}")
    print(f"  pass criterion — cell accuracy ≥ 99%: {'PASS' if pass_acc else 'FAIL'}")
    print(f"  OVERALL:                               {'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
