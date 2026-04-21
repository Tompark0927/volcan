"""Week 2.5 batch evaluation — run Volcan on a diverse set of real ARC-AGI-2 tasks.

Goal: see where Volcan succeeds and where it breaks. Picks tasks across several
difficulty/category buckets:

  - Pure color substitution (local rule)
  - Global property (e.g., "fill with dominant color")
  - Spatial rotation / flip
  - Position-dependent recolor
  - Multi-step / structural

For each task we train Volcan from scratch (overfit-style), record final loss,
content accuracy, and exact-match demo count, and save the prediction PNG.
Output: a summary table printed to stdout, plus per-task PNGs in outputs/week2_5/.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.arc import grid_to_onehot, grids_equal, load_task, onehot_to_grid  # noqa: E402
from volcan.training_volcan import overfit_volcan_single_task, predict_volcan  # noqa: E402
from volcan.viz import plot_task_prediction  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402


# (task_id, category) — chosen to span difficulty.
TASK_BATCH = [
    ("0d3d703e", "color-sub"),       # 3x3 — pure color mapping (baseline)
    ("9565186b", "color-sub"),       # 3x3 — recolor 8→5
    ("25d8a9c8", "row-recolor"),     # 3x3 — row-pattern recolor
    ("a85d4709", "pos-recolor"),     # 3x3 — recolor by row position
    ("5582e5ca", "global"),          # 3x3 — fill with dominant color
    ("6150a2bd", "spatial-rot"),     # 3x3 — 180° rotation
    ("74dd1130", "spatial-flip"),    # 3x3 — diagonal flip / transpose
    ("3c9b0459", "spatial-rot"),     # 3x3 — flip
    ("d037b0a7", "fill-down"),       # 3x3 — column flood-fill
    ("00d62c1b", "interior"),        # 10x10 — interior detection
]


@dataclass
class TaskResult:
    task_id: str
    category: str
    initial_loss: float
    final_loss: float
    loss_drop: float
    content_acc: float
    exact_match: int
    num_demos: int
    test_exact: int
    num_test: int
    train_time: float


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_task(
    task_id: str,
    category: str,
    *,
    data_root: Path,
    output_dir: Path,
    num_steps: int,
    phase_a_max: int,
    phase_b_steps: int,
    device: str,
    seed: int,
) -> TaskResult:
    task = load_task(data_root / "ARC-AGI-2" / "data" / "training" / f"{task_id}.json")
    torch.manual_seed(seed)
    cfg = VolcanConfig(mlp_hidden=128)
    model = VolcanCell(cfg)

    t0 = time.time()
    log = overfit_volcan_single_task(
        model,
        task,
        num_steps=num_steps,
        phase_a_max=phase_a_max,
        phase_b_steps=phase_b_steps,
        device=device,
        log_every=num_steps,  # only log final
    )
    elapsed = time.time() - t0

    # Evaluate demo exact-match (overfit check).
    demos = []
    exact = 0
    for ex in task.train:
        inp = grid_to_onehot(ex.input).to(device)
        pred_logits = predict_volcan(
            model, inp, phase_a_max=phase_a_max, phase_b_steps=phase_b_steps
        )
        pred_grid = onehot_to_grid(pred_logits.cpu())
        demos.append((ex.input, ex.output, pred_grid))
        if grids_equal(pred_grid, ex.output):
            exact += 1

    plot_task_prediction(
        task_id=task_id,
        demos=demos,
        save_to=output_dir / f"{task_id}_predictions.png",
    )

    # Evaluate held-out TEST inputs — this is the real generalization test.
    # ARC public training tasks include the test outputs as ground truth.
    test_demos = []
    test_exact = 0
    for ex in task.test:
        if not ex.output:
            continue  # safety: skip if no ground truth
        inp = grid_to_onehot(ex.input).to(device)
        pred_logits = predict_volcan(
            model, inp, phase_a_max=phase_a_max, phase_b_steps=phase_b_steps
        )
        pred_grid = onehot_to_grid(pred_logits.cpu())
        test_demos.append((ex.input, ex.output, pred_grid))
        if grids_equal(pred_grid, ex.output):
            test_exact += 1

    if test_demos:
        plot_task_prediction(
            task_id=f"{task_id}_TEST",
            demos=test_demos,
            save_to=output_dir / f"{task_id}_test_predictions.png",
        )

    return TaskResult(
        task_id=task_id,
        category=category,
        initial_loss=log.losses[0],
        final_loss=log.losses[-1],
        loss_drop=log.losses[0] / max(log.losses[-1], 1e-12),
        content_acc=log.accuracies[-1],
        exact_match=exact,
        num_demos=task.num_train,
        test_exact=test_exact,
        num_test=len(test_demos),
        train_time=elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Volcan Week 2.5 batch eval")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--phase-a-max", type=int, default=6)
    parser.add_argument("--phase-b-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-filter", type=str, default=None,
                        help="optional comma-separated subset of task IDs")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    output_dir = PROJECT_ROOT / "outputs" / "week2_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = TASK_BATCH
    if args.task_filter:
        keep = set(args.task_filter.split(","))
        tasks = [(tid, cat) for tid, cat in tasks if tid in keep]

    print(f"running {len(tasks)} tasks, {args.num_steps} steps each")
    print(f"  config: phase_a_max={args.phase_a_max}, phase_b_steps={args.phase_b_steps}")
    print()

    results: list[TaskResult] = []
    for tid, cat in tasks:
        print(f"  [{tid}] ({cat}) ...", end=" ", flush=True)
        try:
            r = evaluate_task(
                tid,
                cat,
                data_root=PROJECT_ROOT / "data",
                output_dir=output_dir,
                num_steps=args.num_steps,
                phase_a_max=args.phase_a_max,
                phase_b_steps=args.phase_b_steps,
                device=device,
                seed=args.seed,
            )
            results.append(r)
            print(
                f"loss {r.initial_loss:.2f}→{r.final_loss:.3f}  "
                f"demos={r.exact_match}/{r.num_demos}  "
                f"TEST={r.test_exact}/{r.num_test}  {r.train_time:.0f}s"
            )
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    # Summary table.
    print()
    print("=" * 96)
    print("Volcan Week 2.5 — batch eval summary")
    print("=" * 96)
    header = (
        f"{'task_id':<12} {'category':<14} {'init':>6} {'final':>6} "
        f"{'drop':>7} {'demos':>9} {'TEST':>9} {'time':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.task_id:<12} {r.category:<14} {r.initial_loss:>6.2f} "
            f"{r.final_loss:>6.3f} {r.loss_drop:>6.0f}× "
            f"{r.exact_match:>3}/{r.num_demos:<3}    "
            f"{r.test_exact:>3}/{r.num_test:<3}    {r.train_time:>4.0f}s"
        )

    # Aggregate — the only number that really matters is TEST.
    if results:
        total_demos = sum(r.num_demos for r in results)
        demo_exact = sum(r.exact_match for r in results)
        total_test = sum(r.num_test for r in results)
        test_exact = sum(r.test_exact for r in results)
        demo_solved = sum(1 for r in results if r.exact_match == r.num_demos)
        test_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact == r.num_test
        )
        print("-" * len(header))
        print(
            f"  demo overfit:  {demo_solved}/{len(results)} tasks fully solved   "
            f"({demo_exact}/{total_demos} demos exact)"
        )
        print(
            f"  TEST (held-out): {test_solved}/{len(results)} tasks fully solved   "
            f"({test_exact}/{total_test} test inputs exact)"
        )

    # Per-category breakdown — by TEST accuracy.
    by_cat: dict[str, list[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    print()
    print("by category (held-out TEST):")
    for cat, rs in sorted(by_cat.items()):
        solved = sum(1 for r in rs if r.num_test > 0 and r.test_exact == r.num_test)
        n_test = sum(r.num_test for r in rs)
        n_test_ok = sum(r.test_exact for r in rs)
        print(
            f"  {cat:<14} {solved}/{len(rs)} tasks   "
            f"({n_test_ok}/{n_test} test inputs)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
