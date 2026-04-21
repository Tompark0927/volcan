"""Run Volcan pretraining on the synthetic corpus.

Usage:
    python scripts/pretrain.py                            # default 1000 steps
    python scripts/pretrain.py --num-steps 2000           # longer run
    python scripts/pretrain.py --batch-size 32            # bigger batches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.arc import load_dataset  # noqa: E402
from volcan.pretrain import MultiTaskDataset, pretrain_volcan, pretrain_volcan_icl  # noqa: E402
from volcan.viz import plot_training_curve  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402
from volcan.training import TrainLog  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Volcan pretraining")
    parser.add_argument("--corpus", type=str, default="data/synthetic")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phase-a-max", type=int, default=6)
    parser.add_argument("--phase-b-steps", type=int, default=16)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--use-moe", action="store_true",
                        help="replace dense update MLP with Mixture-of-Experts (Path A)")
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-expert-hidden", type=int, default=128)
    parser.add_argument("--use-hierarchy", action="store_true",
                        help="enable hierarchical macro-cell layer (post-ceiling experiment)")
    parser.add_argument("--macro-channels", type=int, default=16)
    parser.add_argument("--macro-hidden", type=int, default=32)
    parser.add_argument("--macro-block-size", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="outputs/week4/volcan_pretrained.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icl", action="store_true",
                        help="use ICL pretraining (Week 5) instead of single-pair (Week 4)")
    parser.add_argument("--num-demos", type=int, default=3,
                        help="number of demos in each ICL context (Week 5 only)")
    parser.add_argument("--icl-steps-per-clamp", type=int, default=4,
                        help="NCA updates per clamped sub-phase in ICL Phase A")
    parser.add_argument("--lambda-activity", type=float, default=0.01,
                        help="weight on the metabolic-cost activity penalty (Week 6 / CEO #3)")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    torch.manual_seed(args.seed)

    # Load corpus.
    corpus_root = PROJECT_ROOT / args.corpus
    print(f"loading corpus from {corpus_root}")
    tasks = load_dataset(corpus_root, split="training")
    print(f"  {len(tasks)} tasks loaded")

    dataset = MultiTaskDataset(tasks, pad_to=30, seed=args.seed)

    # Build model.
    cfg = VolcanConfig(
        mlp_hidden=args.mlp_hidden,
        use_moe=args.use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_expert_hidden=args.moe_expert_hidden,
        use_hierarchy=args.use_hierarchy,
        macro_channels=args.macro_channels,
        macro_hidden=args.macro_hidden,
        macro_block_size=args.macro_block_size,
    )
    model = VolcanCell(cfg)
    n_params = model.num_params()
    mode = "ICL (Week 5)" if args.icl else "single-pair (Week 4)"
    print(f"model: VolcanCell({cfg.state_channels}ch, mlp_hidden={cfg.mlp_hidden}) — {n_params:,} params")
    print(
        f"pretraining ({mode}): {args.num_steps} steps, batch_size={args.batch_size}, "
        f"lr={args.lr}"
    )
    if args.icl:
        print(
            f"  ICL: K={args.num_demos} demos, "
            f"{args.icl_steps_per_clamp} NCA updates per clamp, Phase B: {args.phase_b_steps}"
        )
    else:
        print(f"  Phase A: ≤{args.phase_a_max}, Phase B: {args.phase_b_steps}")
    print()

    # Train.
    def on_log(step: int, loss: float, acc: float, elapsed: float) -> None:
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (args.num_steps - step) / rate if rate > 0 else 0.0
        print(
            f"  step {step:5d}/{args.num_steps}  loss={loss:.4f}  "
            f"acc={acc*100:5.1f}%  ({rate:4.1f} steps/s, ETA {eta:5.0f}s)"
        )

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if args.icl:
        log = pretrain_volcan_icl(
            model,
            dataset,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            num_demos=args.num_demos,
            lr=args.lr,
            icl_steps_per_clamp=args.icl_steps_per_clamp,
            phase_b_steps=args.phase_b_steps,
            device=device,
            log_every=max(args.num_steps // 20, 25),
            checkpoint_every=max(args.num_steps // 4, 100),
            checkpoint_path=checkpoint_path,
            on_log=on_log,
            seed=args.seed,
            lambda_activity=args.lambda_activity,
        )
    else:
        log = pretrain_volcan(
            model,
            dataset,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            phase_a_max=args.phase_a_max,
            phase_b_steps=args.phase_b_steps,
            device=device,
            log_every=max(args.num_steps // 20, 25),
            checkpoint_every=max(args.num_steps // 4, 100),
            checkpoint_path=checkpoint_path,
            on_log=on_log,
            seed=args.seed,
        )

    print()
    print(f"final loss: {log.losses[-1]:.4f}")
    print(f"final content acc: {log.accuracies[-1]*100:.1f}%")
    print(f"checkpoint saved to: {checkpoint_path}")

    # Save training curve. We reuse the existing plot_training_curve which
    # expects a TrainLog with steps/losses/accuracies attrs — PretrainLog has
    # the same shape so we wrap it.
    curve_log = TrainLog(steps=log.steps, losses=log.losses, accuracies=log.accuracies)
    curve_path = PROJECT_ROOT / "outputs" / "week4" / "pretrain_curve.png"
    plot_training_curve(curve_log, save_to=curve_path, title="Volcan pretraining (Week 4)")
    print(f"training curve saved to: {curve_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
