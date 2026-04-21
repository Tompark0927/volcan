"""Evaluate a pretrained Volcan checkpoint on real ARC-AGI-2 tasks.

For each task in the Week 2.5 batch, runs TWO conditions:
  - Pretrained, NO TTT: load checkpoint, predict on the test input directly.
                        Tests whether pretraining alone gives useful priors.
  - Pretrained + TTT:   load checkpoint, fine-tune on the task's demos for
                        N steps, then predict. Tests the full pipeline.

The Week 2.5 baseline (from-scratch + TTT) gives the third comparison cell;
the from-scratch + no-TTT cell is meaningless (untrained model = noise).

Output: a 3-cell comparison table on the same 10 tasks Week 2.5 used.
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
from volcan.hyperttt import HyperNetwork, LoRASchema, infer_lora_schema  # noqa: E402
from volcan.pretrain import (  # noqa: E402
    load_checkpoint,
    predict_volcan_d8_ensemble,
    predict_volcan_icl,
)
from volcan.training_volcan import overfit_volcan_single_task, predict_volcan  # noqa: E402
from volcan.viz import plot_task_prediction  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402


# Week 2.5 batch (first 10) + Week 6 expansion (20 more, small fixed-size tasks).
TASK_BATCH = [
    # --- Week 2.5 tier (10 tasks) ---
    ("0d3d703e", "color-sub"),
    ("9565186b", "color-sub"),
    ("25d8a9c8", "row-recolor"),
    ("a85d4709", "pos-recolor"),
    ("5582e5ca", "global"),
    ("6150a2bd", "spatial-rot"),
    ("74dd1130", "spatial-flip"),
    ("3c9b0459", "spatial-rot"),
    ("d037b0a7", "fill-down"),
    ("00d62c1b", "interior"),
    # --- Week 6 expansion (20 more small fixed-size tasks) ---
    ("5614dbcf", "?"),
    ("ed36ccf7", "?"),
    ("746b3537", "?"),
    ("68b16354", "?"),
    ("6d0aefbc", "?"),
    ("9dfd6313", "?"),
    ("b1948b0a", "?"),
    ("67a3c6ac", "?"),
    ("4347f46a", "?"),
    ("8be77c9e", "?"),
    ("3af2c5a8", "?"),
    ("62c24649", "?"),
    ("445eab21", "?"),
    ("5bd6f4ac", "?"),
    ("bda2d7a6", "?"),
    ("178fcbfb", "?"),
    ("44d8ac46", "?"),
    ("32597951", "?"),
    ("3e980e27", "?"),
    ("05269061", "?"),
]


@dataclass
class EvalResult:
    task_id: str
    category: str
    test_exact_no_ttt: int       # condition 1: pretrained, no TTT, no ICL
    test_exact_icl: int          # condition 2: pretrained, no TTT, WITH ICL  ← Week 5
    test_exact_ttt: int          # condition 3: pretrained, +TTT, no ICL      ← Week 4
    test_exact_d8_top1: int      # condition 4a: D8-Ensemble best symmetry    ← Week 6
    test_exact_d8_top2: int      # condition 4b: D8-Ensemble top-2 (either matches)
    num_demos: int
    num_test: int
    ttt_time: float


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_test_no_icl(
    model: VolcanCell,
    task,
    *,
    device: str,
    phase_a_max: int,
    phase_b_steps: int,
) -> int:
    """Predict each task.test input WITHOUT ICL conditioning. Returns count exact-match."""
    exact = 0
    for ex in task.test:
        if not ex.output:
            continue
        inp = grid_to_onehot(ex.input).to(device)
        pred_logits = predict_volcan(
            model, inp, phase_a_max=phase_a_max, phase_b_steps=phase_b_steps
        )
        pred_grid = onehot_to_grid(pred_logits.cpu())
        if grids_equal(pred_grid, ex.output):
            exact += 1
    return exact


def evaluate_test_icl(
    model: VolcanCell,
    task,
    *,
    device: str,
    icl_steps_per_clamp: int,
    phase_b_steps: int,
) -> int:
    """Predict each task.test input WITH ICL — feed task.train demos as context."""
    exact = 0
    demo_inputs = [ex.input for ex in task.train]
    demo_outputs = [ex.output for ex in task.train]
    for ex in task.test:
        if not ex.output:
            continue
        query_input = grid_to_onehot(ex.input)
        pred_logits = predict_volcan_icl(
            model,
            demo_inputs,
            demo_outputs,
            query_input,
            icl_steps_per_clamp=icl_steps_per_clamp,
            phase_b_steps=phase_b_steps,
            device=device,
        )
        pred_grid = onehot_to_grid(pred_logits.cpu())
        if grids_equal(pred_grid, ex.output):
            exact += 1
    return exact


def evaluate_test_d8(
    model: VolcanCell,
    task,
    *,
    device: str,
    icl_steps_per_clamp: int,
    phase_b_steps: int,
) -> tuple[int, int]:
    """Predict each task.test input via D8-Ensemble Resonance.

    Returns (top1_exact, top2_exact) — top1 is the most-stable symmetry, top2
    counts a hit if EITHER of the top-2 most-stable symmetries matches.
    """
    top1 = 0
    top2 = 0
    demo_inputs = [ex.input for ex in task.train]
    demo_outputs = [ex.output for ex in task.train]
    for ex in task.test:
        if not ex.output:
            continue
        query_input = grid_to_onehot(ex.input)
        candidates = predict_volcan_d8_ensemble(
            model,
            demo_inputs,
            demo_outputs,
            query_input,
            icl_steps_per_clamp=icl_steps_per_clamp,
            phase_b_steps=phase_b_steps,
            device=device,
            top_k=2,
        )
        cand_grids = [onehot_to_grid(c.cpu()) for c in candidates]
        if cand_grids and grids_equal(cand_grids[0], ex.output):
            top1 += 1
        if any(grids_equal(g, ex.output) for g in cand_grids):
            top2 += 1
    return top1, top2


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval pretrained Volcan checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/week4/volcan_pretrained.pt"
    )
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--use-moe", action="store_true",
                        help="instantiate an MoE VolcanCell (must match the checkpoint)")
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-expert-hidden", type=int, default=128)
    parser.add_argument("--use-hierarchy", action="store_true")
    parser.add_argument("--macro-channels", type=int, default=16)
    parser.add_argument("--macro-hidden", type=int, default=32)
    parser.add_argument("--macro-block-size", type=int, default=3)
    parser.add_argument("--ttt-steps", type=int, default=200)
    parser.add_argument("--ttt-lr", type=float, default=2e-3)
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank for TTT (Phase 1 Step 3 default: 16)")
    parser.add_argument("--use-hypernet", action="store_true",
                        help="initialize LoRA from a trained HyperNetwork prediction (Hyper-TTT)")
    parser.add_argument("--hypernet-checkpoint", type=str, default=None,
                        help="path to hypernet .pt (required if --use-hypernet)")
    parser.add_argument("--hypernet-num-demos", type=int, default=3,
                        help="number of demos the hypernet was trained with")
    parser.add_argument("--phase-a-max", type=int, default=6)
    parser.add_argument("--phase-b-steps", type=int, default=16)
    parser.add_argument("--icl-steps-per-clamp", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-filter", type=str, default=None)
    parser.add_argument("--skip-ttt", action="store_true",
                        help="skip the TTT condition (faster eval if you only care about ICL)")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"  ERROR: checkpoint not found at {checkpoint_path}")
        print("  run scripts/pretrain.py first")
        return 1

    output_dir = PROJECT_ROOT / "outputs" / "week4"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_meta = TASK_BATCH
    if args.task_filter:
        keep = set(args.task_filter.split(","))
        tasks_meta = [(tid, cat) for tid, cat in tasks_meta if tid in keep]

    print(f"running 4-cell eval on {len(tasks_meta)} ARC-AGI-2 tasks")
    print(f"  pretrained checkpoint: {checkpoint_path}")
    print(f"  TTT: {args.ttt_steps} steps, lr {args.ttt_lr}")

    # ---- Hyper-TTT setup: load hypernet if requested ----
    hypernet = None
    hypernet_schema = None
    hypernet_target_mean = None
    hypernet_target_std = None
    if args.use_hypernet:
        if not args.hypernet_checkpoint:
            print("  ERROR: --use-hypernet requires --hypernet-checkpoint")
            return 1
        hn_path = PROJECT_ROOT / args.hypernet_checkpoint
        if not hn_path.exists():
            print(f"  ERROR: hypernet checkpoint not found at {hn_path}")
            return 1
        print(f"  Hyper-TTT: loading hypernet from {hn_path}")
        hn_ckpt = torch.load(hn_path, map_location=device, weights_only=False)
        hypernet_schema = LoRASchema(
            conv_shapes=hn_ckpt["schema_conv_shapes"],
            rank=hn_ckpt["schema_rank"],
        )
        # Build a frozen base Volcan for the hypernet encoder; reload weights
        # per task is unnecessary — base is frozen during hypernet training.
        hn_base_cfg = VolcanConfig(
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
        hn_base = VolcanCell(hn_base_cfg).to(device)
        load_checkpoint(hn_base, checkpoint_path, device=device)
        hn_base.eval()
        hypernet = HyperNetwork(
            base_model=hn_base,
            schema=hypernet_schema,
            task_embed_dim=hn_ckpt.get("task_embed_dim", 64),
            decoder_hidden=hn_ckpt.get("decoder_hidden", 256),
            freeze_base=True,
        ).to(device)
        hypernet.load_state_dict(hn_ckpt["hypernet_state_dict"])
        hypernet.eval()
        hypernet_target_mean = hn_ckpt["target_mean"].to(device)
        hypernet_target_std = hn_ckpt["target_std"].to(device)
        print(f"  Hyper-TTT: schema {len(hypernet_schema.conv_shapes)} convs, "
              f"{hypernet_schema.total_params} params, trained at epoch "
              f"{hn_ckpt.get('epoch', '?')} val_loss={hn_ckpt.get('val_loss', 0):.4f}")
    print()

    results: list[EvalResult] = []

    for tid, cat in tasks_meta:
        print(f"  [{tid}] ({cat}) ...", end=" ", flush=True)
        task = load_task(
            PROJECT_ROOT / "data" / "ARC-AGI-2" / "data" / "training" / f"{tid}.json"
        )
        n_test = len([ex for ex in task.test if ex.output])

        # ----- Condition 1: Pretrained, NO TTT, NO ICL -----
        torch.manual_seed(args.seed)
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
        model = VolcanCell(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device=device)
        model.eval()

        t_no = evaluate_test_no_icl(
            model, task,
            device=device,
            phase_a_max=args.phase_a_max,
            phase_b_steps=args.phase_b_steps,
        )

        # ----- Condition 2 (Week 5): Pretrained, NO TTT, WITH ICL -----
        t_icl = evaluate_test_icl(
            model, task,
            device=device,
            icl_steps_per_clamp=args.icl_steps_per_clamp,
            phase_b_steps=args.phase_b_steps,
        )

        # ----- Condition 4 (Week 6): Pretrained, NO TTT, D8-Ensemble Resonance -----
        d8_top1, d8_top2 = evaluate_test_d8(
            model, task,
            device=device,
            icl_steps_per_clamp=args.icl_steps_per_clamp,
            phase_b_steps=args.phase_b_steps,
        )

        # ----- Condition 3 (Week 4): Pretrained + TTT, no ICL -----
        ttt_time = 0.0
        t_ttt = 0
        if not args.skip_ttt:
            torch.manual_seed(args.seed)
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
            model_ttt = VolcanCell(cfg).to(device)
            load_checkpoint(model_ttt, checkpoint_path, device=device)

            # --- Hyper-TTT: predict LoRA init from demos if enabled ---
            hn_lora_flat = None
            if hypernet is not None:
                num_ctx = min(args.hypernet_num_demos, task.num_train - 1)
                if num_ctx < 1:
                    print("  WARN: task has <2 demos, falling back to Kaiming init")
                else:
                    demo_inputs = [ex.input for ex in task.train]
                    demo_outputs = [ex.output for ex in task.train]
                    query_in = grid_to_onehot(demo_inputs[0]).unsqueeze(0).to(device)
                    context_in = torch.stack(
                        [grid_to_onehot(inp) for inp in demo_inputs[1:1 + num_ctx]]
                    ).unsqueeze(0).to(device)
                    context_out = torch.stack(
                        [grid_to_onehot(out) for out in demo_outputs[1:1 + num_ctx]]
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_norm = hypernet(context_in, context_out, query_in)
                    # Denormalize and take the single-task prediction
                    hn_lora_flat = (
                        pred_norm.squeeze(0) * hypernet_target_std + hypernet_target_mean
                    ).detach()

            ttt_t0 = time.time()
            overfit_volcan_single_task(
                model_ttt,
                task,
                num_steps=args.ttt_steps,
                phase_a_max=args.phase_a_max,
                phase_b_steps=args.phase_b_steps,
                lr=args.ttt_lr,
                device=device,
                log_every=args.ttt_steps,  # silent
                d8_augment=True,       # Phase 1 Step 1: D8 augmentation (proven +1 task)
                loo_validation=False,  # Phase 1 Step 2: disabled — lost 1 task as implemented
                use_lora=True,         # Phase 1 Step 3: freeze base weights, train rank-N LoRA
                lora_rank=args.lora_rank,
                lora_alpha=float(args.lora_rank),  # scale alpha with rank (standard LoRA practice)
                lora_lr_multiplier=5.0,
                hypernet_lora_flat=hn_lora_flat,
                hypernet_lora_schema=hypernet_schema if hn_lora_flat is not None else None,
            )
            ttt_time = time.time() - ttt_t0

            t_ttt = evaluate_test_no_icl(
                model_ttt, task,
                device=device,
                phase_a_max=args.phase_a_max,
                phase_b_steps=args.phase_b_steps,
            )

        # Save the ICL test prediction visualization.
        if task.test:
            test_demos_icl = []
            demo_inputs = [ex.input for ex in task.train]
            demo_outputs = [ex.output for ex in task.train]
            for ex in task.test:
                if not ex.output:
                    continue
                qi = grid_to_onehot(ex.input)
                p_icl = onehot_to_grid(predict_volcan_icl(
                    model, demo_inputs, demo_outputs, qi,
                    icl_steps_per_clamp=args.icl_steps_per_clamp,
                    phase_b_steps=args.phase_b_steps,
                    device=device,
                ).cpu())
                test_demos_icl.append((ex.input, ex.output, p_icl))
            if test_demos_icl:
                plot_task_prediction(
                    f"{tid}_pretrained_icl",
                    test_demos_icl,
                    save_to=output_dir / f"{tid}_pretrained_icl.png",
                )

        r = EvalResult(
            task_id=tid,
            category=cat,
            test_exact_no_ttt=t_no,
            test_exact_icl=t_icl,
            test_exact_ttt=t_ttt,
            test_exact_d8_top1=d8_top1,
            test_exact_d8_top2=d8_top2,
            num_demos=task.num_train,
            num_test=n_test,
            ttt_time=ttt_time,
        )
        results.append(r)
        ttt_str = "skipped" if args.skip_ttt else f"+TTT={t_ttt}/{n_test} ({ttt_time:.0f}s)"
        print(
            f"no-TTT={t_no}/{n_test}  +ICL={t_icl}/{n_test}  "
            f"D8(top1)={d8_top1}/{n_test} D8(top2)={d8_top2}/{n_test}  {ttt_str}"
        )

    # Summary table.
    print()
    print("=" * 110)
    print("Volcan Week 6 — pretrained eval summary (TEST exact match per condition)")
    print("=" * 110)
    header = (
        f"{'task_id':<12} {'category':<14} "
        f"{'no-ICL':>9} {'+ICL':>9} {'D8 top1':>9} {'D8 top2':>9} {'+TTT':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ttt_cell = f"{r.test_exact_ttt}/{r.num_test}" if not args.skip_ttt else "skip"
        print(
            f"{r.task_id:<12} {r.category:<14} "
            f"{r.test_exact_no_ttt:>5}/{r.num_test:<3} "
            f"{r.test_exact_icl:>5}/{r.num_test:<3} "
            f"{r.test_exact_d8_top1:>5}/{r.num_test:<3} "
            f"{r.test_exact_d8_top2:>5}/{r.num_test:<3} "
            f"{ttt_cell:>9}"
        )

    if results:
        total_test = sum(r.num_test for r in results)
        no_t = sum(r.test_exact_no_ttt for r in results)
        icl_t = sum(r.test_exact_icl for r in results)
        d8_top1_t = sum(r.test_exact_d8_top1 for r in results)
        d8_top2_t = sum(r.test_exact_d8_top2 for r in results)
        ttt_t = sum(r.test_exact_ttt for r in results)
        no_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact_no_ttt == r.num_test
        )
        icl_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact_icl == r.num_test
        )
        d8_top1_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact_d8_top1 == r.num_test
        )
        d8_top2_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact_d8_top2 == r.num_test
        )
        ttt_solved = sum(
            1 for r in results if r.num_test > 0 and r.test_exact_ttt == r.num_test
        )
        print("-" * len(header))
        print(
            f"  no-TTT, no-ICL:               {no_t:>3}/{total_test}    "
            f"({no_solved}/{len(results)} fully solved)"
        )
        print(
            f"  no-TTT, +ICL:                 {icl_t:>3}/{total_test}    "
            f"({icl_solved}/{len(results)} fully solved)   ← Week 5"
        )
        print(
            f"  no-TTT, D8-Ensemble (top-1):  {d8_top1_t:>3}/{total_test}    "
            f"({d8_top1_solved}/{len(results)} fully solved)   ← Week 6"
        )
        print(
            f"  no-TTT, D8-Ensemble (top-2):  {d8_top2_t:>3}/{total_test}    "
            f"({d8_top2_solved}/{len(results)} fully solved)   ← Week 6 (2 attempts)"
        )
        if not args.skip_ttt:
            print(
                f"  +TTT, no-ICL:                 {ttt_t:>3}/{total_test}    "
                f"({ttt_solved}/{len(results)} fully solved)   ← Week 4"
            )
        print()
        print("Side-by-side TEST acc on 10 tasks:")
        print(f"  Week 2.5 baseline (from-scratch + TTT):    0/10")
        print(f"  Week 4   (pretrained + TTT, no ICL):       {ttt_solved}/10")
        print(f"  Week 5   (pretrained + ICL, no TTT):       {icl_solved}/10")
        print(f"  Week 6a  (pretrained + D8-Ensemble top-1): {d8_top1_solved}/10")
        print(f"  Week 6a  (pretrained + D8-Ensemble top-2): {d8_top2_solved}/10")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
