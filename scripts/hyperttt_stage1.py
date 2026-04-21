"""Hyper-TTT Stage 1: Collect LoRA target weights for HyperNet training.

For each task in the target corpus:
  1. Load the pretrained Volcan checkpoint (dream_wide base model).
  2. Run the standard D8-augmented LoRA rank-16 TTT for 150 steps on the
     task's demos.
  3. Extract the final LoRA A/B weights from each Conv2d in the update MLP.
  4. Save (task_id, demos, flat_lora_weights) as one training example.

These triples are the supervised training data for the HyperNetwork in Stage 2.

Usage:
    python scripts/hyperttt_stage1.py \
        --checkpoint outputs/week8/volcan_dream_wide.pt \
        --corpus data/dream_500 \
        --out data/hyperttt_targets \
        --max-tasks 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.arc import load_dataset  # noqa: E402
from volcan.hyperttt import LoRASchema, infer_lora_schema  # noqa: E402
from volcan.lora import LoRAConv2dAdapter  # noqa: E402
from volcan.moe import MoEUpdateMLP  # noqa: E402
from volcan.pretrain import load_checkpoint  # noqa: E402
from volcan.training_volcan import overfit_volcan_single_task  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_flat_lora(model: VolcanCell, schema: LoRASchema) -> torch.Tensor:
    """Walk the (already-LoRA-attached) model and flatten A+B weights into one vector.

    Matches the ordering used by `LoRASchema.slice_spec()`: for each conv in
    order, write all of A followed by all of B.
    """
    update = model.update
    if isinstance(update, MoEUpdateMLP):
        # Average across experts — gives us a single canonical LoRA for the task.
        adapters_per_expert = []
        for expert in update.experts:
            expert_adapters = [layer for layer in expert if isinstance(layer, LoRAConv2dAdapter)]
            adapters_per_expert.append(expert_adapters)
        # Transpose: adapters_by_layer[i] = list of N expert adapters for layer i
        num_layers = len(adapters_per_expert[0])
        adapters_by_layer = [
            [adapters_per_expert[e][i] for e in range(len(adapters_per_expert))]
            for i in range(num_layers)
        ]
        # Average each layer's A and B across experts.
        adapters = []
        for layer_adapters in adapters_by_layer:
            avg_A = torch.stack([a.lora_A.weight for a in layer_adapters]).mean(dim=0)
            avg_B = torch.stack([a.lora_B.weight for a in layer_adapters]).mean(dim=0)
            adapters.append((avg_A, avg_B))
    else:
        adapters = []
        for layer in update:
            if isinstance(layer, LoRAConv2dAdapter):
                adapters.append((layer.lora_A.weight, layer.lora_B.weight))

    if len(adapters) != len(schema.conv_shapes):
        raise RuntimeError(
            f"found {len(adapters)} LoRA adapters but schema expects "
            f"{len(schema.conv_shapes)}"
        )

    parts = []
    for A, B in adapters:
        # A: (rank, in_ch, 1, 1) → flat
        # B: (out_ch, rank, 1, 1) → flat
        parts.append(A.detach().flatten())
        parts.append(B.detach().flatten())
    return torch.cat(parts).cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyper-TTT Stage 1: collect LoRA targets")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="pretrained Volcan checkpoint (e.g. outputs/week8/volcan_dream_wide.pt)")
    parser.add_argument("--corpus", type=str, required=True,
                        help="corpus directory to collect targets from (e.g. data/dream_500)")
    parser.add_argument("--out", type=str, default="data/hyperttt_targets",
                        help="output directory for target .pt files")
    parser.add_argument("--max-tasks", type=int, default=100,
                        help="maximum number of tasks to process (saves time)")
    parser.add_argument("--ttt-steps", type=int, default=150)
    parser.add_argument("--ttt-lr", type=float, default=2e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    corpus_root = PROJECT_ROOT / args.corpus
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"corpus: {corpus_root}")
    print(f"output: {out_dir}")
    print(f"ttt: {args.ttt_steps} steps, lr {args.ttt_lr}, lora rank {args.lora_rank}")
    print()

    # Load tasks
    tasks = load_dataset(corpus_root, split="training")
    print(f"loaded {len(tasks)} tasks from corpus")

    # Filter to tasks with enough demos
    tasks = [t for t in tasks if t.num_train >= 3]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(f"processing {len(tasks)} tasks (max-tasks={args.max_tasks})")
    print()

    # Build base model (will be reloaded fresh per task)
    cfg = VolcanConfig(mlp_hidden=args.mlp_hidden)
    probe = VolcanCell(cfg).to(device)
    load_checkpoint(probe, PROJECT_ROOT / args.checkpoint, device=device)
    schema = infer_lora_schema(probe, rank=args.lora_rank)
    print(f"LoRA schema: {len(schema.conv_shapes)} convs, total params per task: {schema.total_params:,}")
    print()

    start = time.time()
    n_saved = 0
    n_skipped = 0
    for i, task in enumerate(tasks):
        out_path = out_dir / f"{task.task_id}.pt"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        t0 = time.time()
        # Fresh model per task
        torch.manual_seed(0)
        model = VolcanCell(cfg).to(device)
        load_checkpoint(model, PROJECT_ROOT / args.checkpoint, device=device)

        # Run standard D8 + LoRA TTT
        try:
            overfit_volcan_single_task(
                model, task,
                num_steps=args.ttt_steps,
                phase_a_max=6,
                phase_b_steps=16,
                lr=args.ttt_lr,
                device=device,
                log_every=args.ttt_steps,
                d8_augment=True,
                loo_validation=False,
                use_lora=True,
                lora_rank=args.lora_rank,
                lora_alpha=float(args.lora_rank),
                lora_lr_multiplier=5.0,
            )
        except Exception as e:
            print(f"  [{i:3d}] {task.task_id} FAIL: {type(e).__name__}: {e}")
            continue

        # Extract the trained LoRA weights
        try:
            flat_lora = extract_flat_lora(model, schema)
        except Exception as e:
            print(f"  [{i:3d}] {task.task_id} EXTRACT FAIL: {type(e).__name__}: {e}")
            continue

        # Save: task_id, the demo inputs/outputs, the LoRA target vector
        torch.save({
            "task_id": task.task_id,
            "demo_inputs": [ex.input for ex in task.train],
            "demo_outputs": [ex.output for ex in task.train],
            "lora_weights": flat_lora,
            "schema_conv_shapes": schema.conv_shapes,
            "schema_rank": schema.rank,
        }, out_path)

        elapsed = time.time() - t0
        n_saved += 1
        if n_saved % 10 == 0 or i < 3:
            total = time.time() - start
            rate = n_saved / total if total > 0 else 0
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i:3d}] {task.task_id}: saved ({elapsed:.1f}s)  running rate={rate:.2f} tasks/s, ETA {eta:.0f}s")

    print()
    print(f"done: {n_saved} saved, {n_skipped} skipped, total {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
