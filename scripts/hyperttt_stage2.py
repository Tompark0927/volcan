"""Hyper-TTT Stage 2: train the HyperNetwork on demo → LoRA-weight pairs.

Loads the .pt files produced by stage 1 (each one: task's demos + final LoRA
weights from a fresh TTT run). Trains a HyperNetwork to regress the LoRA
weights from the demos, so at test time we can initialize LoRA from the
hypernet's prediction instead of from Kaiming/zeros.

Training objective: MSE between predicted and target LoRA weights, weighted
so A and B matrices contribute equally regardless of their dimensional
mismatch (B is typically larger in scale since it aggregates from rank to
out_channels).

Usage:
    python scripts/hyperttt_stage2.py \
        --base-checkpoint outputs/week8/volcan_dream_wide.pt \
        --targets data/hyperttt_targets \
        --out outputs/week8/hypernet.pt \
        --num-epochs 200
"""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.arc import grid_to_onehot  # noqa: E402
from volcan.hyperttt import HyperNetwork, LoRASchema, infer_lora_schema  # noqa: E402
from volcan.pretrain import load_checkpoint  # noqa: E402
from volcan.volcan_cell import VolcanCell, VolcanConfig  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_training_example(
    target_dict: dict,
    num_demos: int,
    pad_to: int = 30,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (demo_inputs, demo_outputs, query_input, lora_target) from a saved .pt.

    The hypernet conditions on num_demos demos + 1 query input; we use demo 0
    as the query and demos 1..num_demos as the context. The target LoRA is
    the one saved during stage 1.
    """
    inputs = target_dict["demo_inputs"]
    outputs = target_dict["demo_outputs"]
    lora_target = target_dict["lora_weights"].to(device)

    if len(inputs) < num_demos + 1:
        raise ValueError(f"task has {len(inputs)} demos, need {num_demos + 1}")

    # First demo is the query, next num_demos are the context
    query_in = grid_to_onehot(inputs[0], pad_to=pad_to).unsqueeze(0).to(device)
    context_in = torch.stack(
        [grid_to_onehot(inp, pad_to=pad_to) for inp in inputs[1 : num_demos + 1]]
    ).unsqueeze(0).to(device)
    context_out = torch.stack(
        [grid_to_onehot(out, pad_to=pad_to) for out in outputs[1 : num_demos + 1]]
    ).unsqueeze(0).to(device)

    return context_in, context_out, query_in, lora_target


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyper-TTT Stage 2: train the HyperNetwork")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="Volcan base checkpoint used to encode demos (frozen)")
    parser.add_argument("--targets", type=str, required=True,
                        help="directory of .pt target files from stage 1")
    parser.add_argument("--out", type=str, required=True,
                        help="output path for hypernet checkpoint")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--task-embed-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    targets_dir = PROJECT_ROOT / args.targets
    base_ckpt = PROJECT_ROOT / args.base_checkpoint
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"base checkpoint: {base_ckpt}")
    print(f"targets: {targets_dir}")
    print(f"output: {out_path}")

    # Load all targets into memory
    target_files = sorted(targets_dir.glob("*.pt"))
    print(f"loaded {len(target_files)} target files")
    if not target_files:
        print("NO TARGETS FOUND — run stage 1 first")
        return 1

    all_targets = [torch.load(p, map_location="cpu", weights_only=False) for p in target_files]
    # Filter to ones with enough demos
    all_targets = [t for t in all_targets if len(t["demo_inputs"]) >= args.num_demos + 1]
    print(f"  {len(all_targets)} usable (>= {args.num_demos + 1} demos)")

    # Train/val split
    rng = _random.Random(args.seed)
    rng.shuffle(all_targets)
    n_val = max(1, int(len(all_targets) * args.val_fraction))
    val_targets = all_targets[:n_val]
    train_targets = all_targets[n_val:]
    print(f"  split: {len(train_targets)} train, {len(val_targets)} val")

    # Build base Volcan (frozen) + hypernet
    cfg = VolcanConfig(mlp_hidden=args.mlp_hidden)
    base_model = VolcanCell(cfg).to(device)
    load_checkpoint(base_model, base_ckpt, device=device)
    base_model.eval()
    schema = infer_lora_schema(base_model, rank=args.lora_rank)
    print(f"schema: {len(schema.conv_shapes)} convs, {schema.total_params} LoRA params")

    hypernet = HyperNetwork(
        base_model=base_model,
        schema=schema,
        task_embed_dim=args.task_embed_dim,
        decoder_hidden=args.decoder_hidden,
        freeze_base=True,
    ).to(device)
    print(f"hypernet trainable params: {hypernet.num_params():,}")

    # Compute target normalization (mean/std) so we can train a unit-scale regressor
    target_stack = torch.stack([t["lora_weights"] for t in train_targets])
    target_mean = target_stack.mean(dim=0).to(device)
    target_std = target_stack.std(dim=0).clamp_min(1e-6).to(device)
    print(f"target tensor: shape={target_stack.shape}, mean norm={target_mean.norm():.3f}, "
          f"std norm={target_std.norm():.3f}")

    # Optimizer — only hypernet params train (base is frozen)
    trainable = [p for p in hypernet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print()
    t0 = time.time()
    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        hypernet.train()
        epoch_loss = 0.0
        n_batches = 0
        rng.shuffle(train_targets)
        for i in range(0, len(train_targets), args.batch_size):
            batch = train_targets[i : i + args.batch_size]
            # Build batched inputs
            batch_ctx_in = []
            batch_ctx_out = []
            batch_q_in = []
            batch_lora_target = []
            for t in batch:
                ctx_in, ctx_out, q_in, lt = build_training_example(
                    t, num_demos=args.num_demos, device=device
                )
                batch_ctx_in.append(ctx_in.squeeze(0))
                batch_ctx_out.append(ctx_out.squeeze(0))
                batch_q_in.append(q_in.squeeze(0))
                batch_lora_target.append(lt)
            ctx_in = torch.stack(batch_ctx_in)
            ctx_out = torch.stack(batch_ctx_out)
            q_in = torch.stack(batch_q_in)
            target = torch.stack(batch_lora_target)
            # Normalize target for stable regression
            target_norm = (target - target_mean) / target_std

            # Predict in normalized space
            pred_norm = hypernet(ctx_in, ctx_out, q_in)
            loss = F.mse_loss(pred_norm, target_norm)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        scheduler.step()

        # Val
        hypernet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_targets:
                ctx_in, ctx_out, q_in, lt = build_training_example(
                    t, num_demos=args.num_demos, device=device
                )
                target_norm = (lt - target_mean) / target_std
                pred_norm = hypernet(ctx_in, ctx_out, q_in).squeeze(0)
                val_loss += F.mse_loss(pred_norm, target_norm).item()
        val_loss /= max(len(val_targets), 1)

        if epoch == 1 or epoch % max(1, args.num_epochs // 20) == 0 or epoch == args.num_epochs:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}/{args.num_epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "hypernet_state_dict": hypernet.state_dict(),
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
                "schema_conv_shapes": schema.conv_shapes,
                "schema_rank": schema.rank,
                "num_demos": args.num_demos,
                "task_embed_dim": args.task_embed_dim,
                "decoder_hidden": args.decoder_hidden,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_path)

    print()
    print(f"best val_loss = {best_val_loss:.4f}")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
