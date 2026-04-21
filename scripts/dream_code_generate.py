"""Week 8 v2 — Hybrid α+δ dream pipeline (CEO's synthesis).

Workflow per attempt:
  1. LLM (via Ollama) writes a Python `transform(grid)` function.
  2. We parse, compile, and sanity-run the function.
  3. We generate 4 demo pairs + 1 test pair by running the function on random
     input grids — consistency guaranteed by construction.
  4. Leave-one-out filter: Volcan trains on 3 demos, must predict the 4th
     (held out). If yes → accept. If no → reject (too hard for current weights).
  5. Accepted tasks land in the Week 8 Gold Corpus.

Usage:
    python scripts/dream_code_generate.py --n 5                          # smoke
    python scripts/dream_code_generate.py --n 50 --model qwen2.5:7b      # day-1
"""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.code_dreamer import dream_one_code_task, save_code_task  # noqa: E402
from volcan.code_filter import leave_one_out_filter  # noqa: E402
from volcan.dream_filter import volcan_overfit_filter  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Code-dreamer + LOO filter")
    parser.add_argument("--n", type=int, default=5, help="target number of accepted tasks")
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="memorize",
        choices=["memorize", "loo"],
        help=(
            "memorize = accept tasks Volcan can overfit (high yield, needs consistency "
            "from code gen); loo = strict leave-one-out (low yield but curriculum-aligned)"
        ),
    )
    parser.add_argument("--filter-steps", type=int, default=120)
    parser.add_argument("--filter-budget", type=float, default=30.0)
    parser.add_argument("--num-train-demos", type=int, default=3)
    parser.add_argument("--num-llm-demos", type=int, default=4,
                        help="total demos to request from each code function (3 train + 1 LOO)")
    parser.add_argument("--num-test", type=int, default=1,
                        help="held-out test inputs per task (after LOO filter passes)")
    parser.add_argument(
        "--out",
        type=str,
        default="data/dream_code/training",
        help="output directory for accepted tasks",
    )
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    max_attempts = args.max_attempts or (10 * args.n)
    out_dir = PROJECT_ROOT / args.out

    print(f"device: {device}")
    print(f"model: {args.model}  temperature: {args.temperature}")
    print(f"target: {args.n} accepted tasks  (max {max_attempts} attempts)")
    print(f"filter mode: {args.filter_mode}")
    if args.filter_mode == "loo":
        print(f"demos per task: {args.num_train_demos} train + 1 LOO + {args.num_test} test")
    else:
        print(f"demos per task: {args.num_train_demos + 1} memorization + {args.num_test} test")
    print(f"filter: {args.filter_steps} steps, {args.filter_budget}s budget")
    print(f"output: {out_dir}")
    print()

    accepted = 0
    rejected_compile = 0
    rejected_filter = 0
    llm_total_sec = 0.0
    filter_total_sec = 0.0

    rng = _random.Random(args.seed)

    t_run_start = time.time()
    for attempt in range(1, max_attempts + 1):
        if accepted >= args.n:
            break

        task_id = f"dream_code_{accepted:06d}"

        # Memorize mode: request num_train_demos + 1 (all used for training) + num_test.
        # LOO mode: request num_train_demos + 1 (last one held out) + num_test.
        # Both end up calling dream_one_code_task with the same shape; the filter
        # uses the demos differently.
        result = dream_one_code_task(
            task_id,
            num_train=args.num_train_demos + 1,
            num_test=args.num_test,
            model=args.model,
            temperature=args.temperature,
            rng=rng,
        )
        if result is None:
            rejected_compile += 1
            print(f"  [{attempt:4d}] REJECT (compile/generate)")
            continue

        llm_total_sec += result.llm_sec

        if args.filter_mode == "loo":
            filt_loo = leave_one_out_filter(
                result.task,
                num_train_demos=args.num_train_demos,
                max_steps=args.filter_steps,
                time_budget_sec=args.filter_budget,
                device=device,
                seed=args.seed + attempt,
            )
            filter_total_sec += filt_loo.elapsed_sec
            passed = filt_loo.passed
            diag = (
                f"memorized={filt_loo.demos_memorized}/{args.num_train_demos}  "
                f"LOO={'✓' if filt_loo.held_out_correct else '✗'}"
            )
            elapsed = filt_loo.elapsed_sec
            reason = filt_loo.reason
        else:
            filt_mem = volcan_overfit_filter(
                result.task,
                max_steps=args.filter_steps,
                time_budget_sec=args.filter_budget,
                acc_threshold=1.0,
                exact_match_ratio=1.0,
                device=device,
                seed=args.seed + attempt,
            )
            filter_total_sec += filt_mem.elapsed_sec
            passed = filt_mem.passed
            diag = f"memorized={filt_mem.demos_exact}/{filt_mem.num_demos}"
            elapsed = filt_mem.elapsed_sec
            reason = filt_mem.reason

        if passed:
            save_code_task(result, out_dir)
            accepted += 1
            print(
                f"  [{attempt:4d}] ACCEPT #{accepted:03d}  "
                f"llm={result.llm_sec:4.1f}s filter={elapsed:4.1f}s  {diag}  "
                f"rule='{result.rule_seed[:55]}'"
            )
        else:
            rejected_filter += 1
            print(
                f"  [{attempt:4d}] REJECT  "
                f"llm={result.llm_sec:4.1f}s filter={elapsed:4.1f}s  {diag}  {reason}"
            )

    total_elapsed = time.time() - t_run_start

    print()
    print("=" * 80)
    print(f"Week 8 v2 (code-dreamer, filter={args.filter_mode}) — summary")
    print("=" * 80)
    print(f"  target:                                  {args.n}")
    print(f"  accepted:                                {accepted}")
    print(f"  rejected (LLM / compile / parse):        {rejected_compile}")
    print(f"  rejected ({args.filter_mode} filter failed):{' ' * max(1, 14 - len(args.filter_mode))}{rejected_filter}")
    print(f"  total attempts:                          {attempt}")
    print(f"  yield rate:                              {accepted/max(attempt,1)*100:.1f}%")
    print()
    print(f"  LLM generation total:                    {llm_total_sec:.0f}s")
    print(f"  Volcan filter total:                     {filter_total_sec:.0f}s")
    print(f"  wall-clock total:                        {total_elapsed:.0f}s")
    print()
    if accepted > 0:
        avg_accepted_sec = total_elapsed / accepted
        print(f"  sec per accepted task:                   {avg_accepted_sec:.1f}s")
        # Scale projection for 10K tasks
        proj_10k_hours = avg_accepted_sec * 10000 / 3600
        print(f"  projected 10K-task wall time:            {proj_10k_hours:.1f}h")
        proj_500_hours = avg_accepted_sec * 500 / 3600
        print(f"  projected  500-task wall time:           {proj_500_hours:.1f}h")

    return 0 if accepted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
