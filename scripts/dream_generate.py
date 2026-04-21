"""Week 8 — Dream pipeline: LLM generates, Volcan filters.

Workflow:
  1. Local LLM (via Ollama) generates a candidate ARC task as JSON
  2. Parse + validate the JSON schema
  3. Run Volcan overfit filter on it — does a fresh Volcan hit 100% demo
     accuracy within the time budget?
  4. If yes → save to Gold Dataset. If no → discard and try again.
  5. Repeat until N valid tasks or time budget exhausted.

Usage:
    python scripts/dream_generate.py --n 20                    # smoke test (20 tasks)
    python scripts/dream_generate.py --n 100 --model qwen2.5:7b
    python scripts/dream_generate.py --n 1000 --model gpt-oss:20b
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

from volcan.dream import generate_one_task, save_dream_task  # noqa: E402
from volcan.dream_filter import volcan_overfit_filter  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dream pipeline: LLM → filter → Gold")
    parser.add_argument("--n", type=int, default=20, help="target number of valid tasks")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name (qwen2.5:3b / qwen2.5:7b / gpt-oss:20b)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--filter-steps", type=int, default=100)
    parser.add_argument("--filter-budget", type=float, default=60.0,
                        help="seconds per task for the Volcan overfit filter")
    parser.add_argument(
        "--out",
        type=str,
        default="data/dream/training",
        help="output directory for accepted tasks",
    )
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="hard cap on generate attempts (default: 10×n)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    max_attempts = args.max_attempts or (10 * args.n)
    out_dir = PROJECT_ROOT / args.out

    print(f"device: {device}")
    print(f"model: {args.model}  temperature: {args.temperature}")
    print(f"target: {args.n} valid tasks (max {max_attempts} generation attempts)")
    print(f"filter: {args.filter_steps} steps, {args.filter_budget}s budget per task")
    print(f"output: {out_dir}")
    print()

    accepted = 0
    rejected_parse = 0
    rejected_filter = 0
    llm_total_sec = 0.0
    filter_total_sec = 0.0

    t_run_start = time.time()
    for attempt in range(1, max_attempts + 1):
        if accepted >= args.n:
            break

        task_id = f"dream_{accepted:06d}"
        gen_result = generate_one_task(
            task_id,
            model=args.model,
            temperature=args.temperature,
            max_retries=2,
        )
        if gen_result is None:
            rejected_parse += 1
            print(f"  [{attempt:4d}] REJECT (parse failed)")
            continue

        task, rule, llm_sec = gen_result
        llm_total_sec += llm_sec

        filt = volcan_overfit_filter(
            task,
            max_steps=args.filter_steps,
            time_budget_sec=args.filter_budget,
            device=device,
            seed=args.seed + accepted,
        )
        filter_total_sec += filt.elapsed_sec

        if filt.passed:
            save_dream_task(task, rule, out_dir)
            accepted += 1
            print(
                f"  [{attempt:4d}] ACCEPT #{accepted:03d}  "
                f"llm={llm_sec:4.1f}s filter={filt.elapsed_sec:4.1f}s  "
                f"acc={filt.final_content_acc*100:5.1f}% exact={filt.demos_exact}/{filt.num_demos}  "
                f"rule='{rule[:60]}'"
            )
        else:
            rejected_filter += 1
            print(
                f"  [{attempt:4d}] REJECT  "
                f"llm={llm_sec:4.1f}s filter={filt.elapsed_sec:4.1f}s  "
                f"{filt.reason}"
            )

    total_elapsed = time.time() - t_run_start

    print()
    print("=" * 72)
    print("Week 8 dream pipeline summary")
    print("=" * 72)
    print(f"  target:                        {args.n}")
    print(f"  accepted:                      {accepted}")
    print(f"  rejected (parse / schema):     {rejected_parse}")
    print(f"  rejected (filter failed):      {rejected_filter}")
    print(f"  total attempts:                {attempt}")
    print(f"  yield rate:                    {accepted/max(attempt,1)*100:.1f}%")
    print()
    print(f"  LLM generation total:          {llm_total_sec:.0f}s")
    print(f"  Volcan filter total:           {filter_total_sec:.0f}s")
    print(f"  wall-clock total:              {total_elapsed:.0f}s")
    print()
    if accepted > 0:
        avg_accepted_sec = total_elapsed / accepted
        print(f"  sec per accepted task:         {avg_accepted_sec:.1f}s")
        # Scale projection for 10K tasks
        proj_10k_sec = avg_accepted_sec * 10000
        print(f"  projected 10K tasks wall time: {proj_10k_sec/3600:.1f}h")

    return 0 if accepted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
