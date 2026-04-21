"""Generate a corpus of synthetic ARC-style tasks (Week 3a — rule-based DSL).

Usage:
    python scripts/generate_synthetic.py --n 100             # quick smoke
    python scripts/generate_synthetic.py --n 5000            # day-1 corpus
    python scripts/generate_synthetic.py --n 100000          # full pretraining corpus

Output: data/synthetic/training/syn_NNNNNN.json — one task per file, ARC native
JSON format. The existing arc.load_dataset can read these directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from volcan.synth import (  # noqa: E402
    GridGenConfig,
    TaskGenConfig,
    generate_corpus,
    save_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Volcan rule-based task generator")
    parser.add_argument("--n", type=int, default=100, help="number of tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-demos", type=int, default=4)
    parser.add_argument("--num-tests", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--p-compose", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--require-square", action="store_true",
                        help="if set, every grid is square (enables Rotate90/270, Transpose)")
    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic/training",
        help="output directory (relative to project root)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="after generating, print the rule histogram and a few sample tasks",
    )
    args = parser.parse_args()

    cfg = TaskGenConfig(
        num_demos=args.num_demos,
        num_tests=args.num_tests,
        max_rule_depth=args.max_depth,
        p_compose=args.p_compose,
        grid=GridGenConfig(
            min_size=args.min_size,
            max_size=args.max_size,
            require_square=args.require_square,
        ),
    )

    print(f"generating {args.n} synthetic tasks (seed={args.seed})")
    print(
        f"  rule depth ≤ {args.max_depth}, p_compose={args.p_compose}, "
        f"grid {args.min_size}-{args.max_size}{'×square' if args.require_square else ''}, "
        f"{args.num_demos} demos + {args.num_tests} test"
    )
    print()

    t0 = time.time()
    tasks = generate_corpus(
        args.n,
        cfg=cfg,
        seed=args.seed,
        progress_every=max(args.n // 10, 1),
    )
    elapsed = time.time() - t0

    out_dir = PROJECT_ROOT / args.out
    save_corpus(tasks, out_dir)
    print()
    print(f"saved {len(tasks)} tasks → {out_dir} in {elapsed:.1f}s")
    print(f"  rate: {len(tasks)/elapsed:.0f} tasks/sec")

    if args.inspect:
        print()
        print("rule histogram (top 25):")
        rule_counter = Counter(getattr(t, "rule_name", "?") for t in tasks)
        for rule, count in rule_counter.most_common(25):
            print(f"  {count:>5}  {rule}")
        print(f"  …{len(rule_counter)} distinct rules total")
        print()
        print("sample tasks:")
        for task in tasks[:3]:
            rule_name = getattr(task, "rule_name", "?")
            print(f"  [{task.task_id}] rule={rule_name}")
            for i, ex in enumerate(task.train[:2]):
                print(f"    demo {i}: in={ex.input}  out={ex.output}")
            for i, ex in enumerate(task.test):
                print(f"    test {i}: in={ex.input}  out={ex.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
