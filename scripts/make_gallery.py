"""Render high-resolution gallery images for the Kaggle Paper Track submission.

For each of the three passing tasks (25d8a9c8, b1948b0a, 32597951), render
a 1280x720 PNG showing the train demos and the test input. This is well
above Kaggle's 640x360 gallery minimum.

Output: paper/gallery_{task_id}.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARC_PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#FFFFFF",  # 10 = outside/pad
]
CMAP = ListedColormap(ARC_PALETTE)

# Task metadata for captions
PASSING = [
    ("25d8a9c8", "row-recolor"),
    ("b1948b0a", "background-fill"),
    ("32597951", "overlay-composite"),
]


def load_task(task_id: str):
    p = PROJECT_ROOT / "data" / "ARC-AGI-2" / "data" / "training" / f"{task_id}.json"
    return json.loads(p.read_text())


def draw_grid(ax, grid, title=None, title_size=10):
    g = np.array(grid)
    ax.imshow(g, cmap=CMAP, vmin=0, vmax=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444"); spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=title_size, pad=4)


def render_task(task_id: str, category: str) -> None:
    task = load_task(task_id)
    n_train = len(task["train"])
    n_test = len(task["test"])
    n_cols = n_train + n_test

    # 12.8 x 7.2 inches at 100 DPI = 1280x720 px
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")

    # Big title bar at the top
    fig.suptitle(
        f"Volcan passes ARC-AGI-2 task {task_id} ({category})",
        fontsize=18, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.90,
        "Top row: input grids. Bottom row: target output grids. "
        f"{n_train} train demos + {n_test} test.",
        fontsize=11, ha="center", color="#444",
    )

    # Grid of subplots: 2 rows (input, output) x n_cols
    grid_top = 0.78
    grid_bottom = 0.10
    grid_left = 0.04
    grid_right = 0.96
    col_w = (grid_right - grid_left) / n_cols
    row_h = (grid_top - grid_bottom) / 2

    for i in range(n_cols):
        if i < n_train:
            pair = task["train"][i]
            col_label = f"Demo {i+1}"
        else:
            pair = task["test"][i - n_train]
            col_label = f"TEST {i - n_train + 1}"

        inp = pair["input"]
        out = pair["output"]

        # Input (top row)
        ax_in = fig.add_axes([
            grid_left + i * col_w + 0.005,
            grid_bottom + row_h + 0.02,
            col_w - 0.01,
            row_h - 0.04,
        ])
        draw_grid(ax_in, inp, title=col_label, title_size=11)

        # Output (bottom row)
        ax_out = fig.add_axes([
            grid_left + i * col_w + 0.005,
            grid_bottom + 0.01,
            col_w - 0.01,
            row_h - 0.04,
        ])
        draw_grid(ax_out, out)

    # Row labels on the left
    fig.text(0.01, grid_bottom + row_h * 1.5, "IN", fontsize=12,
             fontweight="bold", color="#666", rotation=90, va="center")
    fig.text(0.01, grid_bottom + row_h * 0.5, "OUT", fontsize=12,
             fontweight="bold", color="#666", rotation=90, va="center")

    # Footer
    fig.text(
        0.5, 0.03,
        "Volcan (111K params) · D8-augmented LoRA rank-16 TTT · "
        "github.com/Tompark0927/volcan",
        fontsize=9, ha="center", color="#888", style="italic",
    )

    out_path = PROJECT_ROOT / "paper" / f"gallery_{task_id}.png"
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> int:
    for task_id, category in PASSING:
        try:
            render_task(task_id, category)
        except Exception as e:
            print(f"  FAIL {task_id}: {type(e).__name__}: {e}", file=sys.stderr)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
