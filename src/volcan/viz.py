"""Visualization utilities for Volcan.

Two things:
  1. Plot ARC grids using the canonical ARC color palette.
  2. Plot training curves (loss + accuracy) from a TrainLog.

Uses matplotlib's Agg backend so nothing depends on a GUI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from .arc import Grid, OUTSIDE_TOKEN, VOCAB_SIZE
from .training import TrainLog


# Canonical ARC colors, indices 0-9, plus "outside" at index 10.
# Colors from fchollet/ARC-AGI/apps/testing_interface.html.
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 teal
    "#870C25",  # 9 brown
    "#FFFFFF",  # 10 "outside" (rendered as white so it disappears on the page)
]

_ARC_CMAP = ListedColormap(ARC_COLORS)
_ARC_NORM = BoundaryNorm(boundaries=list(range(VOCAB_SIZE + 1)), ncolors=VOCAB_SIZE)


def plot_grid(
    ax: plt.Axes,
    grid: Grid,
    title: str = "",
) -> None:
    """Render a single ARC grid into an axes."""
    arr = np.array(grid, dtype=np.int64)
    ax.imshow(arr, cmap=_ARC_CMAP, norm=_ARC_NORM, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=10)


def plot_task_prediction(
    task_id: str,
    demos: list[tuple[Grid, Grid, Grid]],
    save_to: str | Path,
) -> None:
    """Save a side-by-side figure of (input, target, prediction) per demo.

    Args:
        task_id: used as the figure title.
        demos: list of (input_grid, target_grid, predicted_grid) tuples.
        save_to: path to save the PNG.
    """
    n = len(demos)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2.2 * n), squeeze=False)
    for i, (inp, tgt, pred) in enumerate(demos):
        plot_grid(axes[i, 0], inp, "input" if i == 0 else "")
        plot_grid(axes[i, 1], tgt, "target" if i == 0 else "")
        correct = _grids_equal(pred, tgt)
        pred_title = "prediction ✓" if correct else "prediction ✗"
        plot_grid(axes[i, 2], pred, pred_title if i == 0 else ("✓" if correct else "✗"))
    fig.suptitle(f"task {task_id}", fontsize=12)
    fig.tight_layout()
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    log: TrainLog,
    save_to: str | Path,
    title: str = "",
) -> None:
    """Save a loss + accuracy training curve to a PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(log.steps, log.losses, color="#0074D9")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_yscale("log")
    ax1.set_title("loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(log.steps, log.accuracies, color="#2ECC40")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cell accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax2.set_title("accuracy")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) and ra == rb for ra, rb in zip(a, b))
