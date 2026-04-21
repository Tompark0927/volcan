"""ARC task loading and grid utilities.

An ARC task is a JSON file with the shape:

    {
      "train": [{"input": [[...]], "output": [[...]]}, ...],
      "test":  [{"input": [[...]], "output": [[...]]}]
    }

Grids are 2D integer arrays, values in [0, 9]. Sizes vary from 1×1 up to 30×30.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

NUM_COLORS = 10
OUTSIDE_TOKEN = 10  # Volcan's 11th "color" for padding (see architecture.md §10)
VOCAB_SIZE = NUM_COLORS + 1
MAX_GRID_SIZE = 30


Grid = list[list[int]]  # 2D list of ints, arbitrary size


@dataclass
class Example:
    """One (input, output) demonstration pair."""

    input: Grid
    output: Grid


@dataclass
class Task:
    """One ARC task: some train (demo) pairs + one or more test inputs to solve."""

    task_id: str
    train: list[Example]
    test: list[Example]

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)


def load_task(path: str | Path) -> Task:
    """Load a single ARC task from a .json file."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    train = [Example(input=p["input"], output=p["output"]) for p in data["train"]]
    test = [Example(input=p["input"], output=p.get("output", [])) for p in data["test"]]
    return Task(task_id=path.stem, train=train, test=test)


def load_dataset(
    root: str | Path,
    split: Literal["training", "evaluation"] = "training",
) -> list[Task]:
    """Load all tasks under an ARC data root directory.

    Auto-detects the two layouts we actually use:
      - ARC-AGI/data/{training,evaluation}/*.json
      - ARC-AGI-2/data/{training,evaluation}/*.json
    """
    root = Path(root)
    candidates = [root / "data" / split, root / split]
    for d in candidates:
        if d.is_dir():
            return sorted(
                (load_task(p) for p in d.glob("*.json")),
                key=lambda t: t.task_id,
            )
    raise FileNotFoundError(
        f"No ARC {split} split found under {root} "
        f"(tried: {', '.join(str(c) for c in candidates)})"
    )


# -----------------------------------------------------------------------------
# Grid ↔ tensor conversion
# -----------------------------------------------------------------------------


def grid_to_tensor(
    grid: Grid,
    pad_to: int | None = MAX_GRID_SIZE,
    pad_token: int = OUTSIDE_TOKEN,
) -> torch.Tensor:
    """Convert an ARC grid (2D int list) to a long tensor of shape (H, W).

    If `pad_to` is given, the grid is padded to (pad_to, pad_to) with `pad_token`
    (default = the OUTSIDE token). This gives us a fixed-shape canvas for
    variable-size tasks, as specified in architecture.md §10.
    """
    if not grid or not grid[0]:
        raise ValueError(f"empty grid: {grid!r}")
    h, w = len(grid), len(grid[0])
    if pad_to is None:
        return torch.tensor(grid, dtype=torch.long)
    if h > pad_to or w > pad_to:
        raise ValueError(
            f"grid {h}×{w} exceeds pad_to={pad_to}; should not happen for ARC ({MAX_GRID_SIZE} max)"
        )
    out = torch.full((pad_to, pad_to), pad_token, dtype=torch.long)
    out[:h, :w] = torch.tensor(grid, dtype=torch.long)
    return out


def grid_to_onehot(
    grid: Grid,
    pad_to: int | None = MAX_GRID_SIZE,
    pad_token: int = OUTSIDE_TOKEN,
    vocab_size: int = VOCAB_SIZE,
) -> torch.Tensor:
    """Convert an ARC grid to a one-hot tensor of shape (vocab_size, H, W)."""
    idx = grid_to_tensor(grid, pad_to=pad_to, pad_token=pad_token)
    return torch.nn.functional.one_hot(idx, num_classes=vocab_size).permute(2, 0, 1).float()


def tensor_to_grid(
    tensor: torch.Tensor,
    pad_token: int = OUTSIDE_TOKEN,
) -> Grid:
    """Convert a (H, W) long tensor back to a 2D list, cropping the pad region.

    The "real" grid is the tightest axis-aligned bounding box of non-pad cells.
    """
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(tensor.shape)}")
    mask = tensor != pad_token
    if not mask.any():
        return [[0]]
    rows = torch.where(mask.any(dim=1))[0]
    cols = torch.where(mask.any(dim=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return tensor[r0:r1, c0:c1].tolist()


def onehot_to_grid(
    onehot: torch.Tensor,
    pad_token: int = OUTSIDE_TOKEN,
) -> Grid:
    """Convert a (vocab, H, W) one-hot / logit tensor back to a 2D list."""
    if onehot.ndim != 3:
        raise ValueError(f"expected 3D tensor (C, H, W), got shape {tuple(onehot.shape)}")
    idx = onehot.argmax(dim=0)
    return tensor_to_grid(idx, pad_token=pad_token)


def grids_equal(a: Grid, b: Grid) -> bool:
    """Exact ARC match: same dimensions and same cells."""
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) and ra == rb for ra, rb in zip(a, b))
