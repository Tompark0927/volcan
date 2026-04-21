"""Synthetic task generation (Week 3a — rule-based, no LLM).

Pipeline:
  1. Sample a random rule from the DSL (`dsl.random_rule`).
  2. Sample N random input grids.
  3. Apply the rule to each input to produce demo input/output pairs.
  4. Verify the task is "interesting" (not trivial, not degenerate, etc).
  5. Wrap as an `arc.Task` and optionally save as JSON in ARC's native format.

The output is a corpus of synthetic tasks Volcan can pretrain on. Each task
follows the same JSON schema as the public ARC datasets, so the existing
`arc.load_task` and `arc.load_dataset` loaders work without modification.
"""

from __future__ import annotations

import json
import random as _random
from dataclasses import dataclass
from pathlib import Path

from .arc import Example, Grid, NUM_COLORS, Task
from .dsl import (
    BACKGROUND,
    Primitive,
    random_rule,
)


# -----------------------------------------------------------------------------
# Random grid generation
# -----------------------------------------------------------------------------


@dataclass
class GridGenConfig:
    """Parameters controlling random input grid generation."""

    # Default max_size = 7 so that ScaleX2's doubled output (14×14) still fits
    # comfortably in the 30×30 padded canvas with room for the rest of the
    # task. Bump this up if you disable shape-changing primitives.
    min_size: int = 3
    max_size: int = 7
    min_palette: int = 2  # number of distinct non-background colors
    max_palette: int = 4
    min_density: float = 0.3  # fraction of cells that are non-background
    max_density: float = 0.7
    require_square: bool = False


def random_grid(rng: _random.Random, cfg: GridGenConfig) -> Grid:
    """Sample a random grid according to `cfg`."""
    if cfg.require_square:
        h = w = rng.randint(cfg.min_size, cfg.max_size)
    else:
        h = rng.randint(cfg.min_size, cfg.max_size)
        w = rng.randint(cfg.min_size, cfg.max_size)

    palette_size = rng.randint(cfg.min_palette, cfg.max_palette)
    palette = rng.sample(range(1, NUM_COLORS), palette_size)
    density = rng.uniform(cfg.min_density, cfg.max_density)

    grid = []
    for _ in range(h):
        row = []
        for _ in range(w):
            if rng.random() < density:
                row.append(rng.choice(palette))
            else:
                row.append(BACKGROUND)
        grid.append(row)
    return grid


# -----------------------------------------------------------------------------
# Task generation
# -----------------------------------------------------------------------------


@dataclass
class TaskGenConfig:
    """Top-level config for synthetic task generation."""

    num_demos: int = 4
    num_tests: int = 1
    max_attempts_per_task: int = 50
    max_rule_depth: int = 2
    p_compose: float = 0.5
    allow_shape_changing: bool = True
    p_shape_changing_tail: float = 0.15
    grid: GridGenConfig = None  # type: ignore  # set in __post_init__

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = GridGenConfig()


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) and ra == rb for ra, rb in zip(a, b))


MAX_OUTPUT_SIZE = 30  # match the Volcan padded canvas


def _grid_is_valid(g: Grid) -> bool:
    if not g or not g[0]:
        return False
    h, w = len(g), len(g[0])
    if h > MAX_OUTPUT_SIZE or w > MAX_OUTPUT_SIZE:
        return False
    if any(len(row) != w for row in g):
        return False
    return all(0 <= c < NUM_COLORS for row in g for c in row)


def _grid_is_blank(g: Grid) -> bool:
    return all(c == BACKGROUND for row in g for c in row)


def _all_demo_outputs_equal(examples: list[Example]) -> bool:
    """Reject tasks where every demo output is the same grid (degenerate)."""
    if len(examples) < 2:
        return False
    first = examples[0].output
    return all(_grids_equal(ex.output, first) for ex in examples[1:])


def try_generate_task(
    rule: Primitive,
    rng: _random.Random,
    cfg: TaskGenConfig,
    task_id: str,
) -> Task | None:
    """Try once to build a task from a given rule. Returns None on rejection."""
    examples: list[Example] = []
    test_examples: list[Example] = []

    for slot in range(cfg.num_demos + cfg.num_tests):
        # Multiple sub-attempts per slot to find a non-degenerate input grid.
        success = False
        for _ in range(20):
            inp = random_grid(rng, cfg.grid)
            try:
                out = rule.apply(inp)
            except Exception:
                continue
            if not _grid_is_valid(out):
                continue
            if _grids_equal(inp, out):
                continue  # rule was a no-op on this input
            if _grid_is_blank(out):
                continue  # rule erased everything
            ex = Example(input=inp, output=out)
            if slot < cfg.num_demos:
                examples.append(ex)
            else:
                test_examples.append(ex)
            success = True
            break
        if not success:
            return None

    # Reject if all demo outputs are identical — too easy / degenerate.
    if _all_demo_outputs_equal(examples):
        return None

    return Task(task_id=task_id, train=examples, test=test_examples)


def generate_task(
    rng: _random.Random,
    cfg: TaskGenConfig,
    task_id: str,
) -> Task | None:
    """Sample a rule and try to build a task from it. Returns None if all attempts fail."""
    for _ in range(cfg.max_attempts_per_task):
        rule = random_rule(
            rng,
            max_depth=cfg.max_rule_depth,
            p_compose=cfg.p_compose,
            allow_square_only=cfg.grid.require_square,
            allow_shape_changing=cfg.allow_shape_changing,
            p_shape_changing_tail=cfg.p_shape_changing_tail,
        )
        task = try_generate_task(rule, rng, cfg, task_id)
        if task is not None:
            # Stash the rule name on the task for debugging / analysis.
            task.rule_name = rule.name  # type: ignore[attr-defined]
            return task
    return None


def generate_corpus(
    n_tasks: int,
    *,
    cfg: TaskGenConfig | None = None,
    seed: int = 0,
    progress_every: int = 100,
    task_id_prefix: str = "syn",
) -> list[Task]:
    """Generate a corpus of `n_tasks` synthetic tasks."""
    cfg = cfg or TaskGenConfig()
    rng = _random.Random(seed)
    tasks: list[Task] = []
    attempts = 0
    while len(tasks) < n_tasks:
        task_id = f"{task_id_prefix}_{len(tasks):06d}"
        task = generate_task(rng, cfg, task_id)
        attempts += 1
        if task is not None:
            tasks.append(task)
            if progress_every and len(tasks) % progress_every == 0:
                print(f"  generated {len(tasks)}/{n_tasks} (attempts: {attempts})")
    return tasks


# -----------------------------------------------------------------------------
# Save / load — ARC native JSON format
# -----------------------------------------------------------------------------


def save_task(task: Task, dir_path: str | Path) -> Path:
    """Save a task as ARC-native JSON."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / f"{task.task_id}.json"
    payload = {
        "train": [{"input": ex.input, "output": ex.output} for ex in task.train],
        "test": [{"input": ex.input, "output": ex.output} for ex in task.test],
    }
    rule_name = getattr(task, "rule_name", None)
    if rule_name is not None:
        payload["_rule"] = rule_name  # extension field for our own analysis
    with out_path.open("w") as f:
        json.dump(payload, f)
    return out_path


def save_corpus(tasks: list[Task], dir_path: str | Path) -> int:
    """Save a list of tasks as individual JSON files. Returns the count saved."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        save_task(task, dir_path)
    return len(tasks)
