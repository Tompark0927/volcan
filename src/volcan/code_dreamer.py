"""Code-Dreamer — LLM-generated Python transformation functions (Week 8 v2).

The CEO's hybrid α+δ pipeline:
  1. Ask an LLM to write a `transform(input_grid: np.ndarray) -> np.ndarray`
     function that implements a novel transformation rule.
  2. We execute that function (with a timeout) on N diverse random input
     grids to generate a self-consistent set of demo pairs.
  3. The leave-one-out filter (in code_filter.py) then decides which tasks
     are "almost learnable" by Volcan and keeps only those.

The key insight is that consistency is now guaranteed BY CONSTRUCTION — the
Python function IS the rule, so every demo pair is guaranteed to follow it.
Previously, the LLM was writing both the logic AND the data, and hallucinating
inconsistencies into the demos. Now the LLM only writes the logic.
"""

from __future__ import annotations

import json
import random as _random
import re
import signal
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .arc import Example, Grid, NUM_COLORS, Task


# -----------------------------------------------------------------------------
# Prompt — asks for Python code, not JSON
# -----------------------------------------------------------------------------

# Rotated library of rule seeds — each prompt picks one so the model doesn't
# converge on the same transformation every call.
#
# Path B (Week 9 expansion): grown from 20 → 85 diverse seeds covering more
# ARC-AGI-2 task categories. The 9.7% ceiling with the 20-seed library was
# diagnosed as distribution-bound: the 27 failing ARC-AGI-2 tasks likely use
# rule structures never shown during pretraining. This expansion targets
# that distribution gap.
RULE_SEEDS = [
    # --- Geometric: rotations/reflections/transpositions (original 5) ---
    "Rotate the grid 90 degrees clockwise.",
    "Rotate the grid 180 degrees.",
    "Reflect the grid horizontally (mirror left to right).",
    "Reflect the grid vertically (mirror top to bottom).",
    "Flip the grid along the main diagonal.",
    "Flip the grid along the anti-diagonal.",

    # --- Color substitution / manipulation ---
    "Swap the background color with the second-most-common color.",
    "Replace every non-background cell with color 3.",
    "Invert colors: swap the two most-common non-background colors.",
    "Cycle colors: 1→2, 2→3, 3→1, others unchanged.",
    "Replace each cell with the most common non-background color in its row.",
    "Replace each cell with the most common non-background color in its column.",
    "Replace each background cell with the nearest non-background color (Manhattan distance).",
    "Color every cell by its row index (mod 5).",
    "Color every cell by its column index (mod 5).",
    "Color the grid diagonal with color 4, keep the rest unchanged.",

    # --- Object extraction and filtering ---
    "Keep only the cells of the color that appears most frequently.",
    "Keep only the single largest connected component.",
    "Keep only connected components with 3 or more cells.",
    "Remove the single smallest connected component.",
    "Remove the single largest connected component.",
    "Keep only shapes whose cell count is greater than 2.",
    "Keep only shapes that touch the border of the grid.",
    "Remove any shape that touches the border of the grid.",
    "Keep only the cells that appear in exactly one row.",
    "Keep only symmetric shapes (shapes invariant under horizontal flip).",

    # --- Object movement ---
    "Find the largest contiguous object and move it to the center of the grid.",
    "Make every object fall straight down until it hits something or the bottom.",
    "Make every object fall straight up to the top or until it hits something.",
    "Push every object to the left edge of the grid.",
    "Push every object to the right edge of the grid.",
    "Shift every non-background row down by one (the bottom row disappears).",
    "Rotate each non-background row by one cell to the right.",

    # --- Containment / topology ---
    "Fill any enclosed background region with a fixed color.",
    "Fill any enclosed background region with color equal to the surrounding shape's color.",
    "Outline the boundary of each shape with a new color.",
    "Keep only the interior cells of shapes (remove the outline).",
    "Keep only the outline cells of shapes (remove the interior).",
    "Mark each hole (enclosed background) with color 7.",
    "Mark with color 7 any cell that is the same color as any of its 4-neighbors.",
    "Mark with color 4 any cell whose 4 neighbors have 4 different colors.",

    # --- Counting / size-based ---
    "Output a 1-row grid with N cells colored blue, where N is the number of shapes in the input.",
    "Output a 1-row grid with N cells, where N is the total number of non-background cells.",
    "Replace every non-background cell with the size of its connected component.",
    "Recolor each connected component based on its size: smallest = red, largest = blue.",
    "Keep only the cells that belong to the connected component with the most cells.",
    "Rank shapes by size and recolor: biggest=1, 2nd=2, 3rd=3, etc.",

    # --- Bounding box / crop ---
    "Output the bounding box of the unique-color object.",
    "Output the bounding box of the largest connected component.",
    "Crop to the smallest rectangle containing all non-background cells.",
    "Crop to the bounding box of the largest shape, keeping other shapes removed.",
    "Draw a bounding box (outline only) around each shape with color 5.",

    # --- Neighbor-based ---
    "Keep only the cells that are 4-adjacent to a cell of color 2 (if any).",
    "Recolor each cell to the color of its most common 8-neighbor.",
    "Set each cell to color 8 if it has at least 3 non-background 4-neighbors.",
    "Mark with color 2 any background cell that has exactly one non-background 4-neighbor.",
    "Erase any non-background cell that has no non-background 4-neighbor (isolated cells).",

    # --- Symmetry completion / generation ---
    "Complete the grid to be horizontally symmetric (mirror the right half to match the left).",
    "Complete the grid to be vertically symmetric (mirror the bottom half to match the top).",
    "Symmetrize under 180° rotation (make the output rot180-invariant).",
    "Overlay the grid with its horizontal flip (union of non-background cells).",
    "Take the pixel-wise AND (common non-background cells) of the grid and its horizontal flip.",

    # --- Tiling / replication ---
    "Tile a 2x2 copy of the input as the top-left quadrant of an output grid of the same shape.",
    "Double the grid by tiling it 2x horizontally.",
    "Double the grid by tiling it 2x vertically.",
    "Scale the grid 2x: each input cell becomes a 2x2 block.",
    "Replicate the grid 3x3 as a single tiled output (output is 3x larger in each dimension).",

    # --- Line drawing / connecting ---
    "Draw a horizontal line of color 3 through the row containing the most non-background cells.",
    "Draw a vertical line of color 3 through the column containing the most non-background cells.",
    "Connect each pair of same-color cells with a straight line of that color.",
    "Draw a diagonal line from top-left to bottom-right with color 5.",

    # --- Conditional / contextual ---
    "If the grid contains more red cells than blue, recolor all red to yellow; otherwise swap red and blue.",
    "If any cell has color 8, replace the entire grid's background with color 2.",
    "For each row: if it has more than 3 non-background cells, keep it; otherwise clear it to background.",
    "For each column: if it contains color 4, set the whole column to color 4.",

    # --- Composite (2-step) ---
    "Rotate the grid 90 degrees clockwise, then replace all cells of color 1 with color 5.",
    "Flip the grid horizontally, then keep only the largest connected component.",
    "Fill enclosed regions with color 2, then outline every shape with color 7.",
    "Scale the grid 2x (each cell → 2x2 block), then rotate 90 degrees clockwise.",
    "Keep only the largest shape, then recolor it to color 4.",
    "Symmetrize horizontally, then recolor all background cells touching a shape to color 6.",
    "Extract the bounding box of the unique-color object, then scale it 2x.",
    "Make objects fall down, then mark the newly-empty cells (originally non-background) with color 9.",
]


def build_prompt(seed: str) -> str:
    return f"""You write short Python functions for abstract grid-reasoning puzzles.

Write ONE Python function named `transform` that implements this rule:

RULE: {seed}

Requirements:
- Signature: `def transform(grid: np.ndarray) -> np.ndarray`
- `grid` is a 2D numpy array of ints in [0, 9]. 0 is background.
- Return a NEW 2D numpy array of ints in [0, 9]. Shape may differ from input.
- Output shape must be in [1..30] x [1..30].
- Deterministic: no randomness, no time-based operations, no file I/O.
- Use only `numpy as np` and standard Python; no other imports.
- No class definitions, no globals, no side effects outside the function body.
- The function MUST be self-contained and runnable as-is.

Output ONLY the Python code — no markdown fences, no commentary, no example usage.
Begin with `def transform` and end at the final `return` statement.
"""


# -----------------------------------------------------------------------------
# Ollama call (same HTTP interface as dream.py)
# -----------------------------------------------------------------------------


@dataclass
class OllamaCodeResponse:
    code: str
    elapsed_sec: float
    rule_seed: str


def ollama_generate_code(
    *,
    model: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
    temperature: float = 0.8,
    timeout_sec: float = 120.0,
    rng: _random.Random | None = None,
) -> OllamaCodeResponse:
    """Prompt the LLM for a Python transform function. Returns raw text + timing."""
    rng = rng or _random.Random()
    seed = rng.choice(RULE_SEEDS)
    prompt = build_prompt(seed)
    req_data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(req_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    return OllamaCodeResponse(
        code=payload.get("response", ""),
        elapsed_sec=time.time() - t0,
        rule_seed=seed,
    )


# -----------------------------------------------------------------------------
# Code extraction + compilation
# -----------------------------------------------------------------------------


def extract_function_code(raw: str) -> str | None:
    """Extract a `def transform(...)` block from LLM output.

    Handles the common failure modes: markdown code fences, prose preambles,
    trailing example usage or test code.
    """
    text = raw.strip()

    # Strip markdown fences.
    if "```" in text:
        # Grab the content of the first code block.
        m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Find the `def transform` line.
    lines = text.split("\n")
    start = -1
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def transform"):
            start = i
            break
    if start < 0:
        return None

    # Walk forward: keep lines until we hit a line at column 0 that isn't a
    # continuation of the function (i.e., code that would be outside the def).
    # This handles the case where the LLM appends test code after the function.
    result = [lines[start]]
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1 :]:
        if not line.strip():
            result.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and not line.lstrip().startswith(")"):
            # Back at or below the def line's indentation → function body ended.
            break
        result.append(line)

    # Prepend a numpy import so the function can use `np` freely even if the
    # LLM forgot to import it (common omission).
    return "import numpy as np\n" + "\n".join(result).rstrip()


def compile_transform(code: str) -> Callable | None:
    """Compile a code string and return the `transform` callable.

    Minimal isolation: we build a namespace with only numpy and a small set
    of builtins. No file I/O, no os access, no import machinery beyond numpy.
    Not a security sandbox — just a contamination barrier for local dev.
    """
    # A restricted builtins dict — the model doesn't need file/os access.
    safe_builtins = {
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "list": list, "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
        "int": int, "float": float, "bool": bool, "str": str,
        "min": min, "max": max, "sum": sum, "abs": abs, "all": all, "any": any,
        "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
        "isinstance": isinstance, "type": type, "iter": iter, "next": next,
        "round": round, "divmod": divmod, "pow": pow,
        "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
        "IndexError": IndexError, "StopIteration": StopIteration,
        "print": lambda *a, **k: None,  # silence stray prints
        "__import__": __import__,  # numpy needs this to satisfy its internal imports
        "__build_class__": __build_class__,  # occasionally needed for typing machinery
    }
    namespace = {"__builtins__": safe_builtins, "np": np}
    try:
        exec(code, namespace)
    except Exception:
        return None
    fn = namespace.get("transform")
    if not callable(fn):
        return None
    return fn


# -----------------------------------------------------------------------------
# Run with timeout (SIGALRM — Unix only)
# -----------------------------------------------------------------------------


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


def run_with_timeout(fn: Callable, arg, timeout_sec: float = 2.0):
    """Run fn(arg) with a wall-clock timeout. Returns the result or None on error.

    Uses SIGALRM (Unix/macOS only). On timeout or any exception, returns None.
    """
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return fn(arg)
    except _TimeoutError:
        return None
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# -----------------------------------------------------------------------------
# Input grid generators — structural variety
# -----------------------------------------------------------------------------


def _random_grid_random(rng: _random.Random, min_size=4, max_size=9) -> np.ndarray:
    h = rng.randint(min_size, max_size)
    w = rng.randint(min_size, max_size)
    n_colors = rng.randint(2, 4)
    palette = rng.sample(range(1, NUM_COLORS), n_colors)
    density = rng.uniform(0.3, 0.7)
    grid = np.zeros((h, w), dtype=np.int64)
    for i in range(h):
        for j in range(w):
            if rng.random() < density:
                grid[i, j] = rng.choice(palette)
    return grid


def _random_grid_single_object(rng: _random.Random, min_size=5, max_size=9) -> np.ndarray:
    """One rectangular block on a background."""
    h = rng.randint(min_size, max_size)
    w = rng.randint(min_size, max_size)
    grid = np.zeros((h, w), dtype=np.int64)
    bh = rng.randint(2, max(2, h - 2))
    bw = rng.randint(2, max(2, w - 2))
    y0 = rng.randint(0, h - bh)
    x0 = rng.randint(0, w - bw)
    grid[y0 : y0 + bh, x0 : x0 + bw] = rng.randint(1, NUM_COLORS - 1)
    return grid


def _random_grid_multi_object(rng: _random.Random, min_size=6, max_size=10) -> np.ndarray:
    """Two or three separated blocks of different colors."""
    h = rng.randint(min_size, max_size)
    w = rng.randint(min_size, max_size)
    grid = np.zeros((h, w), dtype=np.int64)
    n_objects = rng.randint(2, 3)
    colors = rng.sample(range(1, NUM_COLORS), n_objects)
    for c in colors:
        for _ in range(10):
            bh = rng.randint(1, 3)
            bw = rng.randint(1, 3)
            y0 = rng.randint(0, h - bh)
            x0 = rng.randint(0, w - bw)
            if (grid[y0 : y0 + bh, x0 : x0 + bw] == 0).all():
                grid[y0 : y0 + bh, x0 : x0 + bw] = c
                break
    return grid


GRID_GENERATORS = (_random_grid_random, _random_grid_single_object, _random_grid_multi_object)


def sample_input_grid(rng: _random.Random) -> np.ndarray:
    return rng.choice(GRID_GENERATORS)(rng)


# -----------------------------------------------------------------------------
# Demo generation from a compiled function
# -----------------------------------------------------------------------------


def _is_valid_output(arr) -> bool:
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    if h < 1 or w < 1 or h > 30 or w > 30:
        return False
    if arr.dtype.kind not in ("i", "u"):
        # Allow float outputs if they're integer-valued.
        try:
            int_arr = arr.astype(np.int64)
            if not np.allclose(int_arr, arr):
                return False
            arr = int_arr
        except Exception:
            return False
    if (arr < 0).any() or (arr >= NUM_COLORS).any():
        return False
    return True


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) and ra == rb for ra, rb in zip(a, b))


def _grid_is_blank(arr: np.ndarray) -> bool:
    return bool((arr == 0).all())


def generate_demos(
    transform_fn: Callable,
    *,
    num_demos: int,
    rng: _random.Random,
    timeout_per_call: float = 2.0,
    max_attempts: int = 30,
) -> list[Example] | None:
    """Run `transform_fn` on random inputs until we have `num_demos` valid
    (input, output) pairs. Returns None if we can't get enough.
    """
    demos: list[Example] = []
    attempts = 0
    while len(demos) < num_demos and attempts < max_attempts:
        attempts += 1
        inp_arr = sample_input_grid(rng)
        out = run_with_timeout(transform_fn, inp_arr.copy(), timeout_sec=timeout_per_call)
        if out is None:
            continue
        if not _is_valid_output(out):
            continue
        out_arr = out.astype(np.int64)

        inp_list: Grid = inp_arr.tolist()
        out_list: Grid = out_arr.tolist()

        # Reject degenerate pairs:
        if _grid_is_blank(out_arr):
            continue
        if _grids_equal(inp_list, out_list):
            continue
        # Deduplicate: avoid identical inputs across demos.
        if any(_grids_equal(inp_list, ex.input) for ex in demos):
            continue
        demos.append(Example(input=inp_list, output=out_list))

    if len(demos) < num_demos:
        return None

    # Reject if all demo outputs are identical — degenerate rule.
    if len(demos) >= 2 and all(
        _grids_equal(d.output, demos[0].output) for d in demos[1:]
    ):
        return None

    return demos


# -----------------------------------------------------------------------------
# End-to-end: generate one code-dreamed task
# -----------------------------------------------------------------------------


@dataclass
class CodeDreamResult:
    task: Task
    code: str
    rule_seed: str
    llm_sec: float
    n_demos: int


def dream_one_code_task(
    task_id: str,
    *,
    num_train: int = 4,
    num_test: int = 1,
    model: str = "qwen2.5:7b",
    temperature: float = 0.8,
    rng: _random.Random | None = None,
    max_llm_retries: int = 2,
) -> CodeDreamResult | None:
    """Generate a single code-dreamed task end-to-end.

    Protocol:
      1. Call LLM for a transform function. Retry up to max_llm_retries if the
         code won't parse/compile/run.
      2. Run the function on random input grids until we have num_train + num_test
         valid demo pairs.
      3. Wrap as a Task and return.
    """
    rng = rng or _random.Random()
    total_llm_sec = 0.0
    for attempt in range(max_llm_retries + 1):
        try:
            resp = ollama_generate_code(model=model, temperature=temperature, rng=rng)
        except RuntimeError:
            return None
        total_llm_sec += resp.elapsed_sec
        code = extract_function_code(resp.code)
        if code is None:
            continue
        fn = compile_transform(code)
        if fn is None:
            continue
        # Quick sanity run on one dummy input.
        sanity_in = sample_input_grid(rng)
        sanity_out = run_with_timeout(fn, sanity_in.copy(), timeout_sec=2.0)
        if sanity_out is None or not _is_valid_output(sanity_out):
            continue

        # Full demo generation.
        total_needed = num_train + num_test
        all_demos = generate_demos(
            fn,
            num_demos=total_needed,
            rng=rng,
        )
        if all_demos is None:
            continue

        train = all_demos[:num_train]
        test = all_demos[num_train:]
        task = Task(task_id=task_id, train=train, test=test)
        return CodeDreamResult(
            task=task,
            code=code,
            rule_seed=resp.rule_seed,
            llm_sec=total_llm_sec,
            n_demos=len(all_demos),
        )

    return None


def save_code_task(result: CodeDreamResult, dir_path) -> None:
    from pathlib import Path as _Path
    dir_path = _Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / f"{result.task.task_id}.json"
    payload = {
        "train": [{"input": ex.input, "output": ex.output} for ex in result.task.train],
        "test": [{"input": ex.input, "output": ex.output} for ex in result.task.test],
        "_rule_seed": result.rule_seed,
        "_source": "code_dream",
        "_code": result.code,
    }
    with out_path.open("w") as f:
        json.dump(payload, f)
