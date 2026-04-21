"""Dream pipeline — LLM-generated synthetic ARC tasks (Week 8).

Uses a local Ollama model to generate ARC-shaped JSON tasks, parses and
validates them, and returns `Task` objects ready to be fed into the same
training pipeline we use for the hand-written DSL corpus.

This is the "cheap-first → upgrade-later" second phase (from Week 3 planning):
where the Week 3a rule-based generator could only produce tasks in the shape
of its 17 hand-written primitives, the LLM is a much broader prior and can
dream up compositional tasks that are structurally closer to real ARC-AGI-2.

The expensive part of this pipeline is NOT the LLM — it's the filter. Every
generated task is only accepted if Volcan can overfit its demos within a
time budget. This guarantees the Gold Dataset is (a) learnable by our specific
model and (b) free of LLM hallucinations where the "rule" is inconsistent
across demo pairs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .arc import Example, NUM_COLORS, Task


# -----------------------------------------------------------------------------
# Prompt — engineered for ARC task generation
# -----------------------------------------------------------------------------

TASK_GENERATION_PROMPT = """You are an ARC-AGI task designer. Create ONE abstract reasoning puzzle.

Output strict JSON matching this schema exactly:
{
  "rule_description": "<one short sentence describing the transformation>",
  "train": [
    {"input": <grid>, "output": <grid>},
    {"input": <grid>, "output": <grid>},
    {"input": <grid>, "output": <grid>}
  ],
  "test": [
    {"input": <grid>, "output": <grid>}
  ]
}

Where each <grid> is a 2D array of integers in [0,9] representing colors (0=black/background).

Constraints:
- Output ONLY the JSON. No markdown fences, no commentary, no "Here is the task", nothing else.
- Grid sizes: 3x3 to 8x8, rectangular OK.
- Exactly 3 train pairs and 1 test pair.
- All train AND test pairs MUST follow the SAME rule.
- The inputs across pairs should be DIFFERENT (not just color-swapped versions of each other).
- The test pair must be consistent with the rule AND different from all train pairs.
- The transformation should be non-trivial: NOT identity, NOT pure color renaming.

Pick ONE interesting transformation type from this list (or similar):
- Rotate/flip/transpose
- Gravity (objects fall to an edge)
- Fill enclosed regions with a color
- Recolor cells based on a neighbor's color
- Keep only the largest connected shape
- Outline the boundaries of shapes
- Find and mark cells of a unique color
- Count cells and draw that many copies elsewhere
- Tile / fractal replication
- Symmetric completion (mirror across an axis)

Generate the JSON now:
"""


# -----------------------------------------------------------------------------
# Ollama client — minimal stdlib-only HTTP POST
# -----------------------------------------------------------------------------


@dataclass
class OllamaResponse:
    text: str
    elapsed_sec: float


def ollama_generate(
    prompt: str,
    *,
    model: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
    temperature: float = 0.8,
    timeout_sec: float = 120.0,
) -> OllamaResponse:
    """Call Ollama's /api/generate endpoint and return the response text."""
    import time

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
    return OllamaResponse(text=payload.get("response", ""), elapsed_sec=time.time() - t0)


# -----------------------------------------------------------------------------
# JSON parsing + schema validation
# -----------------------------------------------------------------------------


def _extract_json_block(text: str) -> str | None:
    """Find the first top-level {...} block in a raw LLM response.

    Handles common failure modes: markdown fences, leading/trailing commentary,
    partial thought processes. Returns None if no parseable block is found.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Find the closing fence
        lines = text.split("\n")
        inner = []
        inside = False
        for line in lines:
            if line.startswith("```"):
                if inside:
                    break
                inside = True
                continue
            if inside:
                inner.append(line)
        text = "\n".join(inner).strip()

    # Brace-counting scan for the first balanced {...} block.
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _is_valid_grid(g) -> bool:
    if not isinstance(g, list) or not g:
        return False
    if not isinstance(g[0], list) or not g[0]:
        return False
    w = len(g[0])
    h = len(g)
    if h > 30 or w > 30 or h < 1 or w < 1:
        return False
    for row in g:
        if not isinstance(row, list) or len(row) != w:
            return False
        for c in row:
            if not isinstance(c, int) or c < 0 or c >= NUM_COLORS:
                return False
    return True


def parse_llm_task(raw_text: str, task_id: str) -> tuple[Task, str] | None:
    """Parse an LLM response into a `Task`. Returns (task, rule_description)
    on success, None on any failure (parse error, bad schema, invalid grids).
    """
    block = _extract_json_block(raw_text)
    if block is None:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    rule = data.get("rule_description", "") or ""

    train_raw = data.get("train")
    test_raw = data.get("test")
    if not isinstance(train_raw, list) or not isinstance(test_raw, list):
        return None
    if len(train_raw) < 2 or len(test_raw) < 1:
        return None

    def _extract_examples(pairs):
        out = []
        for p in pairs:
            if not isinstance(p, dict):
                return None
            inp = p.get("input")
            out_grid = p.get("output")
            if not _is_valid_grid(inp) or not _is_valid_grid(out_grid):
                return None
            out.append(Example(input=inp, output=out_grid))
        return out

    train = _extract_examples(train_raw)
    test = _extract_examples(test_raw)
    if train is None or test is None:
        return None

    task = Task(task_id=task_id, train=train, test=test)
    return task, str(rule)[:200]


# -----------------------------------------------------------------------------
# High-level: generate one task via Ollama, parse, return or None
# -----------------------------------------------------------------------------


def generate_one_task(
    task_id: str,
    *,
    model: str = "qwen2.5:7b",
    temperature: float = 0.8,
    max_retries: int = 2,
) -> tuple[Task, str, float] | None:
    """Generate a single task via Ollama. Retries up to `max_retries` times on
    parse/validation failure. Returns (task, rule_description, elapsed_sec)
    on success, None otherwise.
    """
    total_elapsed = 0.0
    for attempt in range(max_retries + 1):
        try:
            resp = ollama_generate(TASK_GENERATION_PROMPT, model=model, temperature=temperature)
        except RuntimeError:
            return None
        total_elapsed += resp.elapsed_sec
        parsed = parse_llm_task(resp.text, task_id)
        if parsed is not None:
            task, rule = parsed
            return task, rule, total_elapsed
    return None


# -----------------------------------------------------------------------------
# Save a task to ARC-native JSON
# -----------------------------------------------------------------------------


def save_dream_task(
    task: Task,
    rule_description: str,
    dir_path: str | Path,
) -> Path:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / f"{task.task_id}.json"
    payload = {
        "train": [{"input": ex.input, "output": ex.output} for ex in task.train],
        "test": [{"input": ex.input, "output": ex.output} for ex in task.test],
        "_rule": rule_description,
        "_source": "dream",
    }
    with out_path.open("w") as f:
        json.dump(payload, f)
    return out_path
