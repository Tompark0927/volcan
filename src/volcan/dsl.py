"""Volcan DSL — primitives for the rule-based synthetic task generator.

Week 3a:  13 starter primitives (color + spatial)
Week 6c:  + 4 structural primitives per CEO request
            - Gravity: per-column "fall" toward the bottom edge
            - Contact: cells of color A adjacent to color B turn into color X
            - Containment: fill enclosed background regions with a target color
            - ScaleX2: replicate every cell into a 2×2 block (variable output size)

A "primitive" is a deterministic Grid → Grid function with bound parameters.
A "rule" is either a primitive or a composition of primitives. Rules can be
randomly sampled and applied to randomly-generated input grids to produce
verified synthetic ARC-style tasks.
"""

from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod
from collections import deque
from typing import Sequence

from .arc import Grid, NUM_COLORS


# All primitives operate on integers in [0, NUM_COLORS) = [0, 9].
BACKGROUND = 0  # ARC convention: 0 is background/black


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------


class Primitive(ABC):
    """A deterministic Grid → Grid transformation with bound parameters."""

    name: str

    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """Return a new grid; do not mutate the input."""

    @classmethod
    @abstractmethod
    def random(cls, rng: _random.Random) -> "Primitive":
        """Sample a randomly-parameterized instance."""

    def __repr__(self) -> str:
        return self.name


# -----------------------------------------------------------------------------
# Color primitives
# -----------------------------------------------------------------------------


class Identity(Primitive):
    """Returns the input grid unchanged. Used as a baseline / no-op."""

    def __init__(self) -> None:
        self.name = "Identity"

    def apply(self, grid: Grid) -> Grid:
        return [row[:] for row in grid]

    @classmethod
    def random(cls, rng: _random.Random) -> "Identity":
        return cls()


class Recolor(Primitive):
    """Replace every cell of color `from_color` with `to_color`."""

    def __init__(self, from_color: int, to_color: int) -> None:
        self.from_color = from_color
        self.to_color = to_color
        self.name = f"Recolor({from_color}->{to_color})"

    def apply(self, grid: Grid) -> Grid:
        return [
            [self.to_color if c == self.from_color else c for c in row]
            for row in grid
        ]

    @classmethod
    def random(cls, rng: _random.Random) -> "Recolor":
        from_color = rng.randint(1, NUM_COLORS - 1)  # don't recolor the background
        to_color = rng.randint(0, NUM_COLORS - 1)
        if to_color == from_color:
            to_color = (to_color + 1) % NUM_COLORS
        return cls(from_color, to_color)


class SwapColors(Primitive):
    """Swap every cell of color a with color b and vice versa."""

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b
        self.name = f"Swap({a}<->{b})"

    def apply(self, grid: Grid) -> Grid:
        result = []
        for row in grid:
            new_row = []
            for c in row:
                if c == self.a:
                    new_row.append(self.b)
                elif c == self.b:
                    new_row.append(self.a)
                else:
                    new_row.append(c)
            result.append(new_row)
        return result

    @classmethod
    def random(cls, rng: _random.Random) -> "SwapColors":
        a, b = rng.sample(range(1, NUM_COLORS), 2)
        return cls(a, b)


class KeepColor(Primitive):
    """Keep one color unchanged; set all other non-background cells to background."""

    def __init__(self, color: int) -> None:
        self.color = color
        self.name = f"KeepColor({color})"

    def apply(self, grid: Grid) -> Grid:
        return [[c if c == self.color else BACKGROUND for c in row] for row in grid]

    @classmethod
    def random(cls, rng: _random.Random) -> "KeepColor":
        return cls(rng.randint(1, NUM_COLORS - 1))


class RemoveColor(Primitive):
    """Set all cells of one color to background."""

    def __init__(self, color: int) -> None:
        self.color = color
        self.name = f"RemoveColor({color})"

    def apply(self, grid: Grid) -> Grid:
        return [[BACKGROUND if c == self.color else c for c in row] for row in grid]

    @classmethod
    def random(cls, rng: _random.Random) -> "RemoveColor":
        return cls(rng.randint(1, NUM_COLORS - 1))


class FillBackground(Primitive):
    """Replace every background cell with a non-background color."""

    def __init__(self, color: int) -> None:
        self.color = color
        self.name = f"FillBackground({color})"

    def apply(self, grid: Grid) -> Grid:
        return [
            [self.color if c == BACKGROUND else c for c in row]
            for row in grid
        ]

    @classmethod
    def random(cls, rng: _random.Random) -> "FillBackground":
        return cls(rng.randint(1, NUM_COLORS - 1))


class InvertMask(Primitive):
    """For each cell: background → `fg_color`, non-background → background."""

    def __init__(self, fg_color: int) -> None:
        self.fg_color = fg_color
        self.name = f"InvertMask(fg={fg_color})"

    def apply(self, grid: Grid) -> Grid:
        return [
            [BACKGROUND if c != BACKGROUND else self.fg_color for c in row]
            for row in grid
        ]

    @classmethod
    def random(cls, rng: _random.Random) -> "InvertMask":
        return cls(rng.randint(1, NUM_COLORS - 1))


# -----------------------------------------------------------------------------
# Spatial primitives — all preserve grid shape (or transpose square grids)
# -----------------------------------------------------------------------------


class Rotate90(Primitive):
    """Rotate the grid 90° counter-clockwise. Only valid for square grids."""

    def __init__(self) -> None:
        self.name = "Rotate90"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        return [[grid[r][w - 1 - c] for r in range(h)] for c in range(w)]

    @classmethod
    def random(cls, rng: _random.Random) -> "Rotate90":
        return cls()


class Rotate180(Primitive):
    def __init__(self) -> None:
        self.name = "Rotate180"

    def apply(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid[::-1]]

    @classmethod
    def random(cls, rng: _random.Random) -> "Rotate180":
        return cls()


class Rotate270(Primitive):
    """Rotate 270° CCW = 90° clockwise."""

    def __init__(self) -> None:
        self.name = "Rotate270"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]

    @classmethod
    def random(cls, rng: _random.Random) -> "Rotate270":
        return cls()


class FlipHorizontal(Primitive):
    """Flip across the vertical axis (mirror left-right)."""

    def __init__(self) -> None:
        self.name = "FlipHorizontal"

    def apply(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid]

    @classmethod
    def random(cls, rng: _random.Random) -> "FlipHorizontal":
        return cls()


class FlipVertical(Primitive):
    """Flip across the horizontal axis (mirror top-bottom)."""

    def __init__(self) -> None:
        self.name = "FlipVertical"

    def apply(self, grid: Grid) -> Grid:
        return [row[:] for row in grid[::-1]]

    @classmethod
    def random(cls, rng: _random.Random) -> "FlipVertical":
        return cls()


class Transpose(Primitive):
    """Transpose the grid. Square grids stay square; rectangular grids swap dims."""

    def __init__(self) -> None:
        self.name = "Transpose"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        return [[grid[r][c] for r in range(h)] for c in range(w)]

    @classmethod
    def random(cls, rng: _random.Random) -> "Transpose":
        return cls()


# -----------------------------------------------------------------------------
# Structural primitives (Week 6 — CEO additions)
# -----------------------------------------------------------------------------


class Gravity(Primitive):
    """Per-column "fall": every non-background cell drops to the bottom of its
    column, preserving order. Cells are stacked at the bottom; background fills
    the top of each column.

    Direction is fixed to "down" for now (the most common ARC convention).
    """

    def __init__(self) -> None:
        self.name = "Gravity"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        out = [[BACKGROUND] * w for _ in range(h)]
        for j in range(w):
            cells = [grid[i][j] for i in range(h) if grid[i][j] != BACKGROUND]
            for k, c in enumerate(cells):
                out[h - len(cells) + k][j] = c
        return out

    @classmethod
    def random(cls, rng: _random.Random) -> "Gravity":
        return cls()


class Contact(Primitive):
    """Cells of color a that are 4-adjacent to any cell of color b become color x.

    Useful for ARC-like rules: "every blue cell touching a red cell becomes
    yellow". The detection uses 4-connectivity (Moore-4 / von Neumann).
    """

    def __init__(self, a: int, b: int, x: int) -> None:
        self.a = a
        self.b = b
        self.x = x
        self.name = f"Contact({a}~{b}->{x})"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        out = [row[:] for row in grid]
        for i in range(h):
            for j in range(w):
                if grid[i][j] != self.a:
                    continue
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == self.b:
                        out[i][j] = self.x
                        break
        return out

    @classmethod
    def random(cls, rng: _random.Random) -> "Contact":
        a, b, x = rng.sample(range(1, NUM_COLORS), 3)
        return cls(a, b, x)


class Containment(Primitive):
    """Fill enclosed background regions with `color`.

    A "background" cell is one with value `BACKGROUND` (0). A region is
    "enclosed" if it is NOT 4-connected to the grid border via other background
    cells. We do a BFS from the border to mark all "outside" background cells;
    every remaining background cell is "interior" and gets recolored.

    This is the canonical ARC "fill the inside of the hollow shape" operator.
    """

    def __init__(self, color: int) -> None:
        self.color = color
        self.name = f"Containment(fill={color})"

    def apply(self, grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        outside = [[False] * w for _ in range(h)]
        queue: deque[tuple[int, int]] = deque()

        # Seed from all border cells that are background.
        for i in range(h):
            for j in (0, w - 1):
                if grid[i][j] == BACKGROUND and not outside[i][j]:
                    outside[i][j] = True
                    queue.append((i, j))
        for j in range(w):
            for i in (0, h - 1):
                if grid[i][j] == BACKGROUND and not outside[i][j]:
                    outside[i][j] = True
                    queue.append((i, j))

        # BFS through 4-connected background cells.
        while queue:
            i, j = queue.popleft()
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if grid[ni][nj] == BACKGROUND and not outside[ni][nj]:
                        outside[ni][nj] = True
                        queue.append((ni, nj))

        # Anything background AND not marked outside is interior → fill.
        out = [row[:] for row in grid]
        for i in range(h):
            for j in range(w):
                if grid[i][j] == BACKGROUND and not outside[i][j]:
                    out[i][j] = self.color
        return out

    @classmethod
    def random(cls, rng: _random.Random) -> "Containment":
        return cls(rng.randint(1, NUM_COLORS - 1))


class ScaleX2(Primitive):
    """Replicate every cell into a 2×2 block.

    Output dimensions are exactly 2× the input dimensions. This is the
    "fractal/scaling" primitive — it generates variable-output-size synthetic
    tasks (the only such primitive in the current set), which means callers
    must use small inputs (≤ 7×7) to keep outputs within the 30×30 padded
    canvas. The synthetic generator handles this via its grid-size config.
    """

    def __init__(self) -> None:
        self.name = "ScaleX2"

    def apply(self, grid: Grid) -> Grid:
        out: Grid = []
        for row in grid:
            doubled_row = []
            for c in row:
                doubled_row.extend([c, c])
            out.append(doubled_row)
            out.append(doubled_row[:])
        return out

    @classmethod
    def random(cls, rng: _random.Random) -> "ScaleX2":
        return cls()


# -----------------------------------------------------------------------------
# Composition
# -----------------------------------------------------------------------------


class Then(Primitive):
    """Apply primitives in sequence: input → step₁ → step₂ → … → output.

    Multi-step reasoning (Chollet's category #2). Each step's output is the
    next step's input.
    """

    def __init__(self, primitives: Sequence[Primitive]) -> None:
        self.primitives: tuple[Primitive, ...] = tuple(primitives)
        self.name = " >> ".join(p.name for p in self.primitives)

    def apply(self, grid: Grid) -> Grid:
        for p in self.primitives:
            grid = p.apply(grid)
        return grid

    @classmethod
    def random(cls, rng: _random.Random, depth: int = 2) -> "Then":
        primitives = [random_primitive(rng) for _ in range(depth)]
        return cls(primitives)


# -----------------------------------------------------------------------------
# Registry & sampling
# -----------------------------------------------------------------------------

# Primitives that work on rectangular grids (don't care about being square).
SHAPE_PRESERVING_PRIMITIVES: list[type[Primitive]] = [
    Recolor,
    SwapColors,
    KeepColor,
    RemoveColor,
    FillBackground,
    InvertMask,
    Rotate180,
    FlipHorizontal,
    FlipVertical,
    Gravity,         # Week 6
    Contact,         # Week 6
    Containment,     # Week 6
]

# Primitives that require square grids.
SQUARE_ONLY_PRIMITIVES: list[type[Primitive]] = [
    Rotate90,
    Rotate270,
    Transpose,
]

# Primitives that change the output grid shape. These can ONLY appear as the
# last step of a Then-chain (so we don't try to feed an oversized grid into a
# downstream primitive). The synthetic generator should clamp input sizes
# small enough that the output still fits in the 30×30 padded canvas.
SHAPE_CHANGING_PRIMITIVES: list[type[Primitive]] = [
    ScaleX2,         # Week 6 — output is 2× input dims
]


def random_primitive(
    rng: _random.Random,
    *,
    allow_square_only: bool = False,
    allow_shape_changing: bool = False,
) -> Primitive:
    """Pick a random primitive class and sample its parameters."""
    pool: list[type[Primitive]] = list(SHAPE_PRESERVING_PRIMITIVES)
    if allow_square_only:
        pool.extend(SQUARE_ONLY_PRIMITIVES)
    if allow_shape_changing:
        pool.extend(SHAPE_CHANGING_PRIMITIVES)
    cls = rng.choice(pool)
    return cls.random(rng)


def random_rule(
    rng: _random.Random,
    *,
    max_depth: int = 2,
    p_compose: float = 0.5,
    allow_square_only: bool = False,
    allow_shape_changing: bool = True,
    p_shape_changing_tail: float = 0.15,
) -> Primitive:
    """Sample a random rule.

    With probability `p_compose`, the rule is a Then-chain of length 2..max_depth;
    otherwise it's a single primitive.

    Shape-changing primitives (e.g. ScaleX2) are constrained to appear ONLY as
    the last step of a chain (so we don't feed a doubled grid into a downstream
    primitive). With probability `p_shape_changing_tail`, the chain ends with
    a shape-changing primitive instead of a normal one.
    """
    if max_depth >= 2 and rng.random() < p_compose:
        depth = rng.randint(2, max_depth)
        primitives: list[Primitive] = [
            random_primitive(rng, allow_square_only=allow_square_only)
            for _ in range(depth - 1)
        ]
        if allow_shape_changing and rng.random() < p_shape_changing_tail:
            tail_cls = rng.choice(SHAPE_CHANGING_PRIMITIVES)
            primitives.append(tail_cls.random(rng))
        else:
            primitives.append(
                random_primitive(rng, allow_square_only=allow_square_only)
            )
        return Then(primitives)
    # Single-primitive rule. Allow shape-changing as a one-off.
    return random_primitive(
        rng,
        allow_square_only=allow_square_only,
        allow_shape_changing=allow_shape_changing,
    )
