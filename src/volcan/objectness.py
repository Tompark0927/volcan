"""Objectness — pre-segmentation layer (Week 7, CEO's #1).

The failure mode of pure-pixel NCAs is that they treat every cell as an
independent particle. Real ARC tasks are about OBJECTS: "move the blue square,"
"keep only the largest shape," "count the distinct objects." A pixel-level
update rule has to re-discover "these 9 cells belong to the same square" from
scratch every time it runs.

This module fixes that by **pre-segmenting the input into connected components
BEFORE the NCA runs**, then exposing an "object ID embedding" as a static
context channel that every cell can read at every step.

Each cell (i,j) knows:
  - Its color (input channel)
  - Its object ID (new: a small learnable embedding tied to the component it
    belongs to)

When a cell sees that its neighbor has the same object ID, it knows they
belong to the same shape. The NCA update rule can then learn to coordinate
cells that share an object ID — moving them together, recoloring them
together, deleting them together.

The segmentation is computed ONCE from the input and held constant for the
full Phase A + Phase B rollout. It doesn't update with the predicted output.
That's a limitation — if the correct rule is "split this object in half," we
can't represent the split with an input-derived segmentation — but it's the
right starting point for the majority of ARC tasks.
"""

from __future__ import annotations

import torch

from .arc import NUM_COLORS, OUTSIDE_TOKEN


def segment_objects(
    grid_idx: torch.Tensor,
    *,
    connectivity: int = 4,
    treat_background_as_object: bool = False,
) -> torch.Tensor:
    """Label connected components in a grid of color indices.

    Two cells are in the same component iff they share a color AND are
    4-connected (or 8-connected if connectivity=8).

    Args:
        grid_idx: (H, W) long tensor of color indices in [0, VOCAB_SIZE).
        connectivity: 4 or 8 — Moore vs von-Neumann neighborhood.
        treat_background_as_object: if False, all background cells (color 0)
            get object ID 0 as a shared "no-object" label. If True, contiguous
            background regions become their own objects (useful when the rule
            operates on background shapes).

    Returns:
        labels: (H, W) long tensor. Background cells = 0 (unless
        `treat_background_as_object` is True). Foreground components are
        numbered 1, 2, 3, ... in the order they're discovered by a raster scan.
    """
    if grid_idx.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(grid_idx.shape)}")
    h, w = grid_idx.shape
    labels = torch.zeros(h, w, dtype=torch.long)
    next_label = 1

    if connectivity == 4:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    grid_list = grid_idx.tolist()
    labels_list = labels.tolist()

    for i in range(h):
        for j in range(w):
            if labels_list[i][j] != 0:
                continue
            c = grid_list[i][j]
            if c == OUTSIDE_TOKEN:
                continue  # padding never gets a label
            if c == 0 and not treat_background_as_object:
                continue  # background stays at label 0

            # BFS to find all cells in this component.
            stack = [(i, j)]
            labels_list[i][j] = next_label
            while stack:
                y, x = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if labels_list[ny][nx] == 0 and grid_list[ny][nx] == c:
                            labels_list[ny][nx] = next_label
                            stack.append((ny, nx))
            next_label += 1

    return torch.tensor(labels_list, dtype=torch.long)


def object_embedding(
    labels: torch.Tensor,
    embed_dim: int = 4,
    *,
    max_objects: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Assign each object label a random low-dim embedding.

    All cells with label k get the same embedding (so the NCA can tell
    same-object cells from different-object cells). Label 0 (background/no
    object) always gets the zero vector — a sentinel for "unaffiliated."

    Args:
        labels: (H, W) long tensor from `segment_objects`.
        embed_dim: dimensionality of each object's embedding vector.
        max_objects: maximum number of distinct objects supported. Labels
            above this wrap around modulo max_objects.
        seed: RNG seed — same seed means same embedding table across calls,
            so downstream code always sees a consistent "object alphabet."

    Returns:
        embedding: (embed_dim, H, W) float tensor.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    # One embedding row per possible label, plus row 0 = zeros for background.
    table = torch.zeros(max_objects + 1, embed_dim)
    table[1:] = torch.randn(max_objects, embed_dim, generator=g) * 0.5

    # Wrap labels above max_objects.
    labels_mod = labels.clone()
    mask = labels_mod > max_objects
    if mask.any():
        labels_mod[mask] = ((labels_mod[mask] - 1) % max_objects) + 1

    # (H, W) → (H, W, embed_dim) via table lookup
    embedded = table[labels_mod]  # (H, W, embed_dim)
    return embedded.permute(2, 0, 1).contiguous()  # (embed_dim, H, W)


def compute_object_field(
    input_onehot: torch.Tensor,
    embed_dim: int = 4,
    *,
    max_objects: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """End-to-end: one-hot input → object embedding field.

    Args:
        input_onehot: (B, V, H, W) or (V, H, W) one-hot color tensor.

    Returns:
        object_field: (B, embed_dim, H, W) or (embed_dim, H, W) — same batch
            dim as the input. This is the per-cell object binding vector that
            VolcanCell will read as a static context channel alongside color.
    """
    squeeze = False
    if input_onehot.ndim == 3:
        input_onehot = input_onehot.unsqueeze(0)
        squeeze = True

    b, v, h, w = input_onehot.shape
    grid_idx = input_onehot.argmax(dim=1)  # (B, H, W)

    fields = []
    for bi in range(b):
        labels = segment_objects(grid_idx[bi])
        emb = object_embedding(labels, embed_dim=embed_dim, max_objects=max_objects, seed=seed)
        fields.append(emb)

    out = torch.stack(fields, dim=0)  # (B, embed_dim, H, W)
    return out.squeeze(0) if squeeze else out
