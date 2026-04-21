"""Mycelial bypass — Pillar 1b.

Sparse learned long-range attention. Each cell has K hyphal partners selected
from a fixed small-world topology (random distant cells, sampled once per grid
size with a fixed seed). On every update step, each cell aggregates a learned-
weighted message from its partners.

Topology is fixed; weights are learned. This is the "high-speed bypass" the CEO
described — one-hop information transport across arbitrary distance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_symmetric_topology(
    h: int,
    w: int,
    *,
    k_random: int = 1,
    min_distance: int = 6,
    seed: int = 0,
) -> torch.Tensor:
    """Week 7 (CEO #2): deterministic SYMMETRIC partners + optional random one.

    For each cell (y, x), the hyphal partners are:
      0: horizontal mirror         (y,       w-1-x)
      1: vertical mirror           (h-1-y,   x)
      2: 180° rotation / center    (h-1-y,   w-1-x)
      3..k_random: random distant cells

    Total K = 3 + k_random. With k_random=1 we get 4 partners, the same budget
    as the random-only topology but with strong prior structure: every cell
    has a direct 1-hop link to its three symmetry-related counterparts. A
    signal at (y, x) can cross-talk with its mirror image in a single update,
    which is exactly what's needed for symmetry-based ARC tasks.
    """
    n = h * w
    k = 3 + k_random
    partners = torch.empty(n, k, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(seed)

    for i in range(n):
        y = i // w
        x = i % w
        # Deterministic symmetry partners (first 3 slots).
        partners[i, 0] = y * w + (w - 1 - x)                      # h-mirror
        partners[i, 1] = (h - 1 - y) * w + x                      # v-mirror
        partners[i, 2] = (h - 1 - y) * w + (w - 1 - x)            # 180°

        if k_random > 0:
            # Fill remaining slots with random distant cells.
            for slot in range(3, 3 + k_random):
                for _ in range(50):
                    cand = int(torch.randint(0, n, (1,), generator=g).item())
                    cy, cx = cand // w, cand % w
                    if cand == i:
                        continue
                    if max(abs(cy - y), abs(cx - x)) >= min_distance:
                        break
                else:
                    # Fallback: any other cell.
                    cand = (i + 1) % n
                partners[i, slot] = cand

    return partners


def sample_small_world_topology(
    h: int,
    w: int,
    k: int = 4,
    min_distance: int = 6,
    seed: int = 0,
) -> torch.Tensor:
    """Legacy Week 2 random-only topology. Kept for ablation; new code should
    use `sample_symmetric_topology`."""
    g = torch.Generator()
    g.manual_seed(seed)
    n = h * w
    partners = torch.empty(n, k, dtype=torch.long)
    coords = torch.tensor([[i, j] for i in range(h) for j in range(w)])  # (n, 2)

    for i in range(n):
        ri, ci = coords[i].tolist()
        di = (coords[:, 0] - ri).abs()
        dj = (coords[:, 1] - ci).abs()
        far = (torch.maximum(di, dj) >= min_distance) & (torch.arange(n) != i)
        candidates = torch.where(far)[0]
        if len(candidates) == 0:
            candidates = torch.tensor([j for j in range(n) if j != i])
        if len(candidates) >= k:
            picks = candidates[torch.randperm(len(candidates), generator=g)[:k]]
        else:
            idx = torch.randint(0, len(candidates), (k,), generator=g)
            picks = candidates[idx]
        partners[i] = picks
    return partners


class MycelialAttention(nn.Module):
    """Pillar 1b: per-cell aggregation of K hyphal partner messages.

    As of Week 7 the topology is SYMMETRIC by default (CEO #2): three of the
    four partners are deterministic mirror points (h-flip, v-flip, 180°), and
    one is a random distant cell. This injects symmetry as a hard prior — a
    signal at (y,x) has a 1-hop channel to its mirrored counterparts.
    """

    def __init__(
        self,
        max_grid_size: int = 30,
        num_partners: int = 4,
        min_distance: int = 6,
        in_channels: int = 59,
        out_dim: int = 8,
        seed: int = 0,
        topology: str = "symmetric",  # "symmetric" or "random"
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.num_partners = num_partners
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.topology = topology

        if topology == "symmetric":
            # 3 symmetry partners + (num_partners - 3) random distant cells.
            k_random = max(0, num_partners - 3)
            partners = sample_symmetric_topology(
                max_grid_size, max_grid_size,
                k_random=k_random, min_distance=min_distance, seed=seed,
            )
            # Sanity: symmetric topology always has 3 + k_random partners;
            # validate that matches num_partners.
            assert partners.shape[1] == num_partners, (
                f"symmetric topology produced {partners.shape[1]} partners, "
                f"num_partners={num_partners}"
            )
        elif topology == "random":
            partners = sample_small_world_topology(
                max_grid_size, max_grid_size, k=num_partners,
                min_distance=min_distance, seed=seed,
            )
        else:
            raise ValueError(f"unknown topology: {topology}")
        self.register_buffer("partners", partners)  # (N, K) long

        # Per-cell query projection -> K attention scores over its K partners.
        self.query = nn.Linear(in_channels, num_partners)
        # Project partner states down to out_dim.
        self.value = nn.Linear(in_channels, out_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Aggregate one mycelial message per cell.

        Args:
            state: (B, C, H, W) — full per-cell state (excluding position).

        Returns:
            messages: (B, out_dim, H, W) — per-cell aggregated partner signal.
        """
        b, c, h, w = state.shape
        if h != self.max_grid_size or w != self.max_grid_size:
            raise ValueError(
                f"MycelialAttention expected {self.max_grid_size}×{self.max_grid_size}, got {h}×{w}"
            )
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {c}")

        n = h * w
        flat = state.reshape(b, c, n).transpose(1, 2)  # (B, N, C)

        # Each cell queries its K partners.
        attn_logits = self.query(flat)              # (B, N, K)
        attn = F.softmax(attn_logits, dim=-1)       # (B, N, K)

        # Gather partner values.
        values = self.value(flat)                   # (B, N, out_dim)

        # For each cell i, gather values[partners[i, k]] for k=0..K-1.
        # partners is (N, K); gather using index_select per partner slot.
        # Expand to batched gather: (B, N, K, out_dim).
        partners_idx = self.partners.unsqueeze(0).unsqueeze(-1).expand(
            b, n, self.num_partners, self.out_dim
        )  # (B, N, K, out_dim)
        # values has shape (B, N, out_dim); we need (B, N, K, out_dim) by gathering along dim 1.
        gathered = torch.gather(
            values.unsqueeze(2).expand(b, n, self.num_partners, self.out_dim),
            dim=1,
            index=partners_idx,
        )  # (B, N, K, out_dim)

        # Weighted sum over K partners.
        weighted = (gathered * attn.unsqueeze(-1)).sum(dim=2)  # (B, N, out_dim)

        # Reshape back to (B, out_dim, H, W).
        return weighted.transpose(1, 2).reshape(b, self.out_dim, h, w)
