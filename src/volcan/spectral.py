"""Spectral tension channel — Pillar 1c.

Computes the lowest K eigenvectors of the grid Laplacian and projects the cell
state's color channels onto that basis to produce a small global summary that
every cell can read on every update step.

This is the cheap baseline global mechanism. The doc (architecture.md §9c)
notes that if the mycelial bypass (§9b) subsumes its expressivity in ablations,
we drop this. Until then, we keep it.

Eigenvectors are precomputed once per grid size and cached on the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def grid_laplacian(h: int, w: int) -> torch.Tensor:
    """Build the dense Laplacian L = D - A of an H×W grid graph (4-connectivity)."""
    n = h * w
    L = torch.zeros(n, n, dtype=torch.float32)
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            deg = 0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    nidx = ni * w + nj
                    L[idx, nidx] = -1
                    deg += 1
            L[idx, idx] = deg
    return L


def lowest_eigenvectors(L: torch.Tensor, k: int) -> torch.Tensor:
    """Return the k lowest eigenvectors of a symmetric Laplacian L.

    Skips the trivial all-ones eigenvector at eigenvalue 0 if k+1 fit, returning
    eigenvectors 1..k (the informative low-frequency modes).
    """
    eigvals, eigvecs = torch.linalg.eigh(L)
    # Skip the trivial constant mode at index 0; take the next k.
    return eigvecs[:, 1 : k + 1].contiguous()  # (n, k)


class SpectralTension(nn.Module):
    """Pillar 1c: project color channels onto K low-frequency Laplacian modes,
    pass through a small MLP head, broadcast to every cell.

    Args:
        max_grid_size: side length to precompute the eigenbasis for. We always
            run on padded 30×30 grids in v0.2, so this is 30 by default.
        num_modes: K — how many low-frequency eigenvectors to use.
        vocab_size: number of color channels (the source of the projection).
        out_dim: dimensionality of the broadcast tension vector.
    """

    def __init__(
        self,
        max_grid_size: int = 30,
        num_modes: int = 16,
        vocab_size: int = 11,
        out_dim: int = 16,
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.num_modes = num_modes
        self.vocab_size = vocab_size
        self.out_dim = out_dim

        # Precompute and cache the eigenbasis for the canonical grid size.
        L = grid_laplacian(max_grid_size, max_grid_size)
        V = lowest_eigenvectors(L, num_modes)  # (N, K)
        self.register_buffer("eigvecs", V)  # (N, K)

        # Small MLP head: project (vocab_size * num_modes) coefficients down to out_dim.
        self.head = nn.Sequential(
            nn.Linear(vocab_size * num_modes, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, color_logits: torch.Tensor) -> torch.Tensor:
        """Compute the global tension vector from the color channel logits.

        Args:
            color_logits: (B, vocab_size, H, W) — H and W must equal max_grid_size.

        Returns:
            tension: (B, out_dim) — broadcast as a global summary.
        """
        b, v, h, w = color_logits.shape
        if h != self.max_grid_size or w != self.max_grid_size:
            raise ValueError(
                f"SpectralTension expected {self.max_grid_size}×{self.max_grid_size}, got {h}×{w}"
            )
        if v != self.vocab_size:
            raise ValueError(f"expected {self.vocab_size} vocab channels, got {v}")

        # Project each color channel onto the K eigenmodes.
        flat = color_logits.reshape(b, v, h * w)         # (B, V, N)
        coeffs = flat @ self.eigvecs                     # (B, V, K)
        coeffs = coeffs.reshape(b, v * self.num_modes)   # (B, V*K)
        return self.head(coeffs)                          # (B, out_dim)
