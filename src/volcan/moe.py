"""Mixture-of-Experts update MLP (Phase 2 Path A — CEO directive).

Measured bottleneck: at 111K params, Volcan's pretraining accuracy plateaus at
76% regardless of corpus size (43 tasks or 325 tasks → same score). The
update MLP is the bottleneck — it's where per-cell transformation logic lives.

MoE solves this the right way for our "efficient AI" framing: we replace the
single dense MLP with N smaller expert MLPs and a router that picks the top-k
for each cell. Total parameter count goes up ~N× but FLOPs-per-cell stay
roughly constant (only top_k experts run).

Biology framing: specialization. Different "tissue regions" of the NCA
develop specialized update rules — one expert for symmetry, one for gravity,
one for color transforms, etc. The router (driven by mycelial attention in
our implementation) decides which expert handles each cell based on
long-range context.

Design choices:
- **Number of experts**: 4 (small enough to train, large enough to specialize)
- **Top-k**: 2 (soft routing with full gradient flow vs top-1's discontinuity)
- **Router input**: the mycelial-attention output (8-dim) projected to
  num_experts logits. This makes expert choice depend on long-range structure,
  which is the whole point of having mycelial attention in the first place.
- **Load-balancing loss**: standard MoE trick — penalize routers that always
  pick the same expert. Prevents dead experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEUpdateMLP(nn.Module):
    """N experts + top-k router, drop-in replacement for a dense update MLP.

    Forward signature matches the original Sequential (takes B,C_in,H,W,
    returns B,C_out,H,W). The router signal is passed separately so the
    caller controls where it comes from.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        *,
        num_experts: int = 4,
        top_k: int = 2,
        router_in_channels: int = 8,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.in_channels = in_channels
        self.hidden = hidden
        self.out_channels = out_channels

        # N small expert MLPs, each a 3-layer 1x1 conv stack.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden, out_channels, kernel_size=1),
                )
                for _ in range(num_experts)
            ]
        )
        # Initialize the FINAL layer of each expert with small random weights.
        # (Matches the main VolcanCell's original init philosophy — neutral
        # initial dynamics but non-zero so symmetry breaks.)
        for expert in self.experts:
            final_conv = expert[-1]
            nn.init.kaiming_normal_(final_conv.weight, a=0.1)
            with torch.no_grad():
                final_conv.weight.mul_(0.1)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

        # Router: takes a small per-cell context vector (e.g. mycelial message)
        # and outputs num_experts logits per cell. 1×1 conv keeps it per-pixel.
        self.router = nn.Conv2d(router_in_channels, num_experts, kernel_size=1)

        # Track the last routing decision so losses can inspect it
        # (for load-balancing).
        self._last_gate: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        router_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — full input to the update MLP
            router_input: (B, router_in_channels, H, W) — small per-cell context
                used to decide expert routing. In Volcan we pass the mycelial
                attention output here.

        Returns:
            (B, out_channels, H, W) — weighted sum of top-k experts' outputs.
        """
        b, _, h, w = x.shape

        # --- Router ---
        # logits: (B, num_experts, H, W)
        logits = self.router(router_input)

        # Top-k gating: for each cell, pick the top_k experts and renormalize.
        # We keep the non-top-k logits at -inf so they get zero weight after softmax.
        topk_values, topk_indices = logits.topk(self.top_k, dim=1)
        # Build a mask that's True only for the top_k experts per cell.
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, topk_indices, topk_values)
        gates = F.softmax(mask, dim=1)  # (B, num_experts, H, W), top_k nonzero

        # Cache for load-balancing loss (pre-softmax logits as well are useful).
        self._last_gate = gates

        # --- Experts (only the active ones for this pixel contribute) ---
        # We compute each expert's full output but then weight by gates.
        # This is wasteful relative to a proper sparse dispatch, but for
        # N=4 experts and our grid sizes (30×30 at most) the overhead is
        # tiny vs the clarity gain. Proper sparse MoE is for 8B-param models.
        output = torch.zeros(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            gate_i = gates[:, i : i + 1]  # (B, 1, H, W)
            # Skip computing the expert if this expert's gate is exactly zero everywhere
            # (happens when it's not in any cell's top-k — saves FLOPs).
            if (gate_i == 0).all():
                continue
            expert_out = expert(x)  # (B, out_channels, H, W)
            output = output + gate_i * expert_out

        return output

    def load_balancing_loss(self) -> torch.Tensor:
        """Auxiliary loss that penalizes unbalanced expert usage.

        Uses the "importance" formulation: the per-batch mean of each expert's
        gate weights should be uniform (= 1 / num_experts). Variance of the
        mean gate weights across experts is the loss — zero when perfectly
        balanced, higher when one expert dominates.

        Call this after a forward pass; it uses the gates from that pass.
        """
        if self._last_gate is None:
            return torch.tensor(0.0)
        # mean gate weight per expert across (batch, spatial)
        mean_gate = self._last_gate.mean(dim=(0, 2, 3))  # (num_experts,)
        # Target: 1 / num_experts for each. Penalize squared deviation.
        target = 1.0 / self.num_experts
        return ((mean_gate - target) ** 2).mean()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_active_params(self) -> int:
        """Params active in a forward pass (router + top_k experts)."""
        router_params = sum(p.numel() for p in self.router.parameters())
        per_expert = sum(p.numel() for p in self.experts[0].parameters())
        return router_params + self.top_k * per_expert
