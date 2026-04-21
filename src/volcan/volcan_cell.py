"""Volcan cell — the five-pillar Neural Cellular Automaton.

This is the heart of Volcan. Architecture lives here:

  - Per-cell state: 11 color + 32 ghost + 16 hidden = 59 stored channels (+ 2
    position dims that are derived, not stored).
  - Anisotropic update rule: each cell outputs both a state delta AND 8 outgoing
    force messages, one per Moore neighbor (Pillar 1a).
  - Force routing: each cell receives 8 incoming force messages on the next step
    (one from each neighbor), with boundary masking.
  - Mycelial bypass: per-step aggregated long-range message (Pillar 1b).
  - Spectral tension: global summary broadcast every step (Pillar 1c).
  - Two-phase iteration: Phase A clamps color (ghost evolves alone), Phase B
    unclamps color and runs masked denoising (Pillar 3).

Most things here are deliberately uniform-rule, parameter-shared. Translation
equivariance is preserved within each step (the spectral basis breaks it
globally, but only as a small per-cell summary).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arc import VOCAB_SIZE
from .echo import EchoDetector, EchoReading
from .hierarchy import MacroCells
from .moe import MoEUpdateMLP
from .mycelial import MycelialAttention
from .objectness import compute_object_field
from .spectral import SpectralTension


# 8 Moore-neighbor outgoing directions and their (di, dj) shifts.
# direction d at sender (i, j) sends to receiver (i + di, j + dj).
# The receiver stores the message at incoming-direction (d + 4) % 8 (the opposite).
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # 0 N
    (-1, 1),   # 1 NE
    (0, 1),    # 2 E
    (1, 1),    # 3 SE
    (1, 0),    # 4 S
    (1, -1),   # 5 SW
    (0, -1),   # 6 W
    (-1, -1),  # 7 NW
)
NUM_DIRECTIONS = len(DIRECTIONS)


def route_forces(outgoing: torch.Tensor) -> torch.Tensor:
    """Route per-cell outgoing force messages to their destination cells.

    Args:
        outgoing: (B, 8, F, H, W) — outgoing[:, d, :, i, j] is the F-dim message
            cell (i, j) sent in direction d at step t.

    Returns:
        incoming: (B, 8, F, H, W) — incoming[:, d, :, i, j] is the F-dim message
            cell (i, j) received from direction d at step t+1. Boundary cells
            (where the source would have been off-grid) receive zeros.

    Convention: incoming direction d means "received from direction d (in the
    receiver's frame)". A message sent NORTH (sender's d=0) lands at the receiver
    as a message coming from the SOUTH, which is incoming d=4 = (0+4)%8.
    """
    incoming = torch.zeros_like(outgoing)
    for d, (di, dj) in enumerate(DIRECTIONS):
        rolled = torch.roll(outgoing[:, d], shifts=(di, dj), dims=(-2, -1))
        # Zero the wrapped band on the boundary opposite to the shift direction.
        if di == -1:
            rolled[..., -1, :] = 0
        elif di == 1:
            rolled[..., 0, :] = 0
        if dj == -1:
            rolled[..., :, -1] = 0
        elif dj == 1:
            rolled[..., :, 0] = 0
        opposite_d = (d + NUM_DIRECTIONS // 2) % NUM_DIRECTIONS
        incoming[:, opposite_d] = rolled
    return incoming


@dataclass
class VolcanConfig:
    """All hyperparameters for a VolcanCell. See architecture.md §7-9."""

    vocab_size: int = VOCAB_SIZE        # 11 (10 ARC colors + 1 outside)
    ghost_channels: int = 32            # bioelectric latent
    hidden_channels: int = 16           # scratch workspace
    force_dim: int = 4                  # per-direction force vector dim
    mlp_hidden: int = 256               # update MLP hidden width
    mycelial_partners: int = 4
    mycelial_min_distance: int = 6
    mycelial_out_dim: int = 8
    mycelial_topology: str = "symmetric"  # Week 7 (CEO #2): symmetric by default
    spectral_modes: int = 16
    spectral_out_dim: int = 16
    object_embed_dim: int = 4           # Week 7 (CEO #1): object binding dim
    object_seed: int = 42               # RNG seed for the object embedding table
    max_grid_size: int = 30
    mycelial_seed: int = 0
    # Phase 2 Path A (CEO directive): Mixture-of-Experts update MLP.
    # When use_moe=True, the dense update MLP is replaced by N expert MLPs
    # + a router that picks the top-k per cell. Total params ~N×, but FLOPs
    # per cell only scale with top_k (N=4, top_k=2 ≈ same compute as dense).
    # The router is driven by the mycelial attention output.
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_expert_hidden: int = 128        # Per-expert hidden width (smaller than dense mlp_hidden)
    # Post-ceiling experiment (CEO directive): Hierarchical Macro-Cells.
    # When use_hierarchy=True, a separate macro-cell grid sits on top of the
    # 30×30 base grid (10×10 macro cells, each covering a 3×3 base block).
    # The macro cells aggregate local state and broadcast it back to base cells
    # as an additional update-MLP input channel, giving O(1)-iteration local-
    # region context to every base cell. Targets the information bottleneck.
    use_hierarchy: bool = False
    macro_channels: int = 16
    macro_hidden: int = 32
    macro_block_size: int = 3

    @property
    def state_channels(self) -> int:
        """Total stored channels per cell (excludes derived position)."""
        return self.vocab_size + self.ghost_channels + self.hidden_channels  # 59

    @property
    def num_outgoing_force_channels(self) -> int:
        return NUM_DIRECTIONS * self.force_dim  # 32

    @property
    def update_input_dim(self) -> int:
        """Input dimensionality of the update MLP per cell."""
        # 3×3 neighborhood (already includes self) of state channels
        # + incoming forces (8 × force_dim)
        # + mycelial message
        # + spectral tension
        # + 2-dim position
        # + object embedding channels (Week 7)
        # + macro-cell broadcast channels if hierarchy is enabled (post-ceiling)
        neighborhood = 9 * self.state_channels
        base = (
            neighborhood
            + self.num_outgoing_force_channels  # incoming has same shape as outgoing
            + self.mycelial_out_dim
            + self.spectral_out_dim
            + 2
            + self.object_embed_dim
        )
        if self.use_hierarchy:
            base += self.macro_channels
        return base

    @property
    def update_output_dim(self) -> int:
        """Output dimensionality of the update MLP: state delta + outgoing forces."""
        return self.state_channels + self.num_outgoing_force_channels


def make_position_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Per-cell normalized (i, j) position encoding, shape (1, 2, H, W)."""
    yi = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
    xi = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
    return torch.cat([yi, xi], dim=1)  # (1, 2, H, W)


class VolcanCell(nn.Module):
    """The five-pillar morphogenetic cell.

    Forward `step` runs one update of the cell. Forward `phase_a` and `phase_b`
    run the two-phase protocol from architecture.md §11.
    """

    def __init__(self, config: VolcanConfig | None = None):
        super().__init__()
        self.cfg = config or VolcanConfig()
        cfg = self.cfg

        # Components.
        self.spectral = SpectralTension(
            max_grid_size=cfg.max_grid_size,
            num_modes=cfg.spectral_modes,
            vocab_size=cfg.vocab_size,
            out_dim=cfg.spectral_out_dim,
        )
        self.mycelial = MycelialAttention(
            max_grid_size=cfg.max_grid_size,
            num_partners=cfg.mycelial_partners,
            min_distance=cfg.mycelial_min_distance,
            in_channels=cfg.state_channels,
            out_dim=cfg.mycelial_out_dim,
            seed=cfg.mycelial_seed,
            topology=cfg.mycelial_topology,
        )

        # Hierarchical macro-cells (post-ceiling experiment). When enabled, the
        # caller maintains a macro-state tensor across NCA steps; this module
        # produces the per-step update + broadcast field.
        if cfg.use_hierarchy:
            self.macro = MacroCells(
                base_channels=cfg.state_channels,
                macro_channels=cfg.macro_channels,
                macro_hidden=cfg.macro_hidden,
                block_size=cfg.macro_block_size,
                base_grid_size=cfg.max_grid_size,
            )
        else:
            self.macro = None

        # Update MLP. Implemented as 1×1 convolutions so we get
        # parameter sharing across cells for free and stay channel-first.
        if cfg.use_moe:
            # Phase 2 Path A: Mixture-of-Experts update MLP. Router is driven
            # by the mycelial attention output (cfg.mycelial_out_dim channels).
            # MoEUpdateMLP handles its own expert-final-layer init internally.
            self.update = MoEUpdateMLP(
                in_channels=cfg.update_input_dim,
                hidden=cfg.moe_expert_hidden,
                out_channels=cfg.update_output_dim,
                num_experts=cfg.moe_num_experts,
                top_k=cfg.moe_top_k,
                router_in_channels=cfg.mycelial_out_dim,
            )
        else:
            self.update = nn.Sequential(
                nn.Conv2d(cfg.update_input_dim, cfg.mlp_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(cfg.mlp_hidden, cfg.mlp_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(cfg.mlp_hidden, cfg.update_output_dim, kernel_size=1),
            )
            # Final-layer init: proper Kaiming, not the Mordvintsev zero trick.
            # The zero trick works for BasicNCA because its initial state IS the
            # input — gradients have structure to grab. Volcan's Phase B starts
            # from a partially-masked init, which is more symmetric, and zero-init
            # leaves the dynamics dead with no signal to break the symmetry.
            nn.init.kaiming_normal_(self.update[-1].weight, a=0.1)
            with torch.no_grad():
                self.update[-1].weight.mul_(0.1)
            if self.update[-1].bias is not None:
                nn.init.zeros_(self.update[-1].bias)

        # Cached position grid (per-device, lazy).
        self._position_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Channel slicing helpers
    # ------------------------------------------------------------------

    @property
    def color_slice(self) -> slice:
        return slice(0, self.cfg.vocab_size)

    @property
    def ghost_slice(self) -> slice:
        return slice(self.cfg.vocab_size, self.cfg.vocab_size + self.cfg.ghost_channels)

    @property
    def hidden_slice(self) -> slice:
        return slice(
            self.cfg.vocab_size + self.cfg.ghost_channels,
            self.cfg.state_channels,
        )

    def color_logits(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, self.color_slice]

    def ghost(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, self.ghost_slice]

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def init_state(self, input_onehot: torch.Tensor) -> torch.Tensor:
        """Build a zero-initialized state from an input one-hot color tensor.

        Args:
            input_onehot: (B, vocab_size, H, W).

        Returns:
            state: (B, state_channels, H, W) — color = input, ghost = 0, hidden = 0.
        """
        b, v, h, w = input_onehot.shape
        if v != self.cfg.vocab_size:
            raise ValueError(f"expected {self.cfg.vocab_size} color channels, got {v}")
        device = input_onehot.device
        state = torch.zeros(
            b, self.cfg.state_channels, h, w, device=device, dtype=input_onehot.dtype
        )
        state[:, self.color_slice] = input_onehot
        return state

    def init_forces(self, state: torch.Tensor) -> torch.Tensor:
        """Zero-initialized incoming force tensor (B, 8, force_dim, H, W)."""
        b, _, h, w = state.shape
        return torch.zeros(
            b, NUM_DIRECTIONS, self.cfg.force_dim, h, w, device=state.device, dtype=state.dtype
        )

    def _position(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._position_cache:
            self._position_cache[key] = make_position_grid(h, w, device)
        return self._position_cache[key]

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def _gather_neighborhood(self, state: torch.Tensor) -> torch.Tensor:
        """Gather a 3×3 neighborhood of state at every cell.

        Returns: (B, 9 * state_channels, H, W).
        """
        # F.unfold gives us per-cell flattened patches.
        b, c, h, w = state.shape
        patches = F.unfold(state, kernel_size=3, padding=1)  # (B, C*9, H*W)
        return patches.reshape(b, c * 9, h, w)

    def step(
        self,
        state: torch.Tensor,
        incoming_forces: torch.Tensor,
        object_field: torch.Tensor,
        *,
        clamp_color: bool = False,
        clamped_color: torch.Tensor | None = None,
        macro_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One Volcan update step.

        Args:
            state: (B, state_channels, H, W).
            incoming_forces: (B, 8, force_dim, H, W).
            object_field: (B, object_embed_dim, H, W) — static per-cell object
                embedding computed at input time (Week 7, CEO #1).
            clamp_color: if True, the color channels are reset to `clamped_color`
                after the update (Phase A behavior — ghost evolves while color
                stays pinned to the input).
            clamped_color: required if clamp_color is True; (B, vocab_size, H, W).
            macro_state: when hierarchy is enabled, the caller-owned macro-cell
                state tensor of shape (B, macro_channels, macro_H, macro_W).
                Updated each step and fed into the update MLP via broadcast.

        Returns:
            new_state, new_incoming_forces, new_macro_state. The macro state
            is None when `self.macro is None`.
        """
        b, _, h, w = state.shape
        cfg = self.cfg

        # Gather inputs.
        neighborhood = self._gather_neighborhood(state)                         # (B, 9*S, H, W)
        forces_flat = incoming_forces.reshape(b, NUM_DIRECTIONS * cfg.force_dim, h, w)
        mycelial_msg = self.mycelial(state)                                     # (B, M, H, W)
        tension_vec = self.spectral(state[:, self.color_slice])                 # (B, T)
        tension_field = tension_vec.view(b, cfg.spectral_out_dim, 1, 1).expand(
            b, cfg.spectral_out_dim, h, w
        )
        position = self._position(h, w, state.device).expand(b, 2, h, w)

        # Hierarchical macro-cell update + broadcast. When hierarchy is enabled
        # the macro state flows alongside the base state across steps; the
        # per-step update returns a new macro state and a broadcast field that
        # every base cell reads as additional update-MLP input.
        new_macro_state: torch.Tensor | None = None
        if self.macro is not None:
            if macro_state is None:
                macro_state = self.macro.init_macro_state(state)
            new_macro_state, macro_broadcast = self.macro(state, macro_state)
            update_input = torch.cat(
                [neighborhood, forces_flat, mycelial_msg, tension_field, position,
                 object_field, macro_broadcast],
                dim=1,
            )
        else:
            update_input = torch.cat(
                [neighborhood, forces_flat, mycelial_msg, tension_field, position, object_field],
                dim=1,
            )

        # Run the MLP. Output is (B, state_delta + outgoing_forces, H, W).
        # When use_moe=True, the update module is an MoEUpdateMLP that takes
        # an extra router_input argument (the mycelial attention output). The
        # router decides which expert handles each cell based on long-range
        # context, making specialization task-aware.
        if isinstance(self.update, MoEUpdateMLP):
            update_output = self.update(update_input, mycelial_msg)
        else:
            update_output = self.update(update_input)
        state_delta = update_output[:, : cfg.state_channels]
        outgoing_flat = update_output[:, cfg.state_channels :]
        outgoing = outgoing_flat.reshape(b, NUM_DIRECTIONS, cfg.force_dim, h, w)

        # Residual state update. The color channels are treated as UNCONSTRAINED
        # LOGITS, not as a probability distribution — we never softmax inside the
        # iteration loop, because that squashes the signal toward uniform every
        # step and the model can't preserve information across many iterations.
        # The cross-entropy loss applies log_softmax internally; the public
        # `color_logits` accessor returns these raw values.
        new_state = state + state_delta

        if clamp_color:
            assert clamped_color is not None, "clamp_color=True requires clamped_color"
            new_state = torch.cat(
                [clamped_color, new_state[:, self.cfg.vocab_size :]], dim=1
            )

        # Route outgoing forces for the next step.
        new_incoming = route_forces(outgoing)

        return new_state, new_incoming, new_macro_state

    # ------------------------------------------------------------------
    # Phase A — Ghost Dream
    # ------------------------------------------------------------------

    def phase_a(
        self,
        input_onehot: torch.Tensor,
        *,
        max_steps: int = 48,
        min_steps: int = 16,
        echo_threshold: float = 0.97,
        echo_window: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Run Phase A (Ghost Dream) until the ghost field stabilizes.

        The color channels are clamped to `input_onehot` for the entire phase.
        Only ghost, hidden, and force channels evolve.

        Returns:
            final_state, final_forces, object_field, steps_taken.
            The object_field is the static pre-segmentation used throughout
            Phase A; callers must thread it into Phase B to keep segmentation
            consistent across phases.
        """
        state = self.init_state(input_onehot)
        forces = self.init_forces(state)
        clamped = input_onehot

        # Week 7 CEO #1: pre-segment the input and build a per-cell object
        # embedding field that stays constant throughout Phase A and Phase B.
        object_field = compute_object_field(
            input_onehot,
            embed_dim=self.cfg.object_embed_dim,
            seed=self.cfg.object_seed,
        ).to(input_onehot.device)

        ghost_detector = EchoDetector(max_lag=echo_window)
        consecutive_stable = 0
        # Macro state is internal to Phase A; None if hierarchy disabled.
        macro_state: torch.Tensor | None = None

        for t in range(max_steps):
            state, forces, macro_state = self.step(
                state, forces, object_field,
                clamp_color=True, clamped_color=clamped,
                macro_state=macro_state,
            )
            ghost = self.ghost(state)
            if t >= min_steps:
                reading = ghost_detector.echo(ghost)
                if (reading.echo_1 > echo_threshold).all():
                    consecutive_stable += 1
                    if consecutive_stable >= echo_window:
                        return state, forces, object_field, t + 1
                else:
                    consecutive_stable = 0
            ghost_detector.push(ghost)

        return state, forces, object_field, max_steps

    # ------------------------------------------------------------------
    # Phase A (ICL variant) — sequential demo encoding
    # ------------------------------------------------------------------

    def phase_a_icl(
        self,
        demo_inputs: torch.Tensor,   # (B, K, V, H, W) one-hot per demo input
        demo_outputs: torch.Tensor,  # (B, K, V, H, W) one-hot per demo output
        query_input: torch.Tensor,   # (B, V, H, W) one-hot
        *,
        steps_per_clamp: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """In-context learning Phase A.

        For each batch element, the model sees K demo (input, output) pairs as
        context and one query input. The ghost field accumulates rule
        information across all demos before being applied to the query.

        Protocol per batch element:
          - Initialize state from the first demo input.
          - For each demo k:
              * Sub-phase k.in: clamp color to demo_inputs[k], run `steps_per_clamp` updates
              * Sub-phase k.out: clamp color to demo_outputs[k], run `steps_per_clamp` updates
          - Sub-phase Q: clamp color to query_input, run `steps_per_clamp` updates

        The total number of NCA updates is K * 2 * steps_per_clamp + steps_per_clamp.

        Returns:
            final_state: (B, state_channels, H, W) — color = query_input,
                ghost = accumulated rule field, hidden = workspace
            forces: (B, 8, F, H, W)
            object_field: (B, object_embed_dim, H, W) — computed from the
                query input, reused by Phase B
            total_steps: int
        """
        b, k, v, h, w = demo_inputs.shape
        if v != self.cfg.vocab_size:
            raise ValueError(f"expected {self.cfg.vocab_size} color channels, got {v}")
        if demo_outputs.shape != demo_inputs.shape:
            raise ValueError(
                f"demo_outputs shape {demo_outputs.shape} must match demo_inputs {demo_inputs.shape}"
            )
        if query_input.shape != (b, v, h, w):
            raise ValueError(
                f"query_input shape {query_input.shape} must be (B={b}, V={v}, H={h}, W={w})"
            )

        # Initialize the cell state from the first demo input.
        state = self.init_state(demo_inputs[:, 0])
        forces = self.init_forces(state)
        total = 0

        # Week 7 CEO #1: compute the object field from the QUERY input. The
        # query is what Phase B will actually denoise, so it's the segmentation
        # that matters for the output. Demo-level object structure is still
        # indirectly seen by the model via the per-demo color clamp + ghost
        # evolution.
        object_field = compute_object_field(
            query_input,
            embed_dim=self.cfg.object_embed_dim,
            seed=self.cfg.object_seed,
        ).to(query_input.device)

        # Macro state is internal to Phase A ICL; None if hierarchy disabled.
        macro_state: torch.Tensor | None = None

        for demo_idx in range(k):
            # Sub-phase k.in: clamp color to demo k's input, evolve ghost.
            clamp_in = demo_inputs[:, demo_idx]
            for _ in range(steps_per_clamp):
                state, forces, macro_state = self.step(
                    state, forces, object_field,
                    clamp_color=True, clamped_color=clamp_in,
                    macro_state=macro_state,
                )
                total += 1
            # Sub-phase k.out: clamp color to demo k's output, evolve ghost.
            clamp_out = demo_outputs[:, demo_idx]
            for _ in range(steps_per_clamp):
                state, forces, macro_state = self.step(
                    state, forces, object_field,
                    clamp_color=True, clamped_color=clamp_out,
                    macro_state=macro_state,
                )
                total += 1

        # Sub-phase Q: clamp color to the query input, evolve ghost.
        for _ in range(steps_per_clamp):
            state, forces, macro_state = self.step(
                state, forces, object_field,
                clamp_color=True, clamped_color=query_input,
                macro_state=macro_state,
            )
            total += 1

        return state, forces, object_field, total

    # ------------------------------------------------------------------
    # Phase B — Crystallization
    # ------------------------------------------------------------------

    def phase_b(
        self,
        ghost_state: torch.Tensor,
        forces: torch.Tensor,
        *,
        object_field: torch.Tensor | None = None,
        init_color: torch.Tensor | None = None,
        max_steps: int = 40,
        detach_ghost: bool = False,
        collect_activity: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[EchoReading], list[torch.Tensor]]:
        """Run Phase B (Crystallization) — color denoises while ghost is frozen.

        Args:
            ghost_state: (B, state_channels, H, W) from Phase A.
            forces: (B, 8, F, H, W) from Phase A.
            init_color: optional explicit color initialization. If None, the
                color channels are reset to a uniform distribution.
            max_steps: max iterations of Phase B.
            detach_ghost: if True, the ghost/hidden channels from Phase A are
                detached so gradients don't flow back through Phase A. The doc
                spec'd this (Phase A and Phase B as separable phases), but in
                early training the chicken-and-egg dynamic is too severe — we
                default to False so gradients flow end-to-end through both
                phases. Once the model is trained we can re-enable detach for
                inference-time efficiency.
        """
        b, _, h, w = ghost_state.shape
        cfg = self.cfg

        if detach_ghost:
            ghost_only = ghost_state[:, self.ghost_slice].detach()
            hidden_only = ghost_state[:, self.hidden_slice].detach()
        else:
            ghost_only = ghost_state[:, self.ghost_slice]
            hidden_only = ghost_state[:, self.hidden_slice]

        if init_color is None:
            init_color = torch.full(
                (b, cfg.vocab_size, h, w),
                1.0 / cfg.vocab_size,
                device=ghost_state.device,
                dtype=ghost_state.dtype,
            )

        # If the caller didn't provide an object field, compute one from the
        # init_color (Phase B may be called standalone in some tests). Normal
        # training/inference flow passes it down from Phase A.
        if object_field is None:
            object_field = compute_object_field(
                init_color,
                embed_dim=self.cfg.object_embed_dim,
                seed=self.cfg.object_seed,
            ).to(init_color.device)

        state = torch.cat([init_color, ghost_only, hidden_only], dim=1)
        # Forces start fresh in Phase B; ghost-phase forces aren't directly
        # meaningful for color denoising and would otherwise force-couple the
        # two phases.
        forces = self.init_forces(state)

        echo_detector = EchoDetector(max_lag=4)
        echoes: list[EchoReading] = []
        deltas: list[torch.Tensor] = []
        # Phase B owns its own macro state (fresh, not inherited from Phase A).
        # Phase A's macro tracks "rule under input clamp"; Phase B's macro
        # tracks "evolving denoising field". Different purposes, separate state.
        macro_state: torch.Tensor | None = None

        for _ in range(max_steps):
            prev_state = state
            state, forces, macro_state = self.step(
                state, forces, object_field,
                clamp_color=False, macro_state=macro_state,
            )
            if collect_activity:
                deltas.append(state - prev_state)
            color_now = state[:, self.color_slice]
            reading = echo_detector.echo(color_now)
            echoes.append(reading)
            echo_detector.push_with_grad(color_now)

        return state, forces, echoes, deltas

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
