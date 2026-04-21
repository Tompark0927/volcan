# Volcan — Architecture Design Document

**Project:** Volcan — ARC-AGI-2 via Morphogenetic Cellular Automata
**Date:** 2026-04-11
**Version:** 0.1 (first draft, pending CEO review)
**Working title of the paper:** *Morphogenetic Cellular Automata with Tensegral Global Coupling and Bioelectric Latent Fields for Abstract Reasoning*

---

## 1. Mission

Build a Neural Cellular Automaton that solves ARC-AGI-2 puzzles by emergent morphogenesis — a small shared update rule applied locally across a 2D grid of cells, augmented with:

1. A global **tensegrity** coupling channel (instant long-range information flow, via Laplacian spectral basis),
2. A per-cell **bioelectric latent field** (rule-level abstraction in spatial correspondence with the grid),
3. A **self-verification** signal at inference time (dynamical stability + cycle consistency),
4. An **MDL regularizer** on the latent field (Occam's razor, formalized).

Pretrained on ARC-2-targeted synthetic data, fine-tuned per-task at inference via LoRA adaptation of the update rule, deployed inside the Kaggle compute envelope (4× T4/P100, 12h, no internet, ~$0.42/task).

## 2. Thesis (one sentence)

> The ARC-AGI-2 frontier is bottlenecked not by model size but by candidate-generator diversity on compositionally novel tasks, and a morphogenetic architecture with principled global information flow solves this at a fraction of the compute current Kaggle winners use.

## 3. Design philosophy

- **CEO sets direction; engineer makes it buildable.** Every architectural choice traces to either (a) one of the CEO's four biological principles, or (b) a measured weakness in published prior art.
- **Every claim earns its place with a number.** No component ships without an ablation against removing it.
- **Measure on ARC-AGI-2 public eval from Week 1, not from Week 10.** Self-delusion is the #1 failure mode of ambitious research projects.
- **Disagreement is written down.** Anything the engineer argued against but the CEO overruled (and vice versa) is noted in this doc with "*Resolved: [who], [date], [why]*" so we know later whose call it was.

## 4. The enemy — who we are trying to beat

| System | Params | ARC-1 | ARC-2 | Notes |
|---|---|---|---|---|
| **TRM** (Jolicoeur-Martineau, Samsung SAIL, Oct 2025) | 7 M | 45% | **8%** | Iterative refinement of 2D state with global self-attention channel. 1st place ARC Prize 2025 Paper Award ($50K). **Direct architectural competitor.** |
| **ARC-NCA** (Guichard et al., ALIFE 2025) | ~100 K | 12.9% (17.6% ensemble) | — | Pure local NCA, fresh weights per task, no pretraining. Paper Award runner-up. **The strongest NCA baseline.** |
| **NVARC** (NVIDIA KGMoN, Kaggle 2025) | ~4 B | — | **24.03%** | Qwen3-4B + 103K synthetic tasks + TTT + ARChitects pipeline. $0.20/task on Kaggle. **The Kaggle SOTA on ARC-2.** |
| **ARChitects 2025** (masked diffusion) | ~8 B | — | 16.5% | 2D-aware masked-diffusion LM. 2nd place Kaggle 2025. |
| **MindsAI 2025** | ~8 B | 55% (2024) | 12.6% | TTT + augmentation ensembles. 3rd place Kaggle 2025. |
| **Grand Prize threshold** | — | — | 85% (private) | Unclaimed. $700K. |
| **Volcan minimum viable** | ≤ 2 M | — | ≥ 5% | Proves the pipeline works end-to-end. |
| **Volcan target** | ≤ 2 M | — | **≥ 8%** | Matches TRM at smaller param count. Paper-award credible. |
| **Volcan stretch** | ≤ 2 M | — | ≥ 15% | Beats TRM, closes to within 10pp of NVARC, Kaggle top-5 credible. |

**Primary target: TRM.** Volcan and TRM occupy the same architectural niche (small-param iterative refinement), so a head-to-head win is both technically meaningful and a clean paper story. If we can beat TRM at ≤ 2M params on ARC-2, we have a real result regardless of where we land on Kaggle.

## 5. The four pillars — CEO's biology → computational primitives

Each of the CEO's four biological ideas maps to a specific, implementable computational mechanism. This table is load-bearing — when we're debugging in Week 8 and something breaks, we come back here to verify we haven't silently dropped a pillar.

| # | CEO's biology | Computational form | Why it's load-bearing |
|---|---|---|---|
| 1 | **Living Tensegrity** — tension propagates across the whole cell body instantly | Global coupling channel: project cell states onto the K lowest eigenvectors of the grid Laplacian; broadcast the projection coefficients (a small "tension vector") back to every cell as additional input to the local update rule. | **Fixes the known failure mode of all prior NCAs on ARC.** CAX 1D-ARC paper showed pure-local NCAs get 100% on Move but 0% on Count/Compare. Tensegrity gives O(1)-iteration global information flow. Without this pillar, Volcan caps at ~17% on ARC-1 and probably <5% on ARC-2 (prior art's numbers). |
| 2 | **Time-Crystal Clock** — correct rules resonate, wrong rules produce noise | Self-verification at test time via **dynamical stability** (state stops changing under further iteration) + **forward-backward cycle consistency** (forward rule maps input→output and an inverse mode recovers input from output). | Gives us a confidence signal **without needing the ground truth**. This becomes our candidate-ranking mechanism at test time — lets us select among multiple TTT runs / augmentations without a separate voting model. ARChitects' product-of-experts does something analogous; we do it from the dynamics themselves. |
| 3 | **Ghost Grid** — bioelectric pre-pattern (Levin lab, salamander regeneration) | Per-cell latent channel (~32 dimensions) evolving in parallel to the color channels. Loss is computed only on the color channels; the ghost channels are a free workspace the NCA can use for intermediate rule-level state. | **Structurally more expressive than TRM.** TRM has a single latent `z` for the whole puzzle; ours has a latent field with spatial correspondence to the grid. Position (i,j) gets its own latent "what rule is active here." This is the single place where we are architecturally *above* TRM, not just different from it. |
| 4 | **Aestivation** — metabolic entropy minimization during sleep/dreaming | MDL regularizer on the ghost channels (L1 sparsity + transition-entropy penalty), plus a stability loss term that rewards fixed-point convergence. Cross-entropy is still the primary loss; MDL is a regularizer, not a replacement. | **Soft Occam's razor.** Rewards solutions that need a simple latent configuration. CompressARC (3rd place Paper Award 2025) validates this direction on ARC-1 with a 76K-param MDL-trained network. We're importing their loss philosophy, not their architecture. |

### Pillars we are NOT including and why (be explicit)

- **Literal Kuramoto phase oscillators.** Hard to backprop through trig across many timesteps. *Resolved: engineer, 2026-04-11, research risk; the underlying idea is preserved as stability + cycle consistency.*
- **Pure energy-based training (no cross-entropy).** Classical EBMs require MCMC or contrastive divergence, notoriously unstable, multi-month engineering risk. *Resolved: engineer, 2026-04-11; MDL is captured as a regularizer on top of CE loss.*
- **Literal "instant" tension propagation.** Our spectral-basis coupling is O(1)-iterations but still O(K·N) multiply-adds per step. "Instant" in the CEO's frame means "one iteration, not many," which we do deliver. *Resolved: engineer, 2026-04-11.*

## 6. System overview

```
                        ┌──────────────────────────────┐
                        │  OFFLINE (before Kaggle)      │
                        │                              │
   ARC public train ──┬─┤  Synthetic data pipeline:    │
   ARC public eval ───┤ │    100K ARC-2-targeted tasks │
                      │ │    (§10)                     │
                      │ │                              │
                      │ │              ↓               │
                      │ │  Pretrain Volcan update rule │
                      │ │  (§7-9)  ~1 week 1×H100      │
                      │ │              ↓               │
                      │ │  volcan_base.pt (~2 MB)      │
                      │ └──────────────┬───────────────┘
                      │                │
                      │                │  ship weights to Kaggle
                      │                ↓
                      │ ┌──────────────────────────────┐
                      │ │  ONLINE (on Kaggle, per task) │
                      │ │                              │
                      │ │  For each private task:      │
                      │ │    1. Load base weights      │
                      │ │    2. Attach LoRA adapter    │
                      │ │    3. TTT on 2-5 demo pairs  │
                      │ │       (~2 min, 200 steps)    │
                      │ │    4. Generate K=8 candidates│
                      │ │       (augmented views)      │
                      │ │    5. Rank by stability +    │
                      │ │       cycle consistency      │
                      │ │    6. Submit top 2           │
                      │ │                              │
                      │ │   ~5 min/task × 120 = 10h    │
                      │ └──────────────────────────────┘
                      │
                      │   Measure: ARC-AGI-2 public eval
                      └───── (continuous, from Week 4)
```

## 7. Cell state specification

Each cell at position (i,j) carries:

| Channel group | Dimensions | Interpretation | Constraint |
|---|---|---|---|
| **Color** | 11 | One-hot-like probabilities over {10 ARC colors + 1 "outside" token} | Softmax output, used for loss and rendering |
| **Ghost** (bioelectric) | 32 | Latent rule-level workspace | Unconstrained float, free |
| **Hidden** (scratch) | 16 | Short-term computation workspace | Unconstrained float, free |
| **Position encoding** | 2 | Normalized (i,j) ∈ [0,1]² | Read-only, not updated |

**Total per-cell state:** 11 + 32 + 16 + 2 = **61 dimensions**.

For a 30×30 grid: 30 × 30 × 61 = 54,900 scalars per state snapshot.

## 8. Update rule (the cell-level neural network)

Architecture: a small MLP applied identically to every cell in parallel (parameter sharing → translation equivariance where we want it).

**Inputs to one cell's update at time t:**

- Own state: 59 dims (color + ghost + hidden, excluding position)
- 3×3 Moore neighborhood of states: 9 × 59 = 531 dims (includes self)
- Global tension vector τ_t: 16 dims (from §9)
- Position encoding: 2 dims

**Total input width:** 531 + 16 + 2 = **549 dims**.

**MLP shape:**
```
  549 → Linear(256) → GELU → Linear(256) → GELU → Linear(59) → delta
```

**Update:**
```
  s_{t+1}^{ij} = s_t^{ij} + f_θ(context_t^{ij})
```

(Residual update, Mordvintsev-style.)

**After the residual update, the color channels are re-softmaxed to maintain valid probabilities.**

**Parameter count:** ~200 K in the MLP + ~5 K in the tension projection head = **~205 K params total**. ~35× smaller than TRM. Well under our 2 M budget; we have headroom for a second attention-over-cells pass if experiments want it.

## 9. Global coupling — the tensegrity channel

**What it is:** a small, differentiable, global summary of the grid state that every cell can read on every update.

**How it's computed:**

1. **One-time precompute (per grid size, cached):**
   - Build the grid graph: N = H×W cells, 4-connectivity (Moore-8 is also possible; 4-conn is standard for Laplacian spectral methods).
   - Compute the unnormalized graph Laplacian L = D - A.
   - Compute the K = 16 lowest eigenvectors {v₁, …, v₁₆}. Stack as matrix V ∈ ℝ^(N × K).
   - Cache V for grid sizes 1×1 through 30×30. (~9 MB total cache.)

2. **Per update step:**
   - For each color channel c ∈ {0..10}, compute projection α_{t,c} = Vᵀ · color_c ∈ ℝ^K. (16-dim vector per color channel.)
   - Flatten: α_t ∈ ℝ^(11×16) = ℝ^176.
   - Small MLP: 176 → 16 → tension vector τ_t ∈ ℝ^16.
   - Broadcast τ_t to every cell as part of the update context (§8).

**Why Laplacian spectral basis and not attention:**

- **Fixed cost, differentiable, translation-invariant by construction.** No learned attention weights means no training instability around that component.
- **Low-frequency eigenmodes are exactly what "tension" means physically.** A stretched membrane oscillates in its Laplacian modes; a biological cell's cytoskeleton transmits strain through similar global modes. The biology and the math agree.
- **Cost: ~O(K·N) per step = ~14,400 multiplies for 30×30 × K=16.** Negligible compared to the MLP update.
- **Precomputation cost is one-time per grid size.** We amortize it across all training and inference.

**What tensegrity gives us that pure-local NCAs lack:** the first iteration already sees a global summary. "Count the red cells" is one projection away. "Is shape A bigger than shape B" is a difference of two projections.

**Alternative considered:** virtual "tension" node in a GNN sense, connected to all cells. Equivalent expressivity, but adds learned parameters in the coupling step (which we wanted to keep fixed). Kept as backup if spectral approach has issues.

## 10. Synthetic data pipeline — ARC-2-targeted

This is the single most important non-architectural component. NVARC's dominant advantage over ARChitects was their 103K synthetic tasks; we need better synthetic data, not just more of it.

Chollet's four ARC-2 cognitive categories (from the ARC-AGI-2 technical report):

1. **Multi-rule reasoning** — simultaneous application of several interacting rules
2. **Multi-step reasoning** — step N depends on outcome of step N-1
3. **Contextual rule application** — rules modulated by global context
4. **In-context symbol definition** — interpreting task-specific symbol meanings on-the-fly

Existing synthetic datasets (RE-ARC, BARC, NVARC's corpus) re-sample ARC-1 concepts. They do not explicitly target these four categories. **This is the data gap.** Closing it is probably worth 5-10 percentage points by itself.

### Generation protocol

For each of the four categories, target ~25 K verified tasks, 100 K total.

**Step 1 — Concept library.** Start from ~100 base primitives (the union of Icecuber's DSL, BARC's concepts, and our own additions). Each primitive is a Python function that takes a grid and returns a grid, with a natural-language description.

**Step 2 — Category-specific composition.**
- **Multi-rule:** sample 2-3 primitives, combine them via concrete operators (AND, THEN, ONLY-IF). Verify by execution.
- **Multi-step:** sample a sequence of primitives where each step's output becomes the next step's input. Verify dependency by permuting the order and checking the result changes.
- **Contextual:** sample a global predicate (e.g., "if the input contains > 5 red cells") and two different rules; assign rules based on the predicate. Verify by constructing demos that exercise both branches.
- **In-context symbol:** sample a symbol definition (e.g., "the blue cell marks the center"), embed it as part of the demo input, then apply a rule that references the definition. Verify the test task can only be solved by reading the definition.

**Step 3 — LLM sanity check.** Feed each generated task to a small local model (or Claude API offline) and ask "what is the rule?" If the LLM cannot recover a sensible rule, we reject the task. This filters for learnability — tasks that are gibberish to a reasonably capable reader are either broken or too cryptic to train on.

**Step 4 — Execution verification.** For each task, run the rule on ≥ 10 different input grids. Accept only if all executions produce valid outputs. This is NVARC's trick: synthetic data that is self-consistent under many inputs.

**Step 5 — Augmentation at load time.** D₈ symmetry group (8) × random color permutations (up to 10!) × demo reordering. Streaming; ~200× effective dataset size.

**Hard requirement:** none of our synthetic tasks may overlap with the ARC-2 public eval set in concept or structure, verified by hash + human spot-check on a random sample.

## 11. Iteration protocol

- **Minimum iterations:** T_min = 16
- **Maximum iterations:** T_max = 48
- **Early termination:** after T_min steps, check per-step L2 state change. If below ε = 1e-4 for 3 consecutive steps, terminate early and use the current state as the output.
- **Deep supervision during training:** compute cross-entropy loss at steps {16, 24, 32, 40, 48} and sum. This stabilizes long-horizon backprop and ensures the network is incentivized to converge fast.

**Memory:** BPTT across 48 steps with activations → gradient checkpointing every 8 steps. Memory cost per training sample scales as ~6× a single forward pass, tractable on a 40 GB GPU at batch size 32.

## 12. Loss function

The heart of how the four biological pillars manifest in training:

```
L_total = L_ce
        + λ₁ · L_stability
        + λ₂ · L_cycle
        + λ₃ · L_mdl
```

### L_ce — cross-entropy on the color channels

Standard classification loss over the 11-way (10 colors + outside) softmax, summed over all cells in the target grid, averaged over deeply-supervised time steps.

### L_stability — dynamical fixed-point detection

After T_max iterations, run 8 additional "coast" iterations without gradient. Measure:

```
L_stability = (1/8) · Σ_{k=1..8} || s_{T_max+k} - s_{T_max} ||²
```

Low when the state has converged; high when the rule is chaotic or drifting. Trains the network to settle into attractors.

### L_cycle — forward-backward consistency

For each demo pair (input, output):
- Forward: run NCA from input → candidate output, compute CE against the true output.
- Reverse: run NCA in "reverse mode" (a learned one-bit flag flipped in the update rule) from true output → candidate input, compute CE against the true input.

```
L_cycle = L_ce(forward(input), output) + L_ce(reverse(output), input)
```

**Known caveat:** some ARC rules are many-to-one (e.g., "output the count of shapes"). The reverse branch will underperform on those. We will empirically decide whether to:
- (a) skip cycle loss for tasks where the output grid dimension is clearly smaller than the input (heuristic filter), or
- (b) use a learned "task is bijective" predictor to gate the cycle loss.

This is an **open question** (§15).

### L_mdl — minimum description length on the ghost field

```
L_mdl = || ghost_channels ||₁ + β · H(transition(ghost_t → ghost_{t+1}))
```

L1 sparsity on the ghost channels + entropy penalty on transitions. Rewards ghost configurations that are sparse and stable.

### Loss weights (initial guesses, to be tuned)

| Weight | Initial value | Scheduling |
|---|---|---|
| λ₁ (stability) | 0.1 | Warm up from 0 over first 5K steps |
| λ₂ (cycle) | 0.1 | Constant |
| λ₃ (MDL) | 0.01 | Warm up from 0 over first 10K steps |

Warm-up prevents the regularizers from dominating CE before the network has learned anything.

## 13. Training pipeline

### Phase 1: Offline pretraining (before Kaggle)

- **Data:** 100 K synthetic ARC-2 tasks (§10) + ARC public train + ARC public eval (~1400 tasks).
- **Batch size:** 32 tasks.
- **Optimizer:** AdamW, lr 1e-3 → 1e-4 cosine decay over 500 K steps.
- **Gradient clipping:** 1.0.
- **Time estimate:** ~1 week on 1×H100 (or equivalent spot capacity).
- **Continuous eval:** every 10 K steps, evaluate on ARC-2 public eval, log score, log per-category breakdown if we can extract categories from the public eval.
- **Deliverable:** `volcan_base.pt` — the pretrained update rule weights (~2 MB).

### Phase 2: Test-time training (Kaggle, per task)

For each private task:

1. Load `volcan_base.pt`.
2. Attach a LoRA adapter to the update rule (rank 16, applied to both MLP layers).
3. Construct a training set from the task's 2-5 demo pairs via leave-one-out + D₈ + color permutation augmentation (~100 variants).
4. Fine-tune the LoRA for 200 gradient steps, lr 5e-4, AdamW.
5. Generate K=8 candidate outputs:
   - For each of 8 augmented views of the test input, run the NCA forward with early-termination enabled.
   - Un-augment and collect.
6. Rank candidates by stability + cycle consistency score (§12).
7. Submit the top-2 candidates.

**Per-task budget:** ~5 minutes. 120 tasks × 5 min = 600 min = 10 h, under the 12 h Kaggle cap.

## 14. Variable-grid-size handling

**The known NCA hole.** Both existing ARC-NCA papers drop 30-35% of tasks because input and output grids have different dimensions.

**Volcan's approach: "outside" as a first-class color.**

- Pad all grids to 30×30 with a reserved "outside" token (color index 10).
- The NCA update rule treats "outside" as a valid state it can write to or read from.
- During training, include tasks whose target output has different dimensions than the input — the NCA learns to turn cells "outside" (apoptosis) or activate cells from "outside" (cell division).
- At inference, the predicted output is the tightest axis-aligned bounding box of non-outside cells.

**Why this should work:** it's *principled* — the "outside" is part of the state space, not a hack. It's the cleanest existing attempt at this problem, and it lets us train on the full 400 public eval tasks instead of the 262-task subset the NCA papers used.

**Risk:** the network may learn to leave cells "outside" too aggressively. Mitigation: augment with a per-task "expected output size" hint as an additional input channel, derived from the demo pairs at training time.

## 15. Open questions (where we don't know yet)

| # | Question | How we'll resolve |
|---|---|---|
| 1 | Is K=16 spectral eigenmodes enough for global coupling? | Ablate K ∈ {4, 8, 16, 32, 64} on a subset of public eval during Week 2. |
| 2 | Does cycle consistency hurt on non-bijective tasks? | Train with and without a bijectivity gate; compare on public eval. |
| 3 | Does the ghost field actually help, or is it just extra parameters? | Ablate by zero-masking the ghost channels at inference. |
| 4 | What's the optimal ratio of pretrain compute to TTT compute? | Sweep TTT steps ∈ {0, 50, 100, 200, 400, 800} on 20-task subset. |
| 5 | Does the stability loss backprop cleanly through 48 steps? | Start with truncated BPTT over 16 steps, extend iteratively. |
| 6 | Does our synthetic data transfer to real ARC-2? | Track public eval score every 10K pretraining steps. If flat or decreasing, the data is wrong. |
| 7 | Is the "outside" token approach good enough for size-change tasks? | Measure accuracy on size-changing subset separately from size-preserving subset. |
| 8 | Will we actually fit in 10h on Kaggle hardware? | Profile Week 1 smoke test on a cloud T4 instance. |

## 16. Risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Training instability from 48-step BPTT | High | Medium | Gradient clipping, truncated BPTT at first, checkpointing, loss warm-up |
| Synthetic data doesn't transfer to ARC-2 public eval | Medium | High | Continuous eval every 10K steps; kill runs that stall |
| Ghost field adds capacity but doesn't help | Medium | Low | Week 4 ablation; delete the channels if flat |
| Cycle consistency hurts on non-bijective tasks | Medium | Medium | Gate the loss by a learned or heuristic bijectivity signal |
| Resize via "outside" token underperforms on dramatic size changes | Medium | High | Width-predictor head as a safety net |
| MDL regularizer collapses ghost field to zero | Medium | Medium | Soft L1 with warm-up; per-layer instead of global |
| Kaggle runtime blows past 12h | Medium | Fatal | Profile Week 1; if blown, cut TTT steps or candidate count |
| TRM bar is unreachable (8% stays out of reach) | Low-medium | Fatal | Pivot: transfer Volcan insights into a Kaggle top-5 transformer pipeline as a fallback paper |
| CEO and engineer disagree on a load-bearing decision | Guaranteed, repeatedly | Medium | This doc's "*Resolved:*" convention; write it down and move on |

## 17. Phased build plan

### Week 1 — Scaffolding (sanity check the stack)

- Repo: `volcan/` Python package under `arc-agi-2-attack/`.
- Clone and study: [etimush/ARC_NCA](https://github.com/etimush/ARC_NCA), [maxencefaldor/cax](https://github.com/maxencefaldor/cax), [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
- Implement: ARC task loader (JSON → grid tensors), visualization utilities, ARC-2 public eval loader.
- Implement: basic NCA cell update without tensegrity, without ghost field, without MDL — the simplest possible Mordvintsev-style NCA.
- Smoke test: train the basic NCA on a single ARC-1 task, verify it can overfit.

**Deliverable:** a single training curve on one task proving the stack works.

### Week 2 — Core architecture (Volcan, minus synthetic data)

- Implement: Laplacian spectral basis precompute and cache.
- Implement: global tension channel (§9).
- Implement: ghost/bioelectric latent channels (§7).
- Implement: the four-term loss (§12).
- Train on the ARC public training set + public eval (~1400 tasks) with heavy augmentation.

**Deliverable:** Volcan architecture training end-to-end on real ARC data, with per-pillar ablations ready to run.

### Week 3 — Synthetic data pipeline

- Build the concept library (§10, Step 1).
- Build the four category-specific generators (§10, Step 2).
- Generate an initial 5 K task smoke-test corpus.
- LLM sanity check + execution verification passes.
- Train a Volcan run on the 5 K corpus; measure public eval.

**Deliverable:** synthetic data generator + 5 K corpus + first synthetic-data training run.

### Week 4 — TTT infrastructure and first honest number

- Implement: LoRA adapter on the update rule.
- Implement: per-task TTT loop.
- Implement: candidate generation under augmented views.
- Implement: stability + cycle-consistency ranking.
- **Measure: full pipeline on ARC-2 public eval. Honest number, no cherry-picking.**

**Deliverable:** a real ARC-2 public eval score. This is the moment where we decide whether Volcan is working or whether we need to pivot.

### Month 2 — Scale and ablate

- Scale synthetic data to 100 K tasks.
- Run each of the four pillar ablations.
- Iterate on architecture based on what ablations show.
- Begin paper draft in parallel.

### Month 3 — Push the frontier

- Target: cross TRM's 8% on ARC-2 public eval.
- If we cross it, start pushing toward NVARC's 24%.
- If we don't, diagnose which pillar is failing and decide whether to pivot.

### Month 4 — Kaggle submission

- Port to Kaggle notebook format.
- Debug runtime issues on actual T4/P100 hardware.
- Profile and optimize until we fit in 12 h.
- Submit. Measure private-set score.
- Iterate if time permits.

## 18. Success metrics

- **Green light to keep going (Week 4):** ≥ 5% on ARC-2 public eval.
- **Paper-award credible (Month 3):** ≥ 8% on ARC-2 public eval — matches TRM at smaller param count.
- **Kaggle top-5 credible (Month 4):** ≥ 15% on ARC-2 public eval — closes to within 10 pp of NVARC.
- **Grand Prize:** 85% on ARC-2 private set. Honest odds: very low. We aim for it anyway.

## 19. Glossary (CEO reference)

- **NCA (Neural Cellular Automaton):** a small neural network that defines the update rule for every cell in a grid. Same rule applied everywhere, iterated many times.
- **TTT (Test-Time Training):** fine-tune the model on a specific task's demo examples at inference time, not just during offline training.
- **LoRA (Low-Rank Adaptation):** cheap fine-tuning — train a small "delta" on top of frozen base weights instead of updating all of them.
- **Laplacian eigenvectors:** mathematical objects describing the "vibration modes" of a graph. Low-frequency modes = global shape; high-frequency modes = local details.
- **Cycle consistency:** forward(A) = B AND reverse(B) = A. If both hold, the transformation is self-consistent.
- **MDL (Minimum Description Length):** Occam's razor formalized — prefer explanations that can be described with fewer bits.
- **Cross-entropy loss:** standard classification loss. Measures how far the predicted probability distribution is from the true label.
- **Deep supervision:** compute loss at multiple intermediate points during a long forward pass, not just at the end. Stabilizes long-horizon training.
- **BPTT (Backpropagation Through Time):** computing gradients by unrolling a recurrent computation across all its time steps. Memory-expensive; we use checkpointing.
- **Ablation:** remove one component and re-measure, to quantify that component's contribution.
- **Mordvintsev-style residual update:** `s_{t+1} = s_t + f(context)` — the network learns a delta, not an absolute new state. More stable than replacing state directly.
- **Bijective:** a mapping that has a unique inverse. "Rotate 90°" is bijective. "Count the shapes" is not.

## 20. Revision log

- **v0.1 — 2026-04-11** — Initial draft by engineer, awaiting CEO review.

---

*"We aim for the Grand Prize, accept the odds, try our best, and if it becomes a scientific contribution either way, fine."* — CEO, 2026-04-11
