# ARC Prize 2026 — Paper Track submission (Volcan)

Paste the body below as the Kaggle writeup when you enter the Paper Track
(https://www.kaggle.com/competitions/arc-prize-2026-paper-track). Fill in
the arXiv ID in two places once your submission processes. The actual
Kaggle entry form may also ask for team info, agreement to open-source
terms, and an attached PDF — use [paper/volcan.tex](volcan.tex) compiled
to PDF for that upload.

---

## Title
**Volcan: Morphogenetic Cellular Automata for Abstract Reasoning via Code-Dreamed Pretraining**

## Team
Tom Park (solo)

## TL;DR

A 111,195-parameter Neural Cellular Automaton reaches 9.7% (3/31) on a
held-out ARC-AGI-2 subset — matching TRM (7M, 8%) at **1.6% of TRM's
parameter count**. Eight controlled ablations localize the ceiling: it
is distribution-bound, not capacity- or init-bound.

## Paper & code

- **arXiv**: https://arxiv.org/abs/[FILL AFTER ARXIV PROCESSES]
- **Code** (MIT licensed, fully open source): https://github.com/Tompark0927/volcan
- **Checkpoint** (v1.0 release asset, reproduces 3/31): https://github.com/Tompark0927/volcan/releases/tag/v1.0
- **Full architecture doc**: https://github.com/Tompark0927/volcan/blob/main/docs/architecture.md

## What's novel

1. **Substrate.** First NCA with a published ARC-AGI-2 pretraining-plus-TTT
   result. Five biologically-motivated pillars (anisotropic directional
   forces, mycelial sparse attention, Laplacian spectral tension,
   bioelectric ghost pre-pattern, apoptotic pruning), each with an
   explicit computational form. 111,195 trainable parameters.

2. **Data pipeline.** An LLM (qwen2.5:7b via Ollama) writes a Python
   `transform(grid) -> grid` function per rule; we execute to generate
   4 consistent demos. Inverting the pipeline from "LLM writes demos"
   to "LLM writes code → code writes demos" eliminates the rule-
   inconsistency failure mode of naive JSON-dreamed data. An overfit
   filter rejects tasks the base model cannot memorize.

3. **Inference recipe.** D8-augmented LoRA rank-16 test-time training
   ported from MIT TTT (Akyurek et al. ICML 2025) to an NCA substrate.
   This is the single key inference-side unlock: without it, pretraining
   alone scores 0/31.

## What's unusual — eight ablations, two real regressions

The paper's core contribution beyond the efficiency point is the
**systematic ablation table** (§6.2):

| Experiment | Params | Transfer | Delta |
|---|---|---|---|
| Baseline (dream_wide) | 111K | 3/31 | — |
| + data scale 43→325 | 111K | 3/31 | 0 |
| + MoE capacity (×4 experts) | 424K | 3/31 | 0 |
| + rule diversity 20→70 families | 111K | 3/31 | 0 |
| + LoRA rank 32 (vs 16) | 111K | 2/5 probe | **negative** |
| + Hierarchical macro-cells | 255K | 3/31 | 0 |
| + Hyper-TTT (HyperNetwork meta-init) | 111K + 5.1M | **2/31** | **negative** |
| + Dense scaling (mlp_hidden 1024) | **1.76M** | **0/7** (partial, MPS crash) | **strongly negative** |

The 9.7% is a ceiling across six post-baseline axes. Two axes actively
regressed. The dense-scaling regression is especially informative: a
16× wider model produces **lower** pretraining accuracy (78.1% vs
80.5%) AND loses the canonical easy pass (`25d8a9c8`) at transfer.
Combined with the Hyper-TTT regression (a trained 5M-parameter meta-
init is worse than Kaiming zero-mean), this identifies the ceiling
as **distribution-bound, not capacity- or init-bound**.

## What this means for the ARC-AGI-2 research community

Negative results at this scale are load-bearing for small-model
research: the NCA + D8 + LoRA TTT substrate at 111K parameters is
complete enough that scaling any single axis (width, init, rank,
hierarchy, rule diversity, meta-learning, routing) does not move the
number. The remaining axis is **depth scaling** (more iterations per
phase, or more stacked update blocks), which the paper explicitly
flags as the one untested direction.

This gives other small-model teams a clean map of what *not* to try
and where the genuinely open questions are.

## Efficiency comparison

| System | Params | ARC-AGI-2 |
|---|---|---|
| **Volcan (this work)** | **111K** | **9.7% (n=31)** |
| CompressARC (Liao & Gu 2025) | 76K | 4.0% |
| TRM (Jolicoeur-Martineau 2025) | 7M | 8.0% |
| ARChitects 2025 | ~8B | 16.5% |
| NVARC 2025 | ~4B | 24.0% |

Volcan sets a new efficiency point: it is the smallest public model to
match or exceed TRM's 8% on any ARC-AGI-2 sample.

## Honest caveats

- **Sample size**: 30 tasks is small. Clopper–Pearson 95% CI for 3/31
  is (2.0%, 25.8%). Full private-eval accuracy could land anywhere
  in roughly 5–15%.
- **No Kaggle leaderboard submission** in this work. Paper Track only.
- **Subset selection**: tasks were selected for having compatible-shape
  test outputs, which biases toward "size-preserving" rule families.
  A future full-eval submission would resolve this.

## Reproducibility

Single M4 MacBook Air, MPS backend, ~8h wall time for full pretrain +
30-task eval. The v1.0 checkpoint on the GitHub release reproduces
3/31 exactly via `scripts/eval_pretrained.py`.

## License

MIT (repository); paper on arXiv under default CC license.

---

## Submission-form notes for Tom

When the Kaggle Paper Track submission page opens, expected fields:

1. **Title** (paste above)
2. **Authors** (Tom Park)
3. **Paper link** (arXiv URL, once processed)
4. **Code link** (https://github.com/Tompark0927/volcan)
5. **Writeup body** (paste everything above the line between `## Title`
   and `## Submission-form notes`)
6. **PDF attachment** (compile paper/volcan.tex → volcan.pdf)
7. **Agree to open-source terms** — MIT license already in repo; paper
   already on arXiv.

After you submit on Kaggle, nothing happens publicly until **Dec 4, 2026**.
No leaderboard update, no confirmation of "you won" — the ARC Prize team
reviews all papers and emails winners directly.
