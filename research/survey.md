# ARC-AGI-2 Frontier Survey

*Research date: 2026-04-11*

## TL;DR

As of April 2026, the Kaggle-eligible SOTA on ARC-AGI-2 is **NVARC at 24.03% on the private set for ~$0.20/task** (1st place, ARC Prize 2025), achieved with a Qwen3-4B fine-tuned on ~100K synthetic puzzles plus test-time training plus an improved-ARChitects-style ensemble, with a Tiny Recursive Model (TRM) secondary component ([NVARC repo](https://github.com/1ytic/NVARC), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)). Uncapped commercial systems do better — Gemini 3.1 Pro/GPT-5.4/Claude Opus 4.6 Thinking cluster around **~77–78%** ([Best AI Models April 2026](https://www.buildfastwithai.com/blogs/best-ai-models-april-2026)) — but all are outside the Kaggle efficiency envelope. The $700K Grand Prize (85% on the private set within the compute cap) is **unclaimed**, and the gap is ~61 percentage points at the efficiency-constrained frontier. Chollet's team frames the remaining obstacles as (1) separating knowledge from reasoning, (2) efficiency as an unsolved *scientific* problem, and (3) generalizing the "refinement loop" paradigm beyond domain-specific harnesses ([ARC Prize 2025 Technical Report](https://arxiv.org/html/2601.10904v1)). The weakest point in the current frontier is that *every* high-scoring Kaggle approach is now some variant of test-time-trained transformer over a synthetic-data pipeline — there is no credible induction / program-synthesis entry inside the compute cap, which is a real opening.

---

## 1. Icecuber (Johan S. Wind) — ARC-1 Kaggle 2020 Winner

### What they built
A C++ domain-specific language containing **142 hand-crafted unary grid-to-grid functions** (42 distinct function families, multiple variants each) applied through brute-force compositional search up to depth ~4. Input grids for a task are merged into "pieces," then all function compositions are enumerated into a DAG of derived grids. If no single composed program exactly matches the training outputs, a greedy stacker combines multiple DAG nodes to minimize pixel Hamming distance. Diagonal-flip augmentation of the training pairs improves coverage. Parallel, no dependencies, pure enumerative search ([top-quarks/ARC-solution](https://github.com/top-quarks/ARC-solution), [ironbar SOTA notes](https://ironbar.github.io/arc24/03_State_of_the_art/)).

### Score
**20.6% on the 2020 Kaggle ARC-1 private leaderboard** (1st place, 2020 Kaggle competition); ~129/419 on the public evaluation set at depth 2 ([top-quarks/ARC-solution](https://github.com/top-quarks/ARC-solution), [1st place writeup](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/writeups/icecuber-1st-place-solution-code-and-official-docu)).

### Compute budget
Runs on a single Intel i7-7700HQ (16GB RAM); full test-set prediction takes ~9 hours within Kaggle's 9-hour CPU-only limit that year ([GitHub README](https://github.com/top-quarks/ARC-solution)).

### Strong point
Extreme efficiency: raw, brute-force, cache-friendly C++ enumeration with DAG de-duplication is still the fastest known ARC-1 approach per-task. The insight that many ARC-1 tasks decompose into ≤4 DSL-op compositions (documented empirically) directly *motivated* ARC-AGI-2's design — and it means a well-chosen DSL is still a meaningful prior. Diagonal augmentation gave a free +4 tasks ([ironbar SOTA notes](https://ironbar.github.io/arc24/03_State_of_the_art/)).

### Weak point
Intelligence is baked into the DSL by the human author. Any task whose operator isn't in the DSL is unsolvable regardless of search depth. Icecuber himself framed the result as showing ARC-1 was "more brute-forceable than expected," which is exactly the weakness ARC-AGI-2 targets: the ARC-Prize team explicitly removed tasks solvable by naive program search ([ARC-AGI-2 technical report](https://arxiv.org/html/2505.11831v1)).

### What was tried and abandoned
Deeper search (depth 5+) — computationally intractable within the DAG enumeration schema; the search space explodes. The team also noted that training-only DSL hurt generalization and moved to using the evaluation set as additional "training" signal.

### Takeaway for a new solver
A fast DSL+search kernel is still worth building as an *ensemble floor*. It's free points on any task whose solution is short in any reasonable DSL, and it anchors compute/task at near-zero. Don't rely on it to crack ARC-AGI-2.

---

## 2. Ryan Greenblatt (June 2024) — GPT-4o Massive Sampling

### What they built
A pure API pipeline that has GPT-4o generate ~8,000 Python programs per ARC-1 public-eval task, then selects by correctness on the training pairs. Four prompt variants (V0/V1/V1.1/V2) contribute 1,024–2,048 samples each; ~30K-token few-shot prompts include both images and an enhanced text representation (spreadsheet-style cell addresses, connected-component lists, normalized shape views, input–output diffs). An additional ~3,040-sample *revision phase* targets the 12 most-promising partial solutions. Programs are ranked by geometric mean of Hamming-distance ranks across examples; final submission is a majority vote over programs that match all training pairs ([Redwood Research blog](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt)).

### Score
**50% on the ARC-1 public test set** (prior SOTA was 34%); **72% on a held-out 100-task slice of the train set** (human MTurk baseline 85%). On the ARC-1 *private* leaderboard this type of approach scores 10–20 points lower (estimated ~33%) ([Redwood post](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt), [ironbar SOTA notes](https://ironbar.github.io/arc24/03_State_of_the_art/)).

### Compute cost
Not explicitly dollarized per task, but ~8K–11K GPT-4o API calls per task at roughly 30K prompt tokens each — back-of-envelope ~$15–30/task at mid-2024 pricing, far outside the Kaggle $0.42/task envelope. Development: ~6 engineer-days ([Redwood post](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt)).

### Strong point
**Scaling law:** Greenblatt measured ~3% accuracy per doubling of samples — a clean, predictable log-linear. Enhanced grid *text* representation (spreadsheet addresses + component decomposition + diffs) plus revision phase each gave ~10× sampling efficiency. This is direct evidence that "representation engineering" is load-bearing, not cosmetic.

### Weak point
The same log-linear extrapolates *terribly*: reaching 70% accuracy would require ~2M samples; 85% is effectively unreachable with the method as-is (~$30K+/task). Concrete failure modes he enumerated: (a) GPT-4o "totally fails to extract colors from images >12×12," (b) "simple off-by-one errors extremely often," (c) long-context degradation after ~32K–40K tokens, (d) model refuses to generate programs as long as the few-shot examples suggest ([Redwood post](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt)).

### What was tried and abandoned
Multi-round iterative debugging (cost-effectiveness lost to just drawing more samples); further revision rounds (same); heavy prompt-diversity ensembling (adding V2 samples beat adding new prompt variants).

### Takeaway for a new solver
Two things to steal and one to avoid. *Steal:* the grid text-representation playbook (every downstream system should feed models spreadsheet coordinates + connected-components + diffs, not raw bitmaps) and the two-pass "generate-many / revise-top-K" structure. *Avoid:* believing the log-linear scaling implies you can buy your way to 85%. It doesn't — and ARC-AGI-2 is specifically designed to flatten that curve faster.

---

## 3. MindsAI (2024 & 2025) — Test-Time Training Pioneer

### What they built (2024)
Fine-tuned a **Salesforce T5-series model** on the ARC public training set augmented with RE-ARC–style procedurally-regenerated tasks. At inference, for each private task, the model is fine-tuned *again* on just that task's 2–10 demonstration pairs, heavily augmented (rotations, reflections, color permutations, example reorderings) — a scheme they originally branded "active inference." Solution = decoded completion of the test input under the task-adapted weights ([ARC Prize 2024 Technical Report](https://arxiv.org/html/2412.04604v2), [ironbar TTFT iteration notes](https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/)).

### Score (2024)
**55.5% on the ARC-AGI-1 private evaluation set** — the highest 2024 Kaggle score, but they declined to open-source and so were ineligible for the Kaggle prize; they still set the public leaderboard record for the year ([ARC Prize 2024 report](https://arxiv.org/html/2412.04604v2)).

### Score (2025)
**12.64% on the ARC-AGI-2 private set, 3rd place** (5-figure prize). 2025 entry described as "test-time-training pipeline combining test-time fine-tuning, augmentation ensembles, tokenizer dropout, and novel pretraining techniques" ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

### Compute
Fits in Kaggle's 12-hour / 4× small-GPU envelope; under $0.42/task by construction (same cap as everyone else in 2024/2025).

### Strong point
MindsAI *invented* the modern TTT-for-ARC recipe that every top-5 Kaggle entry since 2024 has copied. The key insight: LoRA-adapted per-task fine-tuning converts a few-shot learning problem into a finetuning problem, which transformers are vastly better at. Augmenting a single task into hundreds of "synthetic siblings" via color/rotation/reflection groups is free signal.

### Weak point
On ARC-AGI-2, the 55% → 12% collapse is devastating. TTT's ceiling is *bounded by the diversity of augmentations* — and ARC-AGI-2 tasks are built to be *compositionally* novel in ways that geometric/color augmentation can't simulate. Per the MIT TTT paper, oracle voting over candidates lands near human performance, but the *candidate generator* is the wall ([Akyürek et al., Nov 2024](https://arxiv.org/html/2411.07279v1)).

### What was tried and abandoned
Not public — MindsAI has never released details. The 2025 description ("tokenizer dropout, novel pretraining techniques") is the first time they've hinted at architectural-level innovation rather than just more augmentation, which tells you where they think the ceiling is.

### Takeaway for a new solver
Build TTT into the pipeline — it's table stakes. Don't expect TTT *alone* to get you past ~12–15% on ARC-AGI-2. The augmentation group is a representation bottleneck.

---

## 4. ARChitects — ARC Prize 2024 Kaggle Winners

### What they built (2024)
A single **Mistral-NeMo-Minitron-8B-Base** fine-tuned with LoRA (rank 256 pretraining, rank 32 test-time), running the ARChitects' **product-of-experts** inference pipeline. Custom tokenizer reduced to ~64 task-specific tokens (one per digit/color + structural tokens) to prevent digit-merging ("12" as a single token) that wrecks grid layout. Training data: RE-ARC procedurally-regenerated ARC-train variants, augmented with the full D₈ symmetry group + color permutations + example re-ordering. Candidate generation: *threshold-based depth-first search* through the token tree (prune any partial whose accumulated probability falls below T=9%) — empirically generates only ~9.3 candidates per task while covering 76% of correct solutions. Selection: the same LLM re-scores every candidate under 16 augmented "perspectives" of the task, and the final score is the *geometric mean* across perspectives (product-of-experts over augmentations). Solution-stability under augmentation is the decisive filter ([da-fr.github.io ARChitects paper](https://da-fr.github.io/arc-prize-2024/the_architects.pdf), [arXiv:2505.07859](https://arxiv.org/html/2505.07859v1)).

### Score
- **ARC-AGI-1 private (Kaggle 2024):** 53.5% → 1st place ([ARC Prize 2024 report](https://arxiv.org/html/2412.04604v2))
- **ARC-AGI-1 public eval:** 71.6% (286.5/400) — the open-source SOTA for public eval ([Product-of-Experts paper](https://arxiv.org/html/2505.07859v1))
- **ARC-AGI-2 private (Kaggle 2025):** 16.53% → 2nd place, with a new 2D-aware masked-diffusion variant ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis))

### Compute cost
**~$0.02 per task** on their open-source pipeline — that is two orders of magnitude under the Kaggle cap. 98 H100-hours of initial fine-tuning; ~51 seconds per task of test-time training on a single RTX 4090; ~20:50 hours total inference over the full 400-task public eval ([Product-of-Experts paper](https://arxiv.org/html/2505.07859v1)).

### Strong point
Every individual component is efficient, the whole is *cheap*, and the product-of-experts scoring is a genuinely new idea: requiring a solution to be plausible under *multiple* augmented perspectives of the same task is both principled (Theorem 4.1 in the paper shows KL divergence compositionality) and empirically decisive (product vs. sum: 71.6% vs. 66.6%). DFS beats beam search and multinomial sampling in both quality and wall time ([Product-of-Experts paper](https://arxiv.org/html/2505.07859v1)).

### Weak point
The jump from 53.5% on ARC-1 to 16.5% on ARC-2 is the single most informative data point in this survey. The architectural pipeline is fine — it's the *candidate generator* (the fine-tuned LM) that hits a wall on ARC-2's compositional tasks. The augmentation group the product-of-experts exploits is small: D₈ × color permutations covers visual-symmetry reasoning but has nothing to say about *multi-rule* or *context-dependent* tasks that ARC-2 adds ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)).

### What was tried and abandoned
Stochastic sampling and beam search (both slower and lower-accuracy than threshold-DFS). Sum-aggregation over perspectives (strictly worse than product). Using ARC-Heavy or ConceptARC datasets (introduced conceptual leakage into private eval).

### ARChitects 2025 (masked-diffusion variant)
For 2025 they swapped the autoregressive LM for a **2D-aware masked-diffusion language model with recursive self-refinement and perspective-based scoring**. The masked-diffusion formulation lets the model revise any cell of the output grid over multiple denoising steps, which *is* the "refinement loop" theme Chollet highlights. Result: 16.53% on ARC-2 private for 2nd place — a big gain over the autoregressive approach on ARC-2 specifically ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis), [ARC Prize 2025 Technical Report](https://arxiv.org/html/2601.10904v1)).

### Takeaway for a new solver
The ARChitects pipeline (TTT + DFS + product-of-experts over augmentations) is the single most imitable published recipe, and at $0.02/task it leaves room for other modules in the same compute envelope. The masked-diffusion upgrade is a signal that *iterative refinement of the full output grid* beats autoregressive decoding on ARC-2.

---

## 5. OpenAI o3 (December 2024) — Test-Time Search in CoT Space

### What they built
OpenAI has not published the method. ARC Prize's own characterization after direct engagement with OpenAI: **"natural language program search and execution within token space"** — at inference, o3 searches over the space of chains-of-thought "in a fashion perhaps not too dissimilar to AlphaZero-style Monte Carlo Tree Search," with some learned evaluation or process-reward signal selecting among expansions. Chollet specifically calls it *"deep learning–guided program search,"* where the programs are natural-language reasoning traces rather than executable symbolic code ([ARC Prize: o3 breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough)).

### Score (ARC-AGI-1)
- **High-efficiency / low-compute:** 75.7% on 100-task Semi-Private set, 82.8% on 400-task public eval
- **Low-efficiency / 172× compute:** 87.5% Semi-Private, 91.5% public eval
([ARC Prize: o3 breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough))

### Score (ARC-AGI-2)
At the time of the May 2025 technical report, o3 (Medium) scored **~3.0% on ARC-AGI-2** ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)). Later 2025/2026 frontier models are materially better: GPT-5.2 Pro ~54%, Gemini 3.1 Pro 77.1%, Claude Opus 4.6 Thinking and GPT-5.4 ~78% ([Best AI Models April 2026](https://www.buildfastwithai.com/blogs/best-ai-models-april-2026), [IntuitionLabs benchmark analysis](https://intuitionlabs.ai/articles/gpt-5-2-arc-agi-2-benchmark)).

### Compute cost
- High-efficiency: **$26/task** on Semi-Private, $167/task on public eval
- Low-efficiency (172×): **$4,560/task** on Semi-Private, $1,900/task on public eval; total run cost ~$346K for the 91.5% score ([ARC Prize: o3 breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough))
- Compare: human contractor solves an ARC-1 task for ~$5

### Strong point
o3 demonstrated that a scaled-up, process-rewarded CoT search can crack ARC-1 at all. It forced the field to acknowledge that *test-time search* is a genuinely novel capability over fixed-inference transformers. It did NOT break ARC-2.

### Weak point (and why ARC Prize doesn't call it "solved")
Three reasons explicitly stated by ARC Prize:
1. **Efficiency:** even high-efficiency o3 is ~50× over Kaggle's $0.42/task cap; low-efficiency is ~11,000× over.
2. **Not open / not reproducible:** ARC Prize only recognizes open-source solutions for its headline prize.
3. **Generalization collapse on ARC-2:** ARC Prize's pre-release testing already told them o3 "could score under 30%" on ARC-AGI-2, which was borne out at ~3% ([ARC Prize: o3 breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough), [ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)).

### Takeaway for a new solver
o3-style CoT search is not an option inside Kaggle's compute budget, but the *shape* of it — expand a tree of candidate reasoning/program traces, prune with a learned value, execute to verify — is directly imitable at small scale. The NVARC and ARChitects 2025 pipelines are essentially this pattern compressed into 4B-param models with TTT.

---

## 6. ARC-AGI-2 (2025) — What Makes It Harder, and the 2025 Kaggle Results

### What's new relative to ARC-1
Per the official technical report ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)):
- **Removal of brute-forceable tasks.** ~49% of ARC-1 tasks were solvable by naive DSL enumeration (à la Icecuber); those archetypes are gone.
- **Larger grids, more objects, more concepts per task.** Average human solve time is now 2.7 minutes vs. near-instant for ARC-1.
- **Four specific reasoning capabilities targeted:**
  1. *Multi-rule reasoning* — simultaneous application of several interacting rules
  2. *Multi-step reasoning* — step N depends on outcome of step N-1
  3. *Contextual rule application* — rules modulated by context (conditional logic)
  4. *In-context symbol definition* — interpreting task-specific symbol meanings on-the-fly
- **Human baseline rebuilt from scratch.** 407 participants across 515 sessions; **100% of ARC-AGI-2 tasks were solved by ≥2 humans**; average tester solved 66% of attempted tasks; median solve time 2.2 min ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)).
- **Three partitions recalibrated** (Public / Semi-Private / Private) so mean human accuracy differs by ≤1 percentage point across sets.
- **Efficiency requirement:** the Kaggle prize track enforces ~$0.42/task — explicitly to prevent o3-style brute-compute solutions.

### Frontier model scores on ARC-AGI-2 (as of Apr 2026)
| System | ARC-2 score | Notes |
|---|---|---|
| o3-mini (High) | 3.0% | May 2025 technical report |
| o3 (Medium) | 3.0% | May 2025 |
| ARChitects 2024 (autoregressive) | 2.5% | May 2025 |
| Claude Opus 4.5 (Thinking, 64K) | 37.6% | $2.20/task, late 2025 |
| GPT-5.2 (Thinking) | 52.9% | Late 2025 |
| GPT-5.2 (Pro) | 54.2% | Late 2025 |
| Poetiq (Gemini 3 Pro + refinement) | 54% | $31/task |
| Gemini 3.1 Pro | 77.1% | April 2026 |
| GPT-5.4 | 78.2% | March 2026 |
| Claude Opus 4.6 Thinking | 78.2% | March 2026 |

Sources: [ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1); [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis); [Best AI Models April 2026](https://www.buildfastwithai.com/blogs/best-ai-models-april-2026); [IntuitionLabs GPT-5.2 analysis](https://intuitionlabs.ai/articles/gpt-5-2-arc-agi-2-benchmark). *None of the 77–78% scores are inside the Kaggle compute envelope.*

### 2025 Kaggle ARC Prize results (ARC-AGI-2, private set, Kaggle compute envelope)
- **1st — NVARC (Ivan Sorokin + Jean-François Puget, NVIDIA KGMoN): 24.03% private, 27.64% public, ~$0.20/task, $25K prize** ([NVARC repo](https://github.com/1ytic/NVARC), [NVIDIA dev blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/), [Trelis writeup](https://trelis.substack.com/p/nvarc-2025-arc-prize-winners))
- **2nd — ARChitects: 16.53% private**, 2D-aware masked-diffusion LM, $10K ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis))
- **3rd — MindsAI: 12.64% private**, TTT + tokenizer-dropout + augmentation ensembles, $5K
- 1,455 teams, 15,154 entries, 90 paper submissions
- **Grand Prize: UNCLAIMED.** Gap to the 85% threshold: 60.97 percentage points ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis))

### NVARC technical detail (1st place)
NVARC is an **ensemble of an improved ARChitects-style system plus TRM-based components**, trained almost entirely on synthetic data:

- **Base model:** Qwen3-4B (vision-language), fine-tuned with Unsloth Flash + LoRA. Tokenizer trimmed to ~16 tokens (digits 0–9 for colors + structural tokens); embedding table patched to match.
- **Synthetic data pipeline (offline heavy lifting):** seed dataset = Human-ARC (~1,000 descriptions) + BARC (~600 tasks) = ~2,000 structured summaries broken into 5 components (grid generator, transformation steps, rules, insights, concepts). Used gpt-oss-120B to sample *combinations* of seed concepts from a theoretical 9M-combination space, yielding 260K new puzzles. Each puzzle verified: generate ≥30 valid input grids + ≥20 transformation implementations per puzzle, accept only when ≥8 implementations produce identical outputs. Final dataset: **~103K puzzles, 3.2M augmented variants**.
- **Augmentation:** D₈ (8) × color permutations (up to 10!) — pre-existing datasets get 256 augmentations, synthetic 24–32, TRM training subsampled to ~3,000 for memory.
- **Test-time training:** LoRA fine-tune on the few demo pairs of each private task (ARChitects-style).
- **Inference:** DFS-style candidate generation + perspective-based scoring (inherited from ARChitects 2024).
- **TRM component:** Alexia Jolicoeur-Martineau's Tiny Recursive Model (7M params, recursive refinement), used as a secondary signal in the ensemble.
- **Final Kaggle submission:** Qwen3-4B alone (not the ensemble) due to the 12-hour/4-small-GPU compute wall. The ensemble works in research but doesn't fit in Kaggle.
- Sources: [NVARC repo](https://github.com/1ytic/NVARC), [NVIDIA dev blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/), [Trelis writeup](https://trelis.substack.com/p/nvarc-2025-arc-prize-winners).

### Strong point
NVARC is the clearest proof that ARC-AGI-2 yields to *relentless synthetic-data engineering* under a small-model/TTT skeleton. The offline pipeline does the conceptual work; the runtime model just needs to be a fast adapter.

### Weak point
NVARC is ceiling'd by exactly what it optimizes: synthetic-data coverage over *known* ARC-2 concept archetypes. Tasks that are genuinely compositionally novel — the ones ARC-AGI-2 specifically targets — remain out of reach. Also: the team confirmed they had to drop the ensemble (TRM + ARChitects) for the actual submission because of Kaggle's 12-hour wall, meaning the top score leaves ensemble headroom on the table ([Trelis writeup](https://trelis.substack.com/p/nvarc-2025-arc-prize-winners)).

### Tried and abandoned
Chain-of-thought, tool use, and RL-style agents — per the NVIDIA writeup, "couldn't fit within Kaggle's runtime constraints, necessitating the offline-heavy strategy" ([NVIDIA dev blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/)). Training was still happening the day before the deadline.

---

## 7. Academic Program Synthesis (2024–2025)

### 7a. BARC / "Combining Induction and Transduction" (Li, Ellis, Tavares et al., Nov 2024)
**What:** Trains two neural models on synthetic ARC-like Python programs ("ARC-Heavy," the BARC dataset, generated by LLMs). An *inductive* model outputs programs; a *transductive* model outputs the test grid directly. Both use the same architecture; they solve *disjoint* sets of tasks and ensemble to near-human. Won 1st place in the 2024 ARC Prize Paper Award.
**Key finding:** "Inductive program synthesis excels at precise computations and composing multiple concepts; transduction succeeds on fuzzier perceptual concepts." Ensemble >> either alone, and all subsequent top-3 Kaggle entries have inherited this induction+transduction framing ([arXiv:2411.02272](https://arxiv.org/abs/2411.02272), [ARC Prize 2024 Technical Report](https://arxiv.org/html/2412.04604v2)).
**Score:** ~56.75% on ARC-1 public eval (BARC alone); TTT+BARC reached 62.8%; TTT+BARC+program synthesis 61.9% matching avg human ([MIT TTT paper](https://arxiv.org/html/2411.07279v1)).

### 7b. SOAR (Pourcel, Colas & Oudeyer, Jul 2025) — 2nd-place ARC Prize 2025 Paper Award
**What:** Self-Improving LM for Evolutionary Program Synthesis. An LLM is placed in a self-improving loop: (1) evolutionary search over Python programs, with the LLM mutating/refining candidates, then (2) a *hindsight learning* phase where failed programs are relabeled as correct solutions for the synthetic tasks they *did* solve, and those (task, program) pairs fine-tune the LLM. Repeat. No human-written DSL, no human-generated solutions.
**Score:** **80% on ARC-1 public train, 52% on ARC-1 public test** — SOTA among open-source LLM methods using no hand-crafted data ([arXiv:2507.14172](https://arxiv.org/abs/2507.14172), [Julien Pourcel's site](https://julienp.netlify.app/posts/soar/)).
**Strong point:** Relabeling failed searches as solutions to the tasks they accidentally solve is the cleanest self-improvement signal in the ARC literature and generalizes beyond ARC.
**Weak point:** Still an ARC-1 result; ARC-2 performance has not been reported by the authors. The evolution loop is expensive.

### 7c. HYSYNTH (NeurIPS 2024)
Uses LLM samples to learn a PCFG, then runs bottom-up enumerative synthesis *guided* by the PCFG. Explicit hybrid of neural priors + classical program search. Not a Kaggle contender but a credible academic baseline on ARC sub-tasks ([arXiv:2405.15880](https://arxiv.org/abs/2405.15880)).

### 7d. CodeIt (ICML 2024)
Reformulates ARC as programming-by-examples, introduces *prioritized hindsight replay* for self-improvement. First neuro-symbolic method to scale to the full ARC-1 eval set; solved ~15% — not competitive but influential as a scaling proof ([arXiv:2402.04858](https://arxiv.org/pdf/2402.04858)).

### 7e. Tiny Recursive Model (Jolicoeur-Martineau, Oct 2025) — 1st-place ARC Prize 2025 Paper Award
**What:** A 7M-parameter, 2-layer network that *recursively refines* its own predicted answer. At each of up to 16 refinement steps: (i) update latent z from (x, y, z); (ii) update answer y from (y, z). Beats HRM and is dramatically simpler. Supervised refinement loop — the paper's pitch is that *recursion depth substitutes for parameter count* on reasoning benchmarks.
**Score:** **45% ARC-AGI-1, 8% ARC-AGI-2** — impressive for 7M params; feeds directly into NVARC's ensemble ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871), [SAIL-Montreal GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)).
**Follow-up:** "Test-time Adaptation of Tiny Recursive Models" ([arXiv:2511.02886](https://arxiv.org/abs/2511.02886)) and "Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute" ([arXiv:2512.11847](https://arxiv.org/abs/2512.11847)) are the state of the refinement-loop research thread as of early 2026.

### 7f. CompressARC (Liao & Gu, CMU) — 3rd-place ARC Prize 2025 Paper Award
**What:** Randomly initialized **76K-parameter** network; trained only at inference, only on the single target task (with the test output masked). Explicit Minimum Description Length objective: find the shortest programmatic description (here: smallest NN) that reconstructs the demo pairs. No pretraining, no ARC train set, no synthetic data.
**Score:** 20% ARC-AGI-1 eval (some reports say 20–34% depending on split), ~4% ARC-AGI-2. Each puzzle: ~20 minutes on one RTX 4070 ([arXiv:2512.06104](https://arxiv.org/abs/2512.06104), [ARC-AGI Without Pretraining writeup](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)).
**Why it matters:** It's the *purest* compressive-inductive-bias approach on the leaderboard and proves the MDL framing is viable at small scale. Too slow for Kaggle (20 min × 120 tasks = 40 hours), but the *principle* — "the task is your only training set, and compression is your loss" — is a fertile direction.

---

## 8. Test-Time Training Literature (2024–2025)

### "The Surprising Effectiveness of Test-Time Training" (Akyürek et al., MIT, Nov 2024)
**Paper:** [arXiv:2411.07279](https://arxiv.org/html/2411.07279v1). Formalized what MindsAI had been doing and showed it transfers to other few-shot benchmarks (BBH).

**Recipe (critical ingredients, each ablation-confirmed):**
1. Initial fine-tune on *synthetic* tasks from ReARC generators (~600K tasks) + ~6,400 LLM-generated task generators. LR 2.5e-5, 2 epochs, batch 32, AdamW.
2. **Per-instance LoRA adapters**, rank 128, applied to MLP + attention + output layers. Each task gets its own adapter.
3. **Leave-one-out task construction** — for each demo pair (in, out), build an ICL example with *the other* demo pairs as context; apply geometric + color augmentations; heavy data expansion.
4. Loss = language-modeling loss over demos AND test outputs.
5. **Hierarchical self-consistency voting** under invertible transformations: intra-transform top-3, then global top-2, with row/column majority vote for sparse groups.

**Scores (ARC-AGI-1):**
- Llama-3-8B base fine-tune: 39.25%
- + TTT: 47.125%
- + TTT + BARC neural: 53%
- + TTT + BARC + program synthesis ensemble: **61.875%** (matches average human on the reported 80-task dev set)

**Ablation (on the 80-task dev set, pass@2):** Baseline 5% → full TTT 29% (6×). Remove leave-one-out transformations → 13%. Task-shared LoRA instead of per-instance → 22%.

**Compute:** ~12 hours on a single A100 for 100 tasks → exceeded Kaggle's P100/12-hour cap at the time of writing, so no private-eval submission from MIT directly.

**Abandoned:** LLM-generated synthetic tasks (from GPT-4/GPT-4o) *hurt* fine-tuning by ~5% when included — their filtering isn't good enough.

**Ceiling:** Oracle voting over generated candidates hits ~97.8%, meaning the voting aggregator is fine but *candidate diversity* is the wall. The paper's concluding point is the same as every other TTT paper: **the problem is generating a sufficiently diverse candidate pool, not selecting from it.**

### Follow-ups
- NVARC / ARChitects 2024–2025 are essentially TTT-implementations optimized for the Kaggle envelope.
- [arXiv:2511.02886](https://arxiv.org/abs/2511.02886) "Test-Time Adaptation of Tiny Recursive Models" extends TTT to the 7M-param TRM regime.

### Known TTT ceiling on ARC-2
No approach using TTT-alone has exceeded ~16% on ARC-AGI-2 (ARChitects 2025). Multiple teams have hit this wall: MindsAI at 12.64%, ARChitects at 16.53%, NVARC's non-synthetic baselines similarly. Every point above ~17% on Kaggle ARC-2 has come from combining TTT with aggressive synthetic-data generation ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

---

## 9. Neurosymbolic Hybrids (2024–2025)

The dominant 2024–2025 pattern isn't a clean "neural plus symbolic" separation so much as *neural candidate generation over a symbolic verification substrate*:
- **BARC** (7a): neural induction ↔ neural transduction, ensemble.
- **HYSYNTH** (7c): LLM PCFG prior ↔ enumerative symbolic synthesizer.
- **SOAR** (7b): LLM mutator ↔ Python executor ↔ hindsight self-supervision.
- **NVARC / ARChitects**: LLM candidate generator + TTT + symbolic verification on demos + product-of-experts scoring.
- **TRM**: recursive "refinement" network acting as the learned search operator.

The ARC Prize 2025 Technical Report's single most-repeated theme is the **refinement loop**: a per-task iterative program-optimization loop guided by a feedback signal. This cuts across evolutionary program synthesis (Berman's evo-test-time compute, Pang's evo program synthesis), deep-learning weight refinement (TTT), and commercial application-layer refinements (Poetiq's Gemini 3 Pro refinement loop: 31% → 54% at a 38× cost increase). Chollet et al. explicitly call refinement loops *the defining theme* of 2025 ([ARC Prize 2025 Technical Report](https://arxiv.org/html/2601.10904v1), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

---

## 10. ARC Prize Foundation's Own Analysis

### Direct statements of remaining obstacles from the ARC Prize 2025 blog / report
- *"We still need new ideas, such as methods to separate knowledge and reasoning, among other challenges."*
- *"For the ARC-AGI-1/2 format, we believe the Grand Prize accuracy gap is now primarily bottlenecked by engineering while the efficiency gap remains bottlenecked by science and ideas."*
- *"Current AI reasoning performance is tied to model knowledge"* — i.e., frontier-model improvements are partly just knowledge contamination into pretraining.
- *"Machines capable of highly efficient adaptation ... remain firmly within the realm of science fiction."*

### What they say is underexplored
- **General-purpose refinement harnesses.** All current refinement loops are domain-specific — there's no "DSPy / GEPA for ARC" at the application layer.
- **Knowledge/reasoning separation.** No one has a credible recipe.
- **Refinement loops applied beyond program synthesis** — e.g., directly to weight updates at the ensemble level.

Sources: [ARC Prize 2025 Technical Report](https://arxiv.org/html/2601.10904v1), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis).

### Historical ARC Prize Foundation framing
ARC-AGI-2 was also motivated by *benchmark contamination*: the same 100 ARC-1 private-eval tasks had been used across 4 competitions, and thousands of leaderboard submissions had effectively leaked them into public distribution. ARC-AGI-2 rebuilt three partitions (public, semi-private, private) calibrated so mean human accuracy differs by ≤1 pp across sets ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)).

---

## Cross-Cutting Analysis

### 1. Current public SOTA on ARC-AGI-2
**Inside the Kaggle compute envelope ($0.42/task, P100/T4-class GPUs, 12 h, no internet):** **NVARC at 24.03% on the private set for $0.20/task**, April 2025 ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)). Uncapped: frontier commercial LMs (Gemini 3.1 Pro 77.1%, GPT-5.4 / Opus 4.6 Thinking ~78.2%), April 2026 ([Best AI Models April 2026](https://www.buildfastwithai.com/blogs/best-ai-models-april-2026)).

### 2. Has the $700K Grand Prize been won?
**No.** The threshold is 85% on the ARC-AGI-2 private set inside the Kaggle envelope. The gap from SOTA is 85 − 24.03 = **~61 percentage points**. The competition continues in 2026 ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

### 3. Kaggle compute budget for the prize track
- **GPUs:** Up to 4× P100 / T4-class GPUs (the P100 was the stated single-GPU during 2023; 2024 and 2025 allowed up to 4 small GPUs in parallel)
- **Wall time:** 12 hours per submission (up from 5 hours in earlier years)
- **No internet access** (no GPT-4/Claude/Gemini API calls)
- **Dollar budget proxy:** ~$50 for 120 evaluation tasks → ~$0.42/task on average
- Sources: [Kaggle general forum on GPU limits](https://www.kaggle.com/general/286404), [ironbar SOTA notes](https://ironbar.github.io/arc24/03_State_of_the_art/), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis).

### 4. The 2–3 biggest unsolved obstacles on ARC-2, per people actually working on it
1. **Separating knowledge from reasoning.** Every top approach leans on memorized concept archetypes (RE-ARC, BARC, synthetic puzzles generated by gpt-oss-120B). Concepts outside the training distribution are unsolved. Chollet explicitly calls this out ([ARC Prize 2025 Technical Report](https://arxiv.org/html/2601.10904v1)).
2. **Candidate-generator diversity for compositional tasks.** Multiple teams have shown that voting/scoring/selection works fine when a correct candidate exists in the pool — the wall is that for ARC-2's multi-rule + contextual tasks, the candidate generator (be it a fine-tuned LM or an evolutionary search) doesn't *produce* a correct candidate in the first place. Oracle voting on MIT TTT hits ~98%, but real TTT hits ~47% ([MIT TTT paper](https://arxiv.org/html/2411.07279v1)).
3. **Efficiency as a scientific bottleneck.** The ARC Prize team explicitly splits the gap into *engineering* (the accuracy gap) vs. *science* (the efficiency gap): they believe the efficiency gap is unsolved at the conceptual level — there is no known training objective that produces highly sample-efficient in-context reasoning.

### 5. Consensus weakness across strong teams
**All top Kaggle 2025 approaches hit a wall in the mid-teens on ARC-2 *without* synthetic data scaling, and in the mid-twenties *with* it.** The gap between "TTT + basic augmentation" (MindsAI: 12.64%) and "TTT + 100K synthetic puzzles" (NVARC: 24%) is only ~11 points; adding yet more synthetic data appears to yield diminishing returns. The common bottleneck is **compositional novelty** — tasks that require *combining* concepts in a way no single synthetic training example taught the model. Chollet's "multi-rule / multi-step / contextual / in-context symbol definition" categories are precisely where every TTT model fails ([ARC-AGI-2 report](https://arxiv.org/html/2505.11831v1)).

### 6. Underexplored directions flagged but unpursued
1. **General-purpose refinement harnesses** (Chollet/ARC Prize explicitly flag this). No one has built a DSPy/GEPA-class prompt-optimization loop that targets ARC specifically — all refinement loops in 2025 are hand-coded per-team ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
2. **Induction (program synthesis) inside the Kaggle envelope.** BARC and SOAR show induction works on ARC-1 offline, but no Kaggle-eligible top-5 entry in 2025 is a program-synthesis approach — everyone is doing TTT on transformers. An in-budget DFS-over-programs kernel plus a learned value net would be genuinely novel competitively.
3. **MDL / compression as loss.** CompressARC proves the framing is viable at 76K params; nobody has scaled it up, added pretraining, or built an ensemble around it.
4. **Knowledge/reasoning separation.** Chollet explicitly says "we still need new ideas" here. The SOAR-style "hindsight relabeling" self-improvement loop is the closest thing to a concrete proposal.
5. **Refinement applied to *weights* rather than programs.** TTT is refinement-on-weights but only *once*; nobody has done iterative weight-refinement with a learned value signal.
6. **Masked-diffusion LMs for grid outputs.** ARChitects 2025 showed this beats autoregressive decoding on ARC-2 specifically. This is the single most underexplored architectural choice.

---

## Candidate Attack Vectors

Honest assessment of where a new Kaggle-compliant team could plausibly push the frontier. Each includes a one-line argument.

1. **Masked-diffusion over grids + TTT + product-of-experts.** The ARChitects 2025 architecture change (autoregressive → masked-diffusion) was the largest per-team ARC-2 improvement in the competition; nobody else has copied it yet, and there is probably a lot of low-hanging structural improvement available (e.g., 2D-aware noise schedules, multi-scale denoising).

2. **Synthetic-data pipeline focused on ARC-2's four cognitive categories.** NVARC proves that more (verified) synthetic data is the #1 driver of ARC-2 scores. Nobody has yet targeted synthetic generation *directly at* multi-rule / multi-step / contextual / in-context-symbol tasks — current pipelines mostly re-sample ARC-1-style concept seeds. Building a generator that explicitly composes 2–3 interacting rules into each synthetic task is unexplored, and is probably worth >5 percentage points.

3. **Kaggle-budget SOAR variant.** SOAR's self-improving evolutionary loop with *hindsight relabeling* is the cleanest self-supervision signal for program synthesis and has not been adapted to the Kaggle compute envelope. A small (≤7B) LLM + Python executor + hindsight-relabel + 12-hour wall could plausibly beat MindsAI on the induction side while TTT carries the transduction side.

4. **Induction/transduction ensemble that NVARC leaves on the table.** NVARC dropped the ensemble under the 12-hour wall; the ensemble clearly works. A more compute-efficient inducer (e.g., TRM-scale program synthesizer, or a small code-generation model with DFS) would let the ensemble fit in-budget.

5. **Refinement loop targeting the verification oracle, not just the candidates.** All current refinement loops refine the *candidate* given fixed demos. A refinement loop that *augments the demos* through synthetic sibling tasks, then re-checks candidate stability, is essentially the ARChitects product-of-experts idea extended into a loop — no one has done this explicitly.

6. **Knowledge/reasoning separation via caching + offloading.** Pre-compute a large library of verified concept primitives (DSL-like), then at inference let a small LM compose them. This is closer to Icecuber-2020 in spirit than 2025 approaches, but with an LM-based compositional planner. The fact that no team has shipped this in the last 2 years is mildly suspicious — it might be underrated.

7. **TRM at scale with induction signal.** TRM's 7M-param recursive refinement hits 8% on ARC-2 already. Adding a program-execution signal into the refinement loop (each recursive step can be verified against demos with an executor) is a natural next step that the TRM authors themselves are pursuing ([arXiv:2511.02886](https://arxiv.org/abs/2511.02886), [arXiv:2512.11847](https://arxiv.org/abs/2512.11847)).

8. **CompressARC-style MDL, scaled 10×.** Isaac Liao's 76K-param compression approach gets 20% on ARC-1 and 4% on ARC-2 with *no* pretraining. Scaled to ~10M params + limited pretraining on synthetic data, the MDL framing might break through on ARC-2 because *compression is exactly the objective* that rewards compositional generalization.

9. **Explicit meta-search over pipeline components.** Build a framework where each task is routed through {Icecuber DSL, TTT-LM, program synthesis, TRM, MDL} and a learned router picks which component's output to submit. Today everyone submits their single best pipeline. Routing is unexplored and leaves free points on the table.

The strongest bet combines #1, #2, and a reduced #4 — a masked-diffusion candidate generator trained on ARC-2-category-targeted synthetic data, with a lightweight program-synthesis inducer (#4) as a secondary ensemble branch, all inside the 12-hour wall.

---

## Sources

### ARC Prize Foundation (primary)
- [ARC Prize 2025: Technical Report (arXiv 2601.10904)](https://arxiv.org/html/2601.10904v1)
- [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Prize 2024 Winners & Technical Report](https://arcprize.org/blog/arc-prize-2024-winners-technical-report)
- [ARC Prize 2024: Technical Report (arXiv 2412.04604)](https://arxiv.org/html/2412.04604v2)
- [Announcing ARC-AGI-2 and ARC Prize 2025](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025)
- [ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems (arXiv 2505.11831)](https://arxiv.org/html/2505.11831v1)
- [OpenAI o3 Breakthrough High Score on ARC-AGI-Pub](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- [2025 Competition Details](https://arcprize.org/competitions/2025)
- [ARC Prize Guide](https://arcprize.org/guide/1)

### Approach-specific primary sources
- [top-quarks/ARC-solution (Icecuber 2020 code)](https://github.com/top-quarks/ARC-solution)
- [victorvikram/ARC-icecuber (mirror)](https://github.com/victorvikram/ARC-icecuber)
- [Icecuber 2020 Kaggle writeup](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/writeups/icecuber-1st-place-solution-code-and-official-docu)
- [Redwood Research: Getting 50% (SoTA) on ARC-AGI with GPT-4o](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt)
- [The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective (2024 ARChitects paper PDF)](https://da-fr.github.io/arc-prize-2024/the_architects.pdf)
- [Product of Experts with LLMs: Boosting Performance on ARC (arXiv 2505.07859)](https://arxiv.org/html/2505.07859v1)
- [NVARC repository](https://github.com/1ytic/NVARC)
- [NVIDIA dev blog: NVIDIA Kaggle Grandmasters Win AGI Competition](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/)
- [Trelis: NVARC 2025 ARC Prize Winners writeup](https://trelis.substack.com/p/nvarc-2025-arc-prize-winners)
- [Combining Induction and Transduction for Abstract Reasoning (BARC, arXiv 2411.02272)](https://arxiv.org/abs/2411.02272)
- [The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (MIT, arXiv 2411.07279)](https://arxiv.org/html/2411.07279v1)
- [ekinakyurek/marc (MIT TTT code)](https://github.com/ekinakyurek/marc)
- [Less is More: Recursive Reasoning with Tiny Networks (TRM, arXiv 2510.04871)](https://arxiv.org/abs/2510.04871)
- [Test-time Adaptation of Tiny Recursive Models (arXiv 2511.02886)](https://arxiv.org/abs/2511.02886)
- [Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute (arXiv 2512.11847)](https://arxiv.org/abs/2512.11847)
- [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [ARC-AGI Without Pretraining (CompressARC, arXiv 2512.06104)](https://arxiv.org/abs/2512.06104)
- [CompressARC blog post by Isaac Liao](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)
- [iliao2345/CompressARC code](https://github.com/iliao2345/CompressARC)
- [SOAR: Self-Improving Language Models for Evolutionary Program Synthesis (arXiv 2507.14172)](https://arxiv.org/abs/2507.14172)
- [Julien Pourcel SOAR blog post](https://julienp.netlify.app/posts/soar/)
- [HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis (arXiv 2405.15880)](https://arxiv.org/abs/2405.15880)
- [CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay (arXiv 2402.04858)](https://arxiv.org/pdf/2402.04858)

### Secondary / analysis
- [ironbar arc24: State of the art notes](https://ironbar.github.io/arc24/03_State_of_the_art/)
- [ironbar arc24: Solution Summary](https://ironbar.github.io/arc24/05_Solution_Summary/)
- [ironbar arc24: Test-time fine-tuning iteration](https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/)
- [ARC-AGI 2025: A research review — lewish.io](https://lewish.io/posts/arc-agi-2025-research-review)
- [Epoch AI ARC-AGI-2 page](https://epoch.ai/benchmarks/arc-agi-2/)
- [IntuitionLabs: GPT-5.2 & ARC-AGI-2 analysis](https://intuitionlabs.ai/articles/gpt-5-2-arc-agi-2-benchmark)
- [Best AI Models April 2026](https://www.buildfastwithai.com/blogs/best-ai-models-april-2026)
- [Kaggle general: GPU runtime limits](https://www.kaggle.com/general/286404)
- [Why all ARC-AGI solvers fail today — Mithil Vakde](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)

### Kaggle competition pages
- [ARC Prize 2025 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2025)
- [ARC Prize 2025 Leaderboard](https://www.kaggle.com/competitions/arc-prize-2025/leaderboard)
- [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/)
- [ARC Prize 2024 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2024)
