# NCA-on-ARC Prior Art Check

*Date: 2026-04-11*

## TL;DR

Yes, Neural Cellular Automata have been seriously tried on ARC-AGI — but only very recently (two preprints in mid-2025) and only on ARC-AGI-1. The state of the art is **~13% single-model / ~18% ensemble on the ARC-AGI-1 public eval**, achieved by Guichard et al.'s ARC-NCA ([arXiv:2505.08778](https://arxiv.org/abs/2505.08778)), which is already a runner-up in the ARC Prize 2025 Paper Awards ([ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)). Both existing serious attempts use true local, parameter-shared, iterated NCA update rules and both already train a fresh NCA per task (a strong form of test-time training), but **neither has been scored on ARC-AGI-2, neither uses pretraining + TTT fine-tuning, and neither has solved the variable-grid-resize problem beyond crude 30×30 padding**. There is one open-source codebase (Apache-2.0, Python + notebooks) that a small team can build on directly ([github.com/etimush/ARC_NCA](https://github.com/etimush/ARC_NCA)). This is a narrow, active, under-explored lane — not a dead end, not fully green-field.

## Direct Attempts

### 1. ARC-NCA / EngramNCA — Guichard, Reimers, Kvalsund, Lepperød, Nichele (2025)

- **Paper / venue:** "ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus," [arXiv:2505.08778](https://arxiv.org/abs/2505.08778), published at ALIFE 2025 / MIT Press Proceedings ([direct.mit.edu link](https://direct.mit.edu/isal/proceedings/isal2025/37/5/134057)). Runner-up ($2.5k) in the ARC Prize 2025 Paper Awards ([arcprize.org/blog/arc-prize-2025-results-analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
- **Affiliations:** Østfold University College, University of Oslo, Simula Research Laboratory, Oslo Metropolitan University.
- **What they built:** A true NCA — 50-channel cell state, shared ~64-hidden-unit convolutional update rule, iterated, trained end-to-end with backprop-through-time on MSE against the target grid. Four EngramNCA variants extend the base architecture with: (v1) dual "public interaction" + "private memory" cell states split into a GeneCA (morphogenesis from a seed) and a GenePropCA (propagation of genetic primitives); (v2) learnable sensing filters; (v3) channel-wise local self-attention and split toroidal/non-toroidal world; (v4) 3×3 patch-based training. Architecturally inherits directly from Mordvintsev et al.'s 2020 Growing NCA ([Distill](https://distill.pub/2020/growing-ca/)). ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **Score on ARC:** ARC-AGI-**1** public evaluation set, restricted to 262/400 problems (the ones that don't require grid resize). Strict threshold (log pixel-MSE ≤ −7):
  - Standard NCA: **10.7%**
  - EngramNCA v1: 6.5% / v2: 9.2% / **v3: 12.9% (best single model)** / v4: 10.3%
  - Union of all four models: **17.6%** (24% at loosened threshold)
  - With the 30×30-padding variant that doesn't drop resize tasks: 16% strict / 27% loose
  - **No ARC-AGI-2 numbers reported.** ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **Failure modes identified:** fine-grained local information mismatch; inability to handle multi-step composite reasoning in a single rollout; edge cases absent from the two-or-three training demonstrations; and the whole class of tasks that require input→output grid resize (forced out of the benchmark or handled only crudely with 30×30 padding). ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **TTT?** Yes, in the strongest possible form — **they train a fresh NCA from scratch on each task's 2–3 demos**, 3000 iterations, AdamW, lr 1e-3 → 3.3e-4 at step 2000, pixel-wise MSE. There is no pretraining and no LoRA — the whole NCA's weights *are* the per-task adapter. This is effectively "TTT is the only training." ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **Variable grid sizes:** Two approaches. (a) Drop the 138/400 tasks whose input and output sizes differ from each other and evaluate on the remaining 262. (b) Pad every grid to 30×30 with a reserved "pad" token and let the NCA overwrite padding to effectively change the output size. The padding variant is clearly reported as a workaround, not a principled solution. ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **Compute:** Single RTX 4070 Ti, ~$0.0004 per task, ~$0.10 to evaluate the whole 262-task set. Roughly 1000× cheaper than ChatGPT-4.5 for comparable strict-threshold performance (10.3% for GPT-4.5 at $0.29/task). ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1))
- **Code:** [github.com/etimush/ARC_NCA](https://github.com/etimush/ARC_NCA) — Python + Jupyter, Apache-2.0, contains training scripts for both the drop-resize and 30×30-pad strategies, visualisation/MP4 utilities, and data-analysis notebooks. No pretrained weights shipped (they're per-task), no separate "TTT loop" code (again, the entire training loop is the TTT). Last updated April 2025. Videos gallery at [etimush.github.io/ARC_NCA](https://etimush.github.io/ARC_NCA/).

### 2. Neural Cellular Automata for ARC-AGI — Xu & Miikkulainen (2025)

- **Paper / venue:** "Neural Cellular Automata for ARC-AGI," Kevin Xu & Risto Miikkulainen, UT Austin. [arXiv:2506.15746](https://arxiv.org/abs/2506.15746) / [HTML](https://arxiv.org/html/2506.15746v1). Published as an 8-page ALIFE 2025 paper ([xu.alife25.pdf](https://nn.cs.utexas.edu/downloads/papers/xu.alife25.pdf)).
- **What they built:** A very compact NCA — one-hot encoding of the 10 ARC colors, 20 hidden channels, a single learnable convolutional layer replacing the fixed Sobel filters of Growing NCA, LayerNorm for stability, **~10,000 total parameters**, 10 iteration steps at eval time. They use a direct `s_{t+1} = f_θ(s_t)` update rather than the residual `s_{t+1} = s_t + f_θ(s_t)` of Mordvintsev et al. ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1))
- **Score on ARC:** ARC-AGI-**1** public training set, restricted to 172/400 "feasible" tasks (tasks where input and output dimensions match and no unseen test colors appear). **23 solved perfectly = 13.37%** of the feasible subset. The paper does not convert this to a comparable full-set number (it would be ~5.75% of 400 if you count the excluded tasks as failures). No ARC-AGI-2 result. ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1))
- **Failure modes identified (their own taxonomy):** (1) models solve demos inconsistently or only under specific hyperparameter settings, (2) models fit the training demos but fail to generalise to the held-out example, (3) models can't even memorise the training demos. Category (2) — overfitting-on-2-demos — is reported as the dominant failure mode. ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1))
- **TTT?** Same structure as ARC-NCA: a new NCA is trained from scratch for every task, AdamW, lr linearly decayed 2e-3 → 1e-4 over 800 epochs, "a few minutes per task." No pretraining, no LoRA. ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1))
- **Variable grid sizes:** **Not handled.** The paper explicitly drops tasks whose input and output grids differ in size. They do note that because the update rule is local, a rule learned on small demos can be rolled out on arbitrarily large grids — shown with a spiral example scaled from small training to 100×100 — but this only helps when the *output* shape is predictable, not when the task semantically requires a size change. ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1))
- **Code:** Not released in the paper; no repo linked.

### 3. CAX / 1D-ARC — Faldor et al. (ICLR 2025 Oral)

- **Paper:** "CAX: Cellular Automata Accelerated in JAX," [arXiv:2410.02651](https://arxiv.org/abs/2410.02651), [ICLR 2025 Oral](https://iclr.cc/virtual/2025/oral/31774). Code: [github.com/maxencefaldor/cax](https://github.com/maxencefaldor/cax).
- **What they built:** A JAX library + benchmarks for NCAs, including a **1-D NCA** evaluated on 1D-ARC (a 1-D simplification of ARC from the [LLMs-and-ARC paper, Xu et al. 2023](https://arxiv.org/html/2305.18354v2)), not 2-D ARC-AGI. The 1D-NCA has spatial dim 128, 32 channels, 2 kernels, 256 hidden, 128 rollout steps. ([arXiv:2410.02651v1 HTML](https://arxiv.org/html/2410.02651v1))
- **Score:** 1D-ARC average **60.12% vs GPT-4's 41.56%**. Perfect (100%) on Move-1/2/3, Pattern-Copy and Pattern-Copy-Multicolor; **0%** on Recolor-by-Odd-Even, Recolor-by-Size, Recolor-by-Size-Comparison — any task requiring non-local counting / comparison. No 2-D ARC-AGI result at all.
- **TTT?** No — the NCA is trained normally on the 1D-ARC train split, not per-task.
- **Variable grid sizes?** Not applicable (1-D, fixed length 128).
- **Relevance:** This is the cleanest confirmation that an NCA beats GPT-4 on *local-rule* ARC-style tasks and completely whiffs on *global-reasoning* tasks. The failure pattern is the single most load-bearing data point for forecasting 2-D ARC-AGI-2 behaviour.

## NCA-Adjacent Work

### Tiny Recursive Model (TRM) — Jolicoeur-Martineau 2025 (Samsung SAIL Montreal)

- [arXiv:2510.04871](https://arxiv.org/abs/2510.04871); 1st Place ($50k) ARC Prize 2025 Paper Award ([arcprize.org/blog/arc-prize-2025-results-analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)); repos [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), unofficial [lucidrains/tiny-recursive-model](https://github.com/lucidrains/tiny-recursive-model).
- **What it is:** A 7M-param, 2-layer network that maintains a latent reasoning state `z` and a current answer `y`, and iterates `z ← f(x, y, z); y ← g(y, z)` up to 16 times with deep supervision. For small fixed grids it drops attention and uses attention-free mixing; for 30×30 it keeps self-attention. Scores **45% on ARC-AGI-1 and 8% on ARC-AGI-2** ([marktechpost write-up](https://www.marktechpost.com/2025/10/09/tiny-recursive-model-trm-a-tiny-7m-model-that-surpass-deepseek-r1-gemini-2-5-pro-and-o3-mini-at-reasoning-on-both-arg-agi-1-and-arc-agi-2/)).
- **NCA connection:** No paper I could find explicitly frames TRM as an NCA, and the TRM paper does not cite Mordvintsev. But the *computational substrate* is NCA-adjacent in two important ways: (a) shared-parameter recurrent refinement of a 2-D state, (b) deep supervision across many "steps" of rollout. The crucial difference is that TRM's iteration operates on a *global* latent (with self-attention for 30×30 grids), not a pure local rule — so it is strictly more expressive than an NCA at the cost of losing translation equivariance. Think of TRM as "NCA with a global mixing channel bolted on."
- **Why that matters for a CEO pitch:** The ARC Prize 2025 paper-award ranking puts TRM (global+recursive) as #1 and ARC-NCA (local+recursive) as a runner-up. The delta is almost entirely explained by global information flow, not by the refinement structure.

### Hierarchical Reasoning Model (HRM) — Sapient (2025)

- [arXiv:2506.21734](https://arxiv.org/abs/2506.21734); ARC Prize analysis: [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis).
- **What it is:** A 27M-param recurrent model with a fast "low-level" inner module and a slow "high-level" outer module, iterated in an outer refinement loop. 40.3% reported / 32% verified on ARC-AGI-1 hidden eval.
- **NCA connection:** The ARC Prize team's ablations found that **the "outer loop" refinement is where the performance comes from** — a plain transformer plus that loop gets within ~5 percentage points — and that HRM behaves "functionally similar to test-time training." This is NCA-adjacent in exactly the same way TRM is: iterated refinement of a 2-D state with parameter sharing across iterations. Nobody has written the paper that reframes HRM/TRM as "global NCAs," but the analogy is sitting right there.

### A Path to Universal Neural Cellular Automata — Béna, Faldor, Goodman, Cully (GECCO '25)

- [arXiv:2505.13058](https://arxiv.org/abs/2505.13058). Shows that gradient-trained NCAs can learn matrix multiplication, transposition, and *emulate a feedforward MNIST classifier entirely inside the cell state*. Makes no ARC claim, but is the most serious result on NCA *expressivity* — relevant to the question of whether NCAs can in-principle represent the long-range operations ARC needs.

### Neural Cellular Automata: Applications to Biology and Beyond Classical AI — Hartl, Levin, Pio-Lopez (2025)

- [arXiv:2509.11131](https://arxiv.org/abs/2509.11131). A review framing NCAs as a substrate for "hierarchical reasoning and control" and explicitly citing ARC-AGI-1 as a place where NCAs show "goal-directed dynamics without centralized control." No new ARC numbers, but useful as the field-level position paper.

### Training Language Models via Neural Cellular Automata — Han et al. (2026)

- [arXiv:2603.10055](https://arxiv.org/abs/2603.10055). Proposes **pre-pre-training LLMs on 164M NCA-generated tokens**; reports +6% on downstream LM, 1.6× faster convergence, improvements on GSM8K, HumanEval, BigBench-Lite. Not an ARC attempt, but signals that NCAs are being treated as a legitimate compute substrate for reasoning in 2026.

### Differentiable Logic Cellular Automata — Google Self-Organising Systems (March 2025)

- Follow-up to Growing NCA from Mordvintsev's group, replacing float neural updates with learnable logic gates. Mordvintsev publicly endorsed this direction ([Mordvintsev on X, March 2025](https://x.com/zzznah/status/1897741626262380835)). No ARC results as of this writing — but if someone were going to try DiffLogic-CA on ARC, it would be the most "on-brand Mordvintsev" line of attack and it has not yet been published.

### CompressARC, NVARC, ARChitects

None of these have been framed as NCAs in any source I found. CompressARC's iterative gradient-descent refinement is *spiritually* similar to per-task NCA training, but the substrate is a transformer, not a local rule.

## Technical Obstacles Reported

Synthesising across ARC-NCA, Xu & Miikkulainen, and the CAX 1D-ARC results:

1. **Variable input/output grid sizes.** This is the single most-cited blocker. Both 2-D papers drop 30–35% of public ARC-AGI-1 tasks by fiat because input and output shapes differ. The only workaround shown to work is 30×30 padding plus letting the NCA "erase" padding — which inflates compute, breaks translation equivariance near the pad boundary, and still only reaches ~16–27% on the padded eval.
2. **Lack of long-range / global information flow.** The CAX 1D-ARC results show this cleanly: 100% on Move/Pattern-Copy, 0% on Recolor-by-Size (any task needing a global count or comparison). This is structural: a k-step NCA can at best propagate information O(k) cells, so "count all cells of color X" at scale N takes ≥N steps, and "decide whether X is bigger than Y" takes even more. ARC-AGI-2 has substantially more global-relation tasks than ARC-AGI-1.
3. **Overfitting to 2–3 demos.** Both 2-D papers identify this as the dominant failure category. Training from scratch on 2–3 demos with ~10k–100k parameters is fundamentally few-shot, and without a prior the NCA either memorises or under-fits.
4. **Training instability / hyperparameter sensitivity.** Xu & Miikkulainen explicitly call out that "models solve new examples inconsistently with high variance or only under specific settings." LayerNorm was required in their setup just to get stable training.
5. **No pretrained prior.** Neither serious 2-D attempt has a pretraining phase. Every task starts from random init. This means the NCA has *zero* inductive bias for "what kinds of rules ARC problems use" — a stark contrast to BARC/MindsAI/ARChitects, which all rely heavily on prior exposure to ARC-like distributions.
6. **Discrete colour outputs via MSE loss.** ARC targets are discrete (10 colours) but both papers train against continuous pixel-MSE. This works but is unprincipled and is one of the mechanisms that produces the "almost solved" cluster both papers report.

## Answers to the Critical Questions

1. **Above 5% on any ARC version with an NCA?** Yes. Guichard et al.'s ensemble union reaches **17.6% strict / 24% loose on ARC-AGI-1 public eval (restricted to 262/400)** ([arXiv:2505.08778v1 HTML](https://arxiv.org/html/2505.08778v1)), and EngramNCA v3 alone reaches **12.9%**. Xu & Miikkulainen reach **13.37% on 172 "feasible" ARC-AGI-1 training tasks**, which corresponds to ~5.75% if you count excluded tasks as failures ([arXiv:2506.15746v1 HTML](https://arxiv.org/html/2506.15746v1)). On ARC-AGI-2, **no one has published an NCA result at all** as of 2026-04.
2. **Open-source NCA-for-ARC codebase?** Yes — [github.com/etimush/ARC_NCA](https://github.com/etimush/ARC_NCA), Apache-2.0, Python + Jupyter, last updated April 2025. Covers both the drop-resize and 30×30-padding evaluation modes. Xu & Miikkulainen's codebase does not appear to be released. CAX ([github.com/maxencefaldor/cax](https://github.com/maxencefaldor/cax)) is a production-quality JAX NCA library that's ~2000× faster than the TensorFlow Growing-NCA reference, and would be the natural substrate to port ARC-NCA onto for Kaggle-budget training.
3. **Single biggest technical obstacle on ARC specifically?** A tie between **(a) variable grid sizes** and **(d) lack of inductive bias for global operations**. (a) is the easier one — 30×30 padding + learnable resize tokens is a solved engineering problem. (d) is the hard one: you can't bolt global information flow onto a pure local rule without either (i) many iteration steps (CAX shows this hits a wall on comparison tasks), (ii) a coarse-to-fine pyramid, or (iii) a separate global channel (which is what TRM and HRM effectively add). The CEO's proposal needs to name which of these it is going to do.
4. **NCA combined with TTT or LoRA?** On ARC, yes — in the trivial sense that both 2-D papers train the whole NCA from scratch per task, i.e. "TTT is the only training." But **nobody has tried the pretrain-then-TTT-fine-tune recipe** that has driven every other ARC winner (BARC, MindsAI, the MIT TTT paper). That is the specific green-field opportunity. A closely related NCA+TTT framing exists in the broader NCA literature for non-ARC benchmarks — fine-tuning NCA "hardware" while freezing the update rule ([Training LMs via NCA, arXiv:2603.10055](https://arxiv.org/abs/2603.10055)) — but it has not been brought to ARC.
5. **NCA-TRM connection?** Nobody has written it up explicitly. TRM iterates a 2-D state with shared parameters and deep supervision, which is the NCA loop; the only substantive difference is that TRM has a global mixing operator (self-attention for 30×30) whereas a pure NCA has only a local 3×3 conv. The most honest framing is: **TRM ≈ NCA + global channel**, and the ARC Prize 2025 leaderboard is telling us that the global channel is worth roughly 2–3× on ARC-AGI-1 (TRM 45% vs ARC-NCA 17.6%). Any new NCA+TTT pitch for ARC-AGI-2 should treat this as the benchmark to beat and should plan to add *something* global.
6. **Mordvintsev / Chan / original NCA authors on ARC?** No direct commentary. Mordvintsev continues to publish on Differentiable Logic CA and related NCA directions with his Google Self-Organising Systems group and has publicly endorsed more-discrete NCA variants ([X post](https://x.com/zzznah/status/1897741626262380835)), but I found no blog post, tweet, or talk where he addresses ARC specifically. Bert Chan has not weighed in publicly on ARC. The closest thing to an "NCA old guard" endorsement of the ARC direction is Maxence Faldor (CAX author) co-authoring the Universal NCA paper and the broader Levin/Hartl review ([arXiv:2509.11131](https://arxiv.org/abs/2509.11131)) explicitly citing ARC-AGI-1 as a target domain.
7. **2024–2026 NCA-for-reasoning papers in general?** Meaningful list: Growing NCA (2020, [Distill](https://distill.pub/2020/growing-ca/)); CAX 1D-ARC (ICLR 2025 Oral, [arXiv:2410.02651](https://arxiv.org/abs/2410.02651)); ARC-NCA ([arXiv:2505.08778](https://arxiv.org/abs/2505.08778), 2025); Xu & Miikkulainen ([arXiv:2506.15746](https://arxiv.org/abs/2506.15746), 2025); Universal NCA ([arXiv:2505.13058](https://arxiv.org/abs/2505.13058), GECCO 2025); NCA Biology-and-Beyond review ([arXiv:2509.11131](https://arxiv.org/abs/2509.11131), 2025); LMs via NCA ([arXiv:2603.10055](https://arxiv.org/abs/2603.10055), 2026); Differentiable Logic CA (Google, March 2025). The scaling signal: NCAs beat GPT-4 on *local-structure* benchmarks (1D-ARC, morphogenesis, MNIST-inside-NCA), are competitive with GPT-4.5 at 1/1000th the cost on ARC-AGI-1, and have never been tested on ARC-AGI-2. The field's forward frontier as of 2026-04 is global-information flow, discrete/logic cell updates, and pretraining priors — all three of which a Kaggle-budget team could plausibly contribute to.

## Verdict

**Between green-field and previously-tried dead end — closer to green-field.** Specifically:

- **Not a dead end.** Two serious 2025 papers, one of them an ARC Prize runner-up, demonstrate that NCAs *do* solve a nontrivial fraction of ARC-AGI-1 at 1000× lower cost than GPT-4.5. Open-source code exists. A well-reviewed library (CAX) exists. Mainstream researchers (Faldor, Levin, Nichele) are actively pushing this direction.
- **Not fully green-field.** The local-rule-only NCA has a known ceiling on ARC-AGI-1 around 15–25%, and the biggest reason — lack of global information flow — will bite harder on ARC-AGI-2, where tasks are more relation-heavy and where even TRM (with a global channel) only reaches 8%. If the CEO's pitch is "pure vanilla Growing-NCA on ARC-AGI-2," that is the previously-tried dead-end branch.
- **The actual green-field lane** is the intersection nobody has filled: **pretrain an NCA (or NCA+global-channel hybrid) on a large corpus of ARC-style synthetic tasks, then TTT-fine-tune the update rule on each task's demos**, with a principled resize mechanism (learnable boundary tokens, not dumb 30×30 padding) and discrete/softmax output rather than MSE. Every single one of those four ingredients (pretraining, LoRA-style TTT adaptation of the update rule, learned resize, discrete output) has *not* been combined with NCAs on ARC in any published work I could find. The ARC-NCA codebase is a usable starting point; CAX is the fast substrate; TRM/HRM are the baselines to beat.
- **Kaggle-budget feasibility: plausible.** ARC-NCA trains 262 tasks for ~$0.10 on one consumer GPU. A pretrain-then-TTT recipe would add roughly one GPU-week of pretraining on synthetic data, which fits comfortably in a Kaggle-class budget. TRM's 7M parameters at 45% ARC-AGI-1 / 8% ARC-AGI-2 is the benchmark a new NCA+TTT entry has to clear to be interesting.

**Recommendation to the team:** go, but go with a clear answer to "where does the global information flow come from." If the answer is "just iterate the local rule more," prior art says that plateaus in the teens on ARC-AGI-1 and will not work on ARC-AGI-2. If the answer is "we add a coarse-grained global channel / attention / pyramid and pretrain before TTT," that is an unexplored, credible lane.

## Sources

- [arXiv:2505.08778 — ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus (Guichard, Reimers, Kvalsund, Lepperød, Nichele, 2025)](https://arxiv.org/abs/2505.08778)
- [arXiv:2505.08778v1 HTML — full paper with numerical results](https://arxiv.org/html/2505.08778v1)
- [MIT Press / ALIFE 2025 proceedings page for ARC-NCA](https://direct.mit.edu/isal/proceedings/isal2025/37/5/134057)
- [github.com/etimush/ARC_NCA — open-source ARC-NCA code (Apache-2.0)](https://github.com/etimush/ARC_NCA)
- [etimush.github.io/ARC_NCA — project page + videos](https://etimush.github.io/ARC_NCA/)
- [arXiv:2506.15746 — Neural Cellular Automata for ARC-AGI (Xu & Miikkulainen, 2025)](https://arxiv.org/abs/2506.15746)
- [arXiv:2506.15746v1 HTML — full paper](https://arxiv.org/html/2506.15746v1)
- [xu.alife25.pdf — ALIFE 2025 camera-ready mirror at UT Austin NN group](https://nn.cs.utexas.edu/downloads/papers/xu.alife25.pdf)
- [arXiv:2410.02651 — CAX: Cellular Automata Accelerated in JAX (Faldor et al., ICLR 2025 Oral)](https://arxiv.org/abs/2410.02651)
- [arXiv:2410.02651v1 HTML — includes 1D-ARC results](https://arxiv.org/html/2410.02651v1)
- [github.com/maxencefaldor/cax — CAX library](https://github.com/maxencefaldor/cax)
- [ICLR 2025 CAX Oral listing](https://iclr.cc/virtual/2025/oral/31774)
- [Distill 2020 — Growing Neural Cellular Automata (Mordvintsev et al.)](https://distill.pub/2020/growing-ca/)
- [arXiv:2505.13058 — A Path to Universal Neural Cellular Automata (Béna, Faldor, Goodman, Cully, GECCO 2025)](https://arxiv.org/abs/2505.13058)
- [arXiv:2509.11131 — Neural Cellular Automata: Applications to Biology and Beyond Classical AI (Hartl, Levin, Pio-Lopez, 2025)](https://arxiv.org/abs/2509.11131)
- [arXiv:2603.10055 — Training Language Models via Neural Cellular Automata (2026)](https://arxiv.org/abs/2603.10055)
- [arXiv:2506.21734 — Hierarchical Reasoning Model (Sapient, 2025)](https://arxiv.org/abs/2506.21734)
- [ARC Prize blog — The Hidden Drivers of HRM's Performance on ARC-AGI](https://arcprize.org/blog/hrm-analysis)
- [arXiv:2510.04871 — Tiny Recursive Model (Jolicoeur-Martineau, Samsung SAIL Montreal, 2025)](https://arxiv.org/abs/2510.04871)
- [github.com/SamsungSAILMontreal/TinyRecursiveModels — official TRM code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [github.com/lucidrains/tiny-recursive-model — unofficial TRM implementation](https://github.com/lucidrains/tiny-recursive-model)
- [MarkTechPost — TRM 7M param scores 45% ARC-AGI-1, 8% ARC-AGI-2](https://www.marktechpost.com/2025/10/09/tiny-recursive-model-trm-a-tiny-7m-model-that-surpass-deepseek-r1-gemini-2-5-pro-and-o3-mini-at-reasoning-on-both-arg-agi-1-and-arc-agi-2/)
- [ARC Prize 2025 Results and Analysis — paper awards list, NVARC 24% ARC-AGI-2 top score, ARC-NCA runner-up](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Prize 2024 Winners](https://arcprize.org/blog/arc-prize-2024-winners-technical-report)
- [Mordvintsev (@zzznah) on X — endorsement of Differentiable Logic CA](https://x.com/zzznah/status/1897741626262380835)
- [arXiv:2305.18354 — LLMs and ARC (source of the 1D-ARC dataset)](https://arxiv.org/html/2305.18354v2)
- [awesome-neural-cellular-automata list (dwoiwode)](https://github.com/dwoiwode/awesome-neural-cellular-automata)
