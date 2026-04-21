# Week 8 Scaling Research: 3.2% → 8% → Higher

*Date: 2026-04-12*

---

## 1. Scaling Curves for Synthetic-Data Pretraining on ARC

### Published data points

**NVARC (Sorokin & Puget, 2025):** Scaled from BARC's ~600 seed tasks → 103K verified synthetic puzzles (3.2M augmented variants). Final score: 24.03% on ARC-2 private. They did not publish intermediate scaling curves, but the jump from ARChitects-baseline (~16.5% with RE-ARC-level data) to 24% with 103K tasks implies roughly **+7.5 pp from ~100× more synthetic tasks** — consistent with log-linear scaling with a slope of ~2.5 pp per order of magnitude of tasks ([NVARC repo](https://github.com/1ytic/NVARC), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

**MIT TTT (Akyürek et al., 2024):** On their 80-task ARC-1 dev set (pass@2): base fine-tune alone = 5%. With TTT on RE-ARC (~600 generators) = 29%. Adding BARC neural model = 53%. Adding program synthesis = 61.875%. The critical ablation: removing LLM-generated synthetic tasks from pretraining *hurt* by ~5 pp — raw LLM tasks without verification degrade quality. Their synthetic data was ~6,400 LLM-generated generators on top of ~600 RE-ARC generators. No clean task-count scaling curve was published, but the RE-ARC-only → RE-ARC+BARC jump (~600 → ~7K generators) gave roughly **+24 pp on their dev set** ([arXiv:2411.07279](https://arxiv.org/html/2411.07279v1)).

**BARC (Li, Ellis, Tavares, 2024):** Generated ~160K synthetic ARC-like tasks from LLM-produced Python programs; after filtering, used ~100K for training. BARC transduction model alone: ~40% on ARC-1 public eval. Combined with induction: 56.75%. With TTT: 62.8%. They did not publish a task-count ablation ([arXiv:2411.02272](https://arxiv.org/abs/2411.02272)).

**CompressARC (Liao & Gu, 2025):** No pretraining at all — 76K params trained from scratch per task. 20% ARC-1, 4% ARC-2. The MDL scaling they report is not task-count scaling but network-capacity scaling: accuracy increases with allowed parameter count up to ~76K, then plateaus. Relevant insight: at 76K params, the compression objective alone gets 4% on ARC-2 without any prior — this is our floor for what zero-pretraining achieves at our parameter scale ([arXiv:2512.06104](https://arxiv.org/abs/2512.06104)).

### Projected scaling for Volcan (111K params, NCA)

The key difference from transformer-based systems: our model has 111K params vs. NVARC's 4B or MIT TTT's 8B. Smaller models saturate faster on diverse data. The literature suggests:

| Dream tasks | Projected pretrain accuracy | Projected ARC-2 with TTT | Basis |
|---|---|---|---|
| 43 (current) | 76% | 3.2% (measured) | Actual |
| 500 | ~82-85% | 4-6% | Log-linear extrapolation from NVARC slope, discounted for capacity |
| 5,000 | ~87-90% | 6-9% | This is where we likely hit the 8% TRM target |
| 50,000 | ~90-92% (saturation) | 8-12% | Diminishing returns at 111K params; model capacity becomes the bottleneck |

**Honest assessment:** Reaching 8% likely requires ~5K verified dream tasks AND TTT improvements (Section 2). At 111K params we will saturate earlier than NVARC's 4B-param model. Scaling beyond 50K tasks without increasing model capacity is unlikely to help. The cheap-first path: scale to 500 tasks (validate the slope), then 5K (target 8%), then reassess whether params or data is the bottleneck.

---

## 2. TTT Optimization Techniques Ranked by Impact

Ranked by estimated impact for our specific setup (111K-param NCA, 150 TTT steps, full-weight fine-tuning).

### Tier 1: High impact (each likely +1-3 pp on ARC-2)

**1. D8 augmentation during TTT.** Both ARChitects and MIT TTT use the full dihedral group (8 symmetries) plus color permutations during test-time training. ARChitects' product-of-experts paper shows augmentation-aware scoring alone gives +5 pp on ARC-1 (66.6% → 71.6%). For TTT specifically, the MIT ablation shows removing augmentation drops from 29% → ~13% on their dev set — roughly **halving performance**. For NCAs this is even more important because local rules are naturally equivariant to translation but NOT to rotation/reflection. Cost: zero compute overhead (just transform the 2-3 demos 8 ways). **Do this immediately.** ([arXiv:2505.07859](https://arxiv.org/html/2505.07859v1), [arXiv:2411.07279](https://arxiv.org/html/2411.07279v1))

**2. Leave-one-out augmentation during TTT.** MIT TTT's signature trick: for a task with demos {A, B, C}, construct TTT training examples where A is the "test" and {B, C} are context, B is "test" and {A, C} are context, etc. This multiplies the effective TTT training set by (N_demos - 1) for each augmentation. MIT ablation: removing leave-one-out drops from 29% → 13% on their dev set (confounded with D8 removal). Independently estimated at **+5-8 pp on ARC-1** for transformer TTT. For our NCA (which trains on input→output pairs, not ICL sequences), the analog is: for each demo pair, train on the other demo pairs as additional signal, then validate on the held-out pair. This also gives us a natural **early-stopping criterion** — stop TTT when held-out demo loss stops improving. **Do this immediately.** ([arXiv:2411.07279](https://arxiv.org/html/2411.07279v1))

**3. LoRA instead of full-weight TTT.** MIT TTT uses LoRA rank 128 on all layers. Their ablation: task-shared LoRA vs. per-task LoRA = 22% vs. 29% — a **+7 pp** delta. The argument for LoRA over full-weight: it preserves the pretrained prior. With only 2-3 demos, full-weight fine-tuning of 111K params risks catastrophic forgetting of the pretrained rule library. LoRA rank 16-32 on our NCA's MLP layers would adapt ~5-10K params per task while keeping the other ~100K frozen. **Strong recommendation for our setup specifically** — our model is small enough that full-weight TTT on 2 demos is severe overfitting risk. ([arXiv:2411.07279](https://arxiv.org/html/2411.07279v1))

### Tier 2: Medium impact (each likely +0.5-1.5 pp)

**4. Voting / ensembling over TTT runs.** ARChitects' product-of-experts: score each candidate under 16 augmented perspectives, take geometric mean. This gives +5 pp on ARC-1 (product vs. sum: 71.6% vs. 66.6%). For NCAs, the analog: run TTT from multiple random seeds, collect outputs, majority-vote per pixel. ARC allows 2 submissions per test input — submit the top-2 by vote count. Estimated +1-2 pp on ARC-2 for a 5-seed ensemble. Compute cost: 5× per task. ([arXiv:2505.07859](https://arxiv.org/html/2505.07859v1))

**5. TTT step count optimization.** Xu & Miikkulainen (2025) use 800 epochs; ARC-NCA uses 3000 iterations; our current 150 steps may be too few. The overfitting curve for NCAs on 2-3 demos typically shows: underfit until ~200 steps, good generalization at 300-800 steps, overfit beyond ~1000 steps (per ARC-NCA's reported failure modes). With LoRA constraining capacity, we can safely increase to 300-500 steps. **Use leave-one-out validation loss to find the sweet spot per task.** ([arXiv:2506.15746](https://arxiv.org/abs/2506.15746), [arXiv:2505.08778](https://arxiv.org/abs/2505.08778))

**6. TTT learning rate.** MIT TTT uses LR = 2.5e-5 for pretraining, similar magnitude for TTT. ARC-NCA (no pretraining) uses 1e-3 → 3.3e-4 cosine decay. Xu & Miikkulainen use 2e-3 → 1e-4 linear decay. General principle: TTT LR should be **5-10× the pretraining LR** when using LoRA, because you're adapting a small subspace. With full-weight TTT, use **1-2× pretraining LR** with cosine decay to avoid destroying priors. ([arXiv:2411.07279](https://arxiv.org/html/2411.07279v1), [arXiv:2505.08778](https://arxiv.org/abs/2505.08778))

### Tier 3: Worth trying later

**7. Multi-resolution TTT.** Train the NCA on downsampled versions of the demos first (2× smaller), then fine-tune on full resolution. No ARC-specific results, but standard practice in image generation for faster convergence. Estimated +0.5 pp, low cost.

**8. Temperature/noise injection during TTT.** Inject small Gaussian noise into the NCA state during TTT training steps to regularize and improve generalization. Related to our masked-denoising training objective. The ARChitects 2025 masked-diffusion variant implicitly does this. No ablation numbers available.

### Recommended TTT stack (implement in order)

1. D8 augmentation (free)
2. Leave-one-out construction + early stopping (free)
3. Switch to LoRA rank 16-32 (minor code change)
4. Increase steps to 300-500 with LR cosine decay (tune on 5 tasks)
5. 5-seed voting ensemble (5× compute, last because expensive)

**Combined projected impact: +3-6 pp on ARC-2**, taking us from 3.2% to an estimated 6-9% with the current 43 dream tasks. Combined with data scaling to 5K tasks, 8-12% is realistic.

---

## 3. Small-Model ARC-2 Landscape (2025-2026)

### TRM (7M params, 8% ARC-2)

Two follow-up papers published:
- "Test-time Adaptation of Tiny Recursive Models" ([arXiv:2511.02886](https://arxiv.org/abs/2511.02886)) — adds LoRA-based TTT to TRM. Improvement not quantified on ARC-2 in the abstract but establishes the TTT+TRM recipe.
- "Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute" ([arXiv:2512.11847](https://arxiv.org/abs/2512.11847)) — ablates identity conditioning and test-time compute scaling. Likely improves on the 45% ARC-1 / 8% ARC-2 baseline but I cannot confirm exact updated ARC-2 numbers.

TRM is integrated into NVARC's ensemble as a secondary component. Samsung SAIL Montreal is actively developing this line.

### CompressARC (76K params, 4% ARC-2)

No follow-up publications found as of April 2026. Isaac Liao's blog post remains the primary source. The MDL framing is influential (3rd place paper award) but the 20-min-per-task runtime makes it impractical for Kaggle. No one has published a "CompressARC + pretraining" hybrid.

### ARC-NCA (Guichard et al., ~100K params)

**No ARC-2 results published.** Their ALIFE 2025 paper and the ARC Prize runner-up award are both on ARC-1 only (12.9% single / 17.6% ensemble on 262/400 same-size tasks). The codebase ([github.com/etimush/ARC_NCA](https://github.com/etimush/ARC_NCA)) was last updated April 2025. No indication of ARC-2 work in progress. **We are likely the first NCA with an ARC-2 result.**

### New small-model approaches (late 2025 - 2026)

- **SOAR** (Pourcel et al., Jul 2025): Self-improving evolutionary program synthesis. 52% ARC-1 public test. No ARC-2 numbers. Not small-model per se (uses an LLM backbone) but the hindsight-relabeling trick is model-size-agnostic ([arXiv:2507.14172](https://arxiv.org/abs/2507.14172)).
- **Training LMs via NCA** (Han et al., 2026): Uses NCA-generated tokens for LM pre-pretraining. +6% on downstream tasks. Not an ARC attempt but signals NCA-as-compute-substrate is gaining traction ([arXiv:2603.10055](https://arxiv.org/abs/2603.10055)).
- **No published work combining NCAs with LLM-generated synthetic data for ARC.** This is our specific lane. The green-field status is confirmed.

### Competitive positioning

| System | Params | ARC-2 | Pretraining? | TTT? | Our advantage |
|---|---|---|---|---|---|
| TRM | 7M | 8% | Yes (ARC train + synthetic) | Yes (LoRA) | We're 64× smaller; if we match at 111K params, that's a paper |
| CompressARC | 76K | 4% | No | From-scratch per task | We have pretraining + TTT; they don't |
| ARC-NCA | ~100K | None on ARC-2 | No | From-scratch per task | We have pretraining + TTT + ARC-2 eval |
| Volcan (current) | 111K | 3.2% | Yes (43 tasks) | Yes (full-weight) | First NCA with pretraining+TTT on ARC-2 |

---

## 4. Cost-Effective Compute Strategies

### Synthetic task generation via Ollama on Apple Silicon

**Best local models for code generation (as of April 2026):**
- **Qwen2.5-Coder-32B-Q4** via Ollama: fits in 24GB unified memory on M2/M3 Pro. Generates Python functions at ~15-20 tok/s. Quality is comparable to GPT-4-class for simple grid transformations.
- **DeepSeek-Coder-V2-Lite (16B)** via Ollama: faster (~30 tok/s on M2 Pro), slightly lower quality. Good for high-volume generation with aggressive filtering.
- **CodeLlama-34B-Q4**: adequate but lower quality than Qwen2.5-Coder.

**Throughput estimate for 5K tasks:**
- Each dream task requires: 1 LLM call to generate Python function (~30s) + execution to generate grids (~1s) + overfit filter (~2-5 min per task on MPS).
- At 30s/generation, 5K raw generations = ~42 hours. With ~30-50% pass rate after filtering, need ~10-15K raw generations = ~4-5 days continuous generation on Mac.
- Overfit filter (the bottleneck): 5K tasks × 3 min = 250 hours = ~10 days sequential on MPS.

**Parallelizing the overfit filter:**
- Python `multiprocessing` with MPS is tricky — MPS doesn't support multiple processes sharing the GPU well. Options:
  1. **Sequential GPU, parallel CPU preprocessing**: Use `multiprocessing.Pool` for data loading/augmentation, single-process GPU training. Modest speedup (~1.5×).
  2. **Batch multiple tasks per GPU pass**: If your overfit filter trains one NCA per task, batch 4-8 tasks into a single training loop with independent weight tensors. MPS handles this well. Estimated 3-4× speedup.
  3. **CPU-only overfit filter for screening**: Run a cheap CPU-only version (fewer steps, smaller model) to pre-screen, then GPU-confirm survivors. 5-10× throughput for the screening pass.

### Cloud options for $100-500

| Option | Cost | What you get | Best use |
|---|---|---|---|
| **Vast.ai RTX 4090** | ~$0.30/hr | Single 4090, 10× faster than MPS for NCA training | Overfit filter: 5K tasks in ~25 hrs = ~$8. Pretraining runs. |
| **Vast.ai 2× RTX 3090** | ~$0.40/hr | Two GPUs, data-parallel pretraining | Scale pretraining to 5K+ tasks in hours not days |
| **Lambda Cloud A100** | ~$1.10/hr | Single A100 80GB | Overkill for 111K params but useful for large-batch pretraining |
| **RunPod RTX 4090** | ~$0.40/hr | Similar to Vast.ai, slightly better UX | Same use cases |
| **Google Colab Pro+** | $50/mo | A100/V100 sessions up to 24h | Good for experimentation, unreliable for long runs |

**Recommendation:** Vast.ai RTX 4090 at $0.30/hr. Budget allocation:
- $20-30: Generate + filter 5K dream tasks (overfit filter is the bottleneck, ~80 GPU-hours)
- $50-100: Pretrain on 5K tasks with hyperparameter sweeps
- $50-100: TTT optimization experiments (LoRA, D8, leave-one-out ablations)
- Reserve $100-200: Scale to 50K tasks if 5K results are promising
- **Total: $220-430, well within $2-5K budget**

### Optimal sequencing (cheap-first, upgrade-later)

1. **Week 8-9 (Mac only, $0):** Implement D8 + leave-one-out + LoRA TTT on current 43 tasks. Measure delta. Generate 500 dream tasks via Ollama overnight.
2. **Week 9-10 (Mac + $30 cloud):** Retrain on 500 tasks. Measure scaling slope. Use Vast.ai to run overfit filter on 5K candidate tasks.
3. **Week 10-11 ($100 cloud):** Pretrain on 5K tasks. Full TTT stack. Target: 8% on ARC-2 public eval.
4. **Week 11-12 ($100-200 cloud, only if slope holds):** Scale to 50K tasks. Voting ensemble. Target: 10-12%.

---

## Summary: What's Realistic vs. Aspirational

| Target | Realistic? | What it requires |
|---|---|---|
| **5% ARC-2** | Yes, high confidence | D8 + leave-one-out TTT on current 43 tasks |
| **8% ARC-2** (match TRM) | Yes, moderate confidence | 5K dream tasks + full TTT stack (LoRA, D8, LOO, voting) |
| **12% ARC-2** (match MindsAI) | Possible but hard | 50K dream tasks + model capacity increase to ~500K params |
| **15%+ ARC-2** | Aspirational | Would require architectural innovations beyond current plan |

**The single most important finding:** No one has published NCA + pretraining + TTT on ARC-2. CompressARC (no pretraining, 76K params) gets 4%. ARC-NCA (no pretraining, ~100K params) has no ARC-2 result. We are at 3.2% with 43 tasks and naive TTT. The literature strongly predicts that D8 augmentation alone during TTT should roughly double our score, and scaling to 5K verified tasks should add another 2-4 pp on top. **8% is within reach with known techniques and modest compute spend.**

---

## Sources

- NVARC: [github.com/1ytic/NVARC](https://github.com/1ytic/NVARC), [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- MIT TTT: [arXiv:2411.07279](https://arxiv.org/html/2411.07279v1)
- BARC: [arXiv:2411.02272](https://arxiv.org/abs/2411.02272)
- CompressARC: [arXiv:2512.06104](https://arxiv.org/abs/2512.06104)
- ARC-NCA: [arXiv:2505.08778](https://arxiv.org/abs/2505.08778)
- Xu & Miikkulainen NCA: [arXiv:2506.15746](https://arxiv.org/abs/2506.15746)
- TRM: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
- TRM TTT adaptation: [arXiv:2511.02886](https://arxiv.org/abs/2511.02886)
- TRM ablations: [arXiv:2512.11847](https://arxiv.org/abs/2512.11847)
- ARChitects product-of-experts: [arXiv:2505.07859](https://arxiv.org/html/2505.07859v1)
- ARC-AGI-2 technical report: [arXiv:2505.11831](https://arxiv.org/html/2505.11831v1)
- ARC Prize 2025 technical report: [arXiv:2601.10904](https://arxiv.org/html/2601.10904v1)
- SOAR: [arXiv:2507.14172](https://arxiv.org/abs/2507.14172)
- NCA for LM pretraining: [arXiv:2603.10055](https://arxiv.org/abs/2603.10055)
