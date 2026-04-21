# Volcan

**Morphogenetic Cellular Automata for ARC-AGI-2** — a 111,195-parameter Neural Cellular Automaton that reaches **9.7% exact-match (3/31) on an ARC-AGI-2 held-out subset** using biologically-inspired priors and $D_8$-augmented LoRA-rank-16 test-time training.

At **1.6% of TRM's parameter count** (7M → 111K) Volcan matches TRM's 8% on ARC-AGI-2 on our sample, setting a new small-model efficiency point for the benchmark.

- Paper: [docs/paper_draft.md](docs/paper_draft.md) (arXiv-ready LaTeX in [paper/](paper/))
- Full design: [docs/architecture.md](docs/architecture.md)

## Headline result

| System | Params | ARC-AGI-2 |
|---|---|---|
| **Volcan (this work)** | **111K** | **9.7%** ($n=31$) |
| CompressARC (Liao & Gu 2025) | 76K | 4.0% |
| TRM (Jolicoeur-Martineau 2025) | 7M | 8.0% |
| ARChitects 2025 | ~8B | 16.5% |
| NVARC 2025 | ~4B | 24.0% |

Through **seven controlled ablations** (data scale, MoE capacity, rule diversity, LoRA rank, hierarchical macro-cells, HyperNetwork meta-init, base architecture) we show 9.7% is a robust ceiling for this class of approach — not a tuning artifact.

## Quickstart

```bash
# create venv + install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# eval the 9.7% checkpoint on the 30-task ARC-AGI-2 subset
# (checkpoint is a GitHub Release asset — see the Releases tab)
python scripts/eval_pretrained.py \
    --checkpoint outputs/week8/volcan_dream_wide.pt \
    --ttt-steps 150
```

Requires Apple Silicon (MPS) or CUDA. Wall time on M-series MPS: ~2-3 min per task × 30 tasks ≈ 1–1.5 hours for the full eval.

## The five pillars

Volcan's architecture is organized around five biologically-motivated mechanisms:

1. **Multi-scale force transmission** — anisotropic directional forces (8 Moore-neighbor channels) + mycelial sparse small-world attention + Laplacian spectral tension for $O(1)$ global flow.
2. **Resonance detection** — echo metrics ($\cos(s_t, s_{t-k})$) as dynamical-regime classifier; period-2 oscillation is information, not failure.
3. **Sequential bioelectric → pixel** — Phase A (Ghost Dream, color clamped) stabilizes a ghost pre-pattern before Phase B (Crystallization, color denoising). Inspired by Levin's bioelectric morphogenesis.
4. **Masked denoising** — replaces cross-entropy; target is corrupted at $\sigma \in [0.2, 0.95]$ and the NCA denoises toward clean.
5. **Apoptotic pruning** — cells incoherent with the emerging regime commit to background.

See [docs/architecture.md](docs/architecture.md) for the formal per-pillar spec with biological and computational justifications.

## Data pipeline

- `src/volcan/code_dreamer.py` — LLM (qwen2.5:7b via Ollama) writes Python `transform(grid) -> grid` functions. We execute them to generate 4 guaranteed-consistent demos per task.
- `src/volcan/dream_filter.py` — overfit filter rejects tasks our base model can't memorize within a time budget.
- `data/dream_wide/` — 242 tasks, 70 rule families, our final corpus.

## Layout

```
volcan/
├── docs/
│   ├── architecture.md      v0.2.1 design doc with 8 weeks of revision log
│   └── paper_draft.md       the paper, Markdown source
├── paper/
│   ├── volcan.tex           LaTeX source (ready for arXiv)
│   ├── refs.bib             verified bibliography
│   └── README.md            build instructions
├── src/volcan/              the Python package
│   ├── volcan_cell.py       core 5-pillar NCA cell
│   ├── code_dreamer.py      LLM-to-Python data pipeline
│   ├── training_volcan.py   pretrain + D8+LoRA TTT loops
│   ├── hyperttt.py          HyperNetwork meta-init variant (negative result)
│   ├── hierarchy.py         macro-cell multi-timescale variant (negative result)
│   ├── moe.py               mixture-of-experts variant (negative result)
│   └── lora.py              LoRA adapter (dense + MoE aware)
├── scripts/                 runnable entrypoints
├── data/                    ARC + dream_wide (weights in Releases)
└── outputs/                 checkpoints + training logs
```

## Citing

If you use Volcan or its negative-result ablations, please cite:

```bibtex
@misc{park2026volcan,
  title        = {Volcan: Morphogenetic Cellular Automata for Abstract Reasoning via Code-Dreamed Pretraining},
  author       = {Park, Tom},
  year         = {2026},
  eprint       = {arXiv:TBD},
  url          = {https://github.com/Tompark0927/volcan},
}
```

## License

MIT. See [LICENSE](LICENSE).
