## Volcan v1.0 — 9.7% on ARC-AGI-2 subset

Base checkpoint for the Volcan NCA (111,195 parameters). Reproduces the
3/31 exact-match result from the paper on the 30-task ARC-AGI-2 subset.

**Reproduce:**

```bash
git clone https://github.com/Tompark0927/volcan
cd volcan
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# download this checkpoint into outputs/week8/
mkdir -p outputs/week8
curl -L -o outputs/week8/volcan_dream_wide.pt \
    https://github.com/Tompark0927/volcan/releases/download/v1.0/volcan_dream_wide.pt

# run the full eval (30 tasks, ~1-1.5h on Apple Silicon MPS)
python scripts/eval_pretrained.py \
    --checkpoint outputs/week8/volcan_dream_wide.pt \
    --ttt-steps 150
```

Expected output: **3/31 = 9.7% +TTT accuracy** on the 30-task batch
(passing tasks: `25d8a9c8`, `b1948b0a`, `32597951`).

**Checkpoint provenance:** 111K-parameter dense Volcan, pretrained on
`dream_wide` (242 LLM-code-dreamed tasks, 70 rule families) for 2000
gradient steps with ICL on (K=3 demos, 1 query) tuples. Pretraining peak
80.5% masked-denoising accuracy on `dream_wide`.

**Not included:** hypernet.pt (Hyper-TTT variant regressed to 2/31),
volcan_dense1024.pt (dense-scaling variant went 0/7 before MPS crash).
Both negative-result checkpoints are reproducible from scratch via
`scripts/pretrain.py` + `scripts/hyperttt_stage1.py`/`stage2.py`.

**Paper:** docs/paper_draft.md (Markdown) and paper/volcan.tex (arXiv
source). Eight controlled ablations documented in §6.2.
