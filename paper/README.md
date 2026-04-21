# Volcan — arXiv submission package

Files:
- `volcan.tex` — main paper source (LaTeX, `article` class, `natbib`).
- `refs.bib` — bibliography. **Verify all entries before arXiv upload** — some
  venue/arxiv-ID fields are approximated and should be confirmed against the
  actual sources.

## Local build (requires a TeX distribution — e.g. MacTeX, TeX Live)

```sh
pdflatex volcan
bibtex   volcan
pdflatex volcan
pdflatex volcan
```

Install TeX Live if you don't have it (`brew install --cask mactex-no-gui` on
macOS; the full MacTeX is overkill for a single paper).

## Pre-submission checklist

1. Fill in the author block: affiliation on line 19 of `volcan.tex` if not
   "Independent researcher".
2. Confirm the citation details in `refs.bib` — placeholder URLs and venue
   strings need to be replaced with the actual arXiv IDs / DOIs / URLs.
3. Replace `https://github.com/[author]/volcan` with the real repo URL once
   the public repo is live (section 8, "Code and checkpoint availability").
4. Compile locally; check that all `\cite{...}` calls resolve (no `?` in the
   output) and all cross-references render.
5. `arxiv-prep.sh` below creates the arXiv submission tarball.

## arXiv submission

arXiv wants a `.tar.gz` containing `.tex`, `.bib`, `.bbl`, and any figures
(none here). After a successful local build:

```sh
tar -czf volcan-arxiv.tar.gz volcan.tex refs.bib volcan.bbl
```

Upload `volcan-arxiv.tar.gz` via the arXiv web interface. Categories:
primary `cs.LG` (Machine Learning), cross-list `cs.NE` (Neural and
Evolutionary Computing).

## Endorsement

If you don't already have an arXiv account with prior submissions in
`cs.LG`, you'll need an endorsement from an existing `cs.LG` submitter
before uploading. arXiv will tell you this on first upload attempt.
