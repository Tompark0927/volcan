# Data

This directory holds ARC datasets. They are **not committed to git** (see `.gitignore`) because:
- ARC-AGI-1: ~2.5 MB but we want the canonical source
- ARC-AGI-2: ~4 MB, same reason
- Synthetic data: can grow to 100K+ tasks, too big

## Downloading ARC-AGI-1 and ARC-AGI-2

```bash
cd data/
git clone https://github.com/fchollet/ARC-AGI.git
git clone https://github.com/arcprize/ARC-AGI-2.git
```

After cloning:

```
data/
├── ARC-AGI/
│   └── data/
│       ├── training/       # 400 public training tasks
│       └── evaluation/     # 400 public eval tasks
└── ARC-AGI-2/
    └── data/
        ├── training/       # ARC-AGI-2 public training tasks
        └── evaluation/     # ARC-AGI-2 public eval tasks
```

Volcan's [arc.py](../src/volcan/arc.py) loader auto-detects both layouts and picks ARC-AGI-2 by default.

## Task format

Each task is a JSON file: `{"train": [{"input": [[...]], "output": [[...]]}, ...], "test": [{"input": [[...]], "output": [[...]]}]}`.

Grids are 2D arrays of integers in [0, 9]. Grid sizes vary from 1×1 up to 30×30.
