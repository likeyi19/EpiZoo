# EpiZoo

EpiZoo is a multi-task foundation model toolkit for single-cell chromatin accessibility modeling across species, tasks, and sequence-level perturbations. This repository contains a refactored, modular implementation of the EpiZoo model family, including foundation fine-tuning, data imputation, cell type annotation, cross-species transfer, cancer-context modeling, sequence accessibility prediction, and attribution analysis.

The current implementation follows a clean separation of responsibilities:

- `epizoo.models`: model definitions and checkpoint transfer utilities
- `epizoo.data`: datasets, collators, cCRE utilities, and preprocessing helpers
- `epizoo.train`: task-specific trainers and training losses
- `epizoo.inference`: embedding extraction, signal prediction, sequence prediction, and mutation scoring
- `epizoo.metrics`: task metrics and correlation/LoA score calculation
- `epizoo.visualization`: plotting utilities

## Installation

Create a clean Python environment first. Python 3.10 or newer is recommended.

```bash
git clone <your-repo-url> EpiZoo_v3
cd EpiZoo
pip install -e .
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Some functionality depends on external command-line tools that are not installed by `pip`:

- UCSC `liftOver`, required by `epizoo.data.ccre.build_ccre_map`
- `bedtools`, required by `epizoo.data.ccre.build_ccre_map`
- DNABERT-2 weights/tokenizer, required by `SEAM` and `EpiZooSeq`

Optional visualization and attribution utilities require `logomaker` and `captum`.

## Citation

If you use EpiZoo in your work, please cite the corresponding manuscript once available.
