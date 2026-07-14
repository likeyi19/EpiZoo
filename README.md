# EpiZoo

Large-scale single-cell epigenomic atlases characterize chromatin regulatory landscapes across diverse biological contexts, spanning cell types, tissues, individuals and species. Foundation models provide an opportunity to capture the full spectrum of cellular diversity in these atlases, yet current models remain largely confined to individual species by genomic coordinate dependence and overlook regulatory information encoded in DNA sequences. Here we introduce EpiZoo, a DNA sequence-aware foundation model for cross-species single-cell epigenomics. EpiZoo converts million-dimensional single-cell epigenomic profiles from diverse species into compact cell sentences that integrate DNA-encoded regulatory information, sequence-independent epigenomic context and accessibility-based importance. Built around a mixture-of-experts transformer and containing 2.6 billion parameters in total, EpiZoo is pretrained on our manually curated multi-species Omni-scATAC corpus of approximately 20.9 million cells to learn regulatory programs across species. On external datasets excluded from pretraining, EpiZoo achieves state-of-the-art performance in fundamental single-cell analysis tasks, including feature extraction, cell type annotation and data imputation. Its sequence-aware architecture enables extension to evolutionarily diverse species, and supports comparative analysis of regulatory conservation and divergence during primate evolution. Benefiting from this modeling design, EpiZoo enables context-aware prioritization of somatic mutations in cancer and prediction of cell-type-specific chromatin accessibility from DNA sequences across genomic regions and species.

<p align="center">
  <img src="https://github.com/likeyi19/EpiZoo/blob/main/inst/model.png" width="700" height="385" alt="image">
</p>

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
