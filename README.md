# EpiZoo

<p align="center">
  <img src="https://raw.githubusercontent.com/likeyi19/EpiZoo/main/inst/model.png" width="700" alt="EpiZoo model overview">
</p>

## DNA sequence-aware foundation model for cross-species single-cell epigenomics

EpiZoo is a DNA sequence-aware foundation model designed to learn
transferable regulatory programs from large-scale single-cell chromatin
accessibility atlases.

Single-cell ATAC-seq profiles are extremely sparse and high-dimensional,
while cis-regulatory elements (cCREs) are defined by species-specific
genomic coordinates and often undergo rapid evolutionary turnover.
EpiZoo addresses these challenges by converting single-cell epigenomic
profiles into compact **cell sentences**, incorporating DNA
sequence-derived regulatory priors, epigenomic context, and
accessibility importance into a unified transformer framework.

EpiZoo is pretrained on the manually curated **Omni-scATAC corpus**,
containing approximately **20.9 million human and mouse single-cell
epigenomic profiles** across diverse tissues and cell states. The model
combines a sequence-aware embedding module based on SEAM, a
mixture-of-experts (MoE) transformer, and species-specific signal
decoders to learn generalizable regulatory representations.

## Overview

EpiZoo is designed to bridge three levels of regulatory modeling:

    DNA sequence
          |
          v
    Sequence-derived regulatory priors
          |
          +----------------+
          |                |
          v                v
    Cell-level regulatory states ----> Cross-species analysis
          |
          v
    Context-aware regulatory interpretation

The major design principles are:

-   **Sequence-aware representation learning**
    -   Encode cCRE DNA sequences using SEAM to provide transferable
        regulatory priors.
    -   Reduce dependence on fixed species-specific genomic coordinates.
-   **Large-scale multi-species pretraining**
    -   Learn conserved and species-specific regulatory programs from
        millions of scATAC-seq profiles.
-   **Efficient cell sentence representation**
    -   Transform sparse accessibility profiles into TF-IDF-ranked cCRE
        token sequences.
-   **Flexible downstream adaptation**
    -   Support fine-tuning, new species adaptation, sequence-based
        prediction, and regulatory interpretation.

## Main capabilities

### 1. Single-cell epigenomic representation learning

EpiZoo learns compact cell embeddings that preserve cellular
heterogeneity from sparse chromatin accessibility profiles.

Applications:

-   Cell embedding extraction
-   Cell clustering
-   Developmental trajectory analysis
-   Low-dimensional visualization

### 2. Cell type annotation

Using learned regulatory representations, EpiZoo supports accurate cell
type annotation across datasets and tissues.

Applications:

-   Reference-based annotation
-   Cross-dataset annotation
-   Fine-grained cell state identification

### 3. Chromatin accessibility imputation

EpiZoo reconstructs missing accessibility signals by projecting cell
representations back into the cCRE space.

Applications:

-   Dropout correction
-   Signal enhancement
-   Improved downstream analysis

### 4. Cross-species epigenomic modeling

The sequence-aware architecture enables adaptation to species beyond
human and mouse.

Supported examples include:

-   Macaque
-   Zebrafish
-   Fruit fly
-   Maize

EpiZoo can establish species-specific or cross-species foundation models
through post-training.

### 5. Evolutionary regulatory analysis

EpiZoo-Evo extends EpiZoo for comparative primate epigenomics.

It enables:

-   Joint human-macaque regulatory modeling
-   Identification of conserved and divergent cCRE programs
-   Discovery of putative functionally analogous regulatory elements

### 6. Cancer regulatory variant prioritization

EpiZoo-Cancer integrates cancer context with sequence-aware modeling to
estimate the regulatory effects of noncoding somatic mutations.

Applications:

-   Mutation prioritization
-   Cancer-type-specific regulatory interpretation
-   Identification of candidate driver-associated regulatory disruptions

### 7. Sequence-to-accessibility prediction

By combining DNA sequence embeddings with learned cell-type embeddings,
EpiZoo predicts cell-type-specific chromatin accessibility from genomic
sequences.

Applications:

-   Regulatory element interpretation
-   Enhancer activity prediction
-   Motif-level sequence attribution

## Repository structure

    EpiZoo/
    |
    ├── epizoo/
    │   ├── models/          # Model architectures and parameter transfer utilities
    │   ├── data/            # Dataset processing and cCRE utilities
    │   ├── train/           # Training modules and objectives
    │   ├── inference/       # Embedding extraction and prediction workflows
    │   ├── metrics/         # Evaluation metrics
    │   └── visualization/   # Visualization utilities
    |
    ├── examples/
    │   ├── 01_extract_cell_embeddings.ipynb
    │   ├── 02_finetune_epizoo.ipynb
    │   ├── 03_posttrain_epizoo_for_new_species.ipynb
    │   ├── 04_annotate_cell_types.ipynb
    │   ├── 05_impute_data.ipynb
    │   ├── 06_compute_loa_score.ipynb
    │   └── 07_predict_peak.ipynb
    |
    ├── config/
    ├── data/
    └── requirements.txt

## Installation

Create a Python environment (Python \>= 3.10 recommended):

``` bash
git clone https://github.com/likeyi19/EpiZoo.git
cd EpiZoo
pip install -e .
```

or:

``` bash
pip install -r requirements.txt
```

Some functions require additional external tools:

-   UCSC liftOver
-   bedtools
-   DNABERT-2 model weights/tokenizer

## Quick start

### Extract cell embeddings

See:

    examples/01_extract_cell_embeddings.ipynb

### Fine-tune EpiZoo

See:

    examples/02_finetune_epizoo.ipynb

### Adapt EpiZoo to new species

See:

    examples/03_posttrain_epizoo_for_new_species.ipynb

### Cell type annotation

See:

    examples/04_annotate_cell_types.ipynb

### Data imputation

See:

    examples/05_impute_data.ipynb

### Cancer mutation prioritization

See:

    examples/06_compute_loa_score.ipynb

### Sequence-based accessibility prediction

See:

    examples/07_predict_peak.ipynb

## Citation

If you use EpiZoo in your research, please cite:

    EpiZoo: a DNA sequence-aware foundation model for cross-species single-cell epigenomics.

## License

This project is released under the MIT License.
