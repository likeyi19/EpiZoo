<img width="432" height="48" alt="image" src="https://github.com/user-attachments/assets/f9cc16fa-dea2-4773-9aeb-c96996d7cb64" /><img width="432" height="12" alt="image" src="https://github.com/user-attachments/assets/c9103695-97c3-487d-bdd8-c827f2c4467b" /><img width="432" height="24" alt="image" src="https://github.com/user-attachments/assets/b269f436-4ba8-492a-9270-44c0b8a76c2e" /># EpiZoo

## A DNA sequence-aware foundation model for cross-species single-cell epigenomics

<p align="center">
  <img src="https://raw.githubusercontent.com/likeyi19/EpiZoo/main/inst/model.png" width="700" alt="EpiZoo model overview">
</p>

Single-cell epigenomic atlases characterize chromatin regulatory landscapes across diverse biological contexts, providing an opportunity to capture the full spectrum of cellular diversity in these atlases. However, current models remain largely confined to individual species by genomic coordinate dependence and overlook regulatory information encoded in DNA sequences.

**EpiZoo** is a DNA sequence-aware foundation model for cross-species single-cell epigenomics.

EpiZoo integrates:

-   DNA-encoded regulatory information
-   Sequence-independent epigenomic context
-   Accessibility-based importance

to build a unified representation space for single-cell epigenomics and enable cross-species regulatory analysis.

## Model architecture

EpiZoo converts single-cell ATAC-seq profiles into compact **cell sentences** composed of accessible cCRE tokens.

The model contains three core modules:

### DNA sequence-aware embedding module

For each cCRE token:

``` text
token embedding = sequence embedding + identity embedding + rank embedding
```

The sequence-to-embedding anchoring module (SEAM) encodes the underlying DNA sequence of each cCRE to provide regulatory priors and reduce dependence on species-specific genomic coordinates. The learnable identity embedding captures sequence-independent epigenomic context. The rank embedding encodes the importance of each accessible cCRE according to its TF-IDF ranking.

### Mixture-of-experts (MoE) transformer

The MoE transformer captures long-range co-accessibility patterns while allowing different experts to specialize in regulatory heterogeneity associated with species, tissues and cell types.

### Species-specific signal decoders

Species-specific decoders reconstruct accessibility landscapes and support signal prediction and imputation.

## Pretraining on Omni-scATAC

EpiZoo is pretrained on Omni-scATAC, a manually curated multi-species scATAC-seq corpus.

  Feature               Description
  --------------------- -----------------------------
  Species               Human and mouse
  Cells                 \~20.9 million
  Datasets              42 public datasets
  Biological contexts   \>30 tissues and cell lines
  Model size            \~2.6 billion parameters

## Capabilities

### Cell embedding extraction

EpiZoo generates informative cell embeddings for:

-   cell clustering
-   feature extraction
-   trajectory analysis

### Cell type annotation

EpiZoo enables robust cell type annotation across tissues and independent datasets.

### Data imputation

EpiZoo reconstructs missing accessibility signals and improves downstream analysis of sparse scATAC-seq data.

## Cross-species foundation modeling

The sequence-aware design enables adaptation to species beyond human and mouse.

Demonstrated adaptations include:

-   macaque
-   zebrafish
-   fruit fly
-   maize

## EpiZoo-Evo: regulatory evolution across primates

EpiZoo-Evo enables joint analysis of human and macaque brain epigenomes by learning shared and divergent regulatory representations.

Applications:

-   cross-species cell comparison
-   conserved and divergent cCRE discovery
-   regulatory evolution analysis

## EpiZoo-Cancer

EpiZoo-Cancer combines DNA sequence modeling with cancer regulatory contexts to prioritize noncoding somatic mutations.

Applications:

-   regulatory mutation prioritization
-   cancer-specific interpretation

## Chromatin accessibility prediction

By combining SEAM-derived sequence embeddings with learned cell-type embeddings, EpiZoo predicts cell-type-specific chromatin accessibility from DNA sequences and supports nucleotide-level interpretation.

## Installation

``` bash
git clone https://github.com/likeyi19/EpiZoo.git
cd EpiZoo
pip install -e .
```

## Tutorials

  Tutorial                     Description
  ---------------------------- ---------------------------------
  01_extract_cell_embeddings   Extract EpiZoo cell embeddings
  02_finetune_epizoo           Fine-tune EpiZoo
  03_posttrain_new_species     Adapt EpiZoo to new species
  04_annotation                Cell type annotation
  05_data_imputation           Accessibility imputation
  06_cancer                    Mutation prioritization
  07_sequence_prediction       Sequence-to-function prediction

## Citation

If you use EpiZoo in your research, please cite:

``` text
Li K, Chen X et al.
EpiZoo: a DNA sequence-aware foundation model for cross-species single-cell epigenomics.
```

## License

MIT License

This project is released under the MIT License.
