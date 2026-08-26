# EpiZoo

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

| Feature | Description |
| --- | --- |
| Species | Human and mouse |
| Cells | ~20.9 million |
| Datasets | 42 public datasets |
| Biological contexts | >30 tissues and cell lines |
| Model size | ~2.6 billion parameters |

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

EpiZoo has been tested with the following environment:

| Component | Tested version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.7.1 |
| PyTorch CUDA | 12.8 |
| CUDA Toolkit | 12.8.1 |
| FlashAttention | 2.8.3 |
| GPU | NVIDIA RTX 4090 |

We recommend using a Conda environment to ensure reproducibility.

### 1. Clone EpiZoo

``` bash
git clone https://github.com/likeyi19/EpiZoo.git
cd EpiZoo
```

### 2. Create the Conda environment

``` bash
conda env create -f environment.yml
conda activate epizoo
```

The environment includes Python 3.11 and CUDA Toolkit 12.8.1.

### 3. Install PyTorch

Install the tested PyTorch build with CUDA 12.8 support:

``` bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install FlashAttention

EpiZoo uses FlashAttention for efficient Transformer computation.

The tested version is FlashAttention 2.8.3.

``` bash
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
```

### 5. Install EpiZoo dependencies

``` bash
pip install -r requirements.txt
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

## Model checkpoint

The pretrained EpiZoo checkpoint can be downloaded at [pretrained_EpiZoo.pth](https://drive.google.com/file/d/1Xs5R_LAMbB_Zqpg7SFHlrAMfVcdcGwVE/view?usp=drive_link))

## License

MIT License

This project is released under the MIT License.
