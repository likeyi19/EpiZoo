# EpiZoo

EpiZoo is a multi-task foundation model toolkit for single-cell chromatin accessibility modeling across species, tasks, and sequence-level perturbations. This repository contains a refactored, modular implementation of the EpiZoo model family, including foundation fine-tuning, data imputation, cell type annotation, cross-species transfer, cancer-context modeling, sequence accessibility prediction, and attribution analysis.

The current implementation follows a clean separation of responsibilities:

- `epizoo.models`: model definitions and checkpoint transfer utilities
- `epizoo.data`: datasets, collators, cCRE utilities, and preprocessing helpers
- `epizoo.train`: task-specific trainers and training losses
- `epizoo.inference`: embedding extraction, signal prediction, sequence prediction, and mutation scoring
- `epizoo.metrics`: task metrics and correlation/LoA score calculation
- `epizoo.visualization`: plotting utilities

## Repository layout

```text
EpiZoo_v3/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore

  epizoo/
    __init__.py

    models/
      __init__.py
      epizoo.py             # base EpiZoo model
      epizoo_di.py          # data imputation model, MSE signal loss
      epizoo_anno.py        # cell type annotation model
      epizoo_x.py           # single-vocabulary cross-species/post-training model
      epizoo_cancer.py      # cancer-context EpiZoo model
      epizoo_seq.py         # sequence accessibility prediction model
      seam.py               # Sequence Embedding Alignment Module
      lora.py               # LoRA utilities
      transfer.py           # checkpoint/vocabulary transfer utilities
      moe_transformer.py    # MoE Transformer encoder

    data/
      __init__.py
      datasets.py           # datasets and collate functions
      processing.py         # TF-IDF and cell sentence generation
      ccre.py               # cCRE mapping, joint vocabulary, FASTA sequence extraction

    train/
      __init__.py
      loss.py               # reusable training losses
      finetune.py           # EpiZoo/EpiZooDI fine-tuning trainer
      posttrain.py          # EpiZooX post-training trainer
      annotation.py         # cell type annotation trainer
      cancer.py             # cancer-context trainer
      seq.py                # EpiZooSeq trainer

    inference/
      __init__.py
      utils.py
      embeddings.py         # cell, sequence, and cell type embedding extraction
      signals.py            # signal prediction utilities
      seq.py                # sequence accessibility prediction and attribution
      mutations.py          # SNV LoA inference pipeline
      annotation.py         # cell type prediction

    metrics/
      __init__.py
      cca.py
      classification.py
      cluster.py
      correlations.py
      loa.py

    visualization/
      __init__.py
      umap.py
      violin.py
      heatmap.py
      density.py
      attribution.py
```

## Installation

Create a clean Python environment first. Python 3.10 or newer is recommended.

```bash
git clone <your-repo-url> EpiZoo_v3
cd EpiZoo_v3
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

## Token conventions

EpiZoo uses the following token convention:

```text
0 = [PAD]
1 = [CLS]
2 = [SEP]
3 = reserved
cCRE token ids start from 4
```

For datasets and signal targets:

- `num_ccres` means the number of cCREs, excluding special tokens.
- `vocab_size` means the full token vocabulary size, including special tokens.
- For a single-vocabulary model, `vocab_size = num_ccres + 4`.
- For base multi-species EpiZoo, human cCRE token ids start at `4`, while mouse cCRE token ids start at `4 + human_num_ccres`.

## Core models

### EpiZoo

`EpiZoo` is the base model for human/mouse multi-species single-cell chromatin accessibility modeling. It uses:

```text
ccre_emb + seq_emb + rank_emb -> MoE Transformer -> cell embedding
```

It supports two downstream objectives through explicit methods, not inside `forward`:

- `compute_signal_loss`
- `compute_cca_loss`

The `seq_emb` table is expected to be precomputed by SEAM and loaded as a lookup table. It is not dynamically computed by SEAM during EpiZoo training. Trainers freeze `seq_emb` by default.

### EpiZooDI

`EpiZooDI` reuses the base EpiZoo architecture but changes the signal reconstruction loss to MSE loss for data imputation.

### EpiZooAnno

`EpiZooAnno` removes CCA/SR heads and attaches a classifier to the cell embedding for cell type annotation. Its training objective uses focal loss.

### EpiZooX

`EpiZooX` is the single-vocabulary model used for post-training and cross-species settings. It has one signal decoder rather than species-specific decoders.

### EpiZooCancer

`EpiZooCancer` extends base EpiZoo with cancer-type context:

```text
cell_emb + cancer_emb -> context-aware cell_emb
```

During training, the cancer embedding can be randomly dropped at the sample level to prevent over-reliance on cancer type alone.

### SEAM

`SEAM` maps DNA sequence tokens into the EpiZoo sequence embedding space:

```text
DNABERT-2 hidden CLS embedding -> Linear projection -> 512-dim seq embedding
```

Tokenizer logic is kept in the data layer through `SEAMDataset`; the `SEAM` model only accepts `input_ids` and `attention_mask`.

### EpiZooSeq

`EpiZooSeq` predicts sequence accessibility across cell types:

```text
SEAM(sequence) + cell_type_embedding -> MLP -> Softplus accessibility
```

It is designed for predicting the accessibility of a DNA sequence in multiple cell types and supports gradient attribution analysis.

## Data preprocessing

### TF-IDF normalization

```python
from epizoo.data import compute_tfidf

adata = compute_tfidf(
    adata,
    df=None,                 # optional document frequency
    n_cells=None,            # defaults to adata.n_obs when df is None
    scale_factor=10_000.0,
    layer=None,              # use adata.X by default
    target="X",             # or target="obsm"
    obsm_key="X_tfidf",
)
```

### Generate cell sentences

```python
from epizoo.data import make_cell_sentences

adata = make_cell_sentences(
    adata,
    layer=None,              # use adata.X by default
    obsm_key=None,
    obs_key="cell_indices",
    species=0,               # 0 human, 1 mouse, or None
    base_offset=4,
    species_offset=0,
)
```

### Extract DNA sequences from cCRE regions

```python
from epizoo.data import extract_dna_sequences

sequences = extract_dna_sequences(
    fasta_path="/path/to/genome.fa",
    regions=adata.var_names.tolist(),  # chr:start-end
    on_error="raise",
)
```

### Build cCRE maps with liftOver and bedtools

```python
from epizoo.data import build_ccre_map

ccre_map = build_ccre_map(
    new_bed="new_species.ccre.bed",
    ref_bed="human.ccre.bed",
    chain_file="newToHuman.over.chain.gz",
    liftover_bin="liftOver",
    bedtools_bin="bedtools",
)

# ccre_map format:
# {new_ccre_idx: reference_ccre_idx}
```

### Build joint cCRE vocabulary maps

```python
from epizoo.data import build_joint_ccre_maps, get_joint_ccre_count

ref_to_joint, new_to_joint = build_joint_ccre_maps(
    ref_num_ccres=1_355_445,
    new_num_ccres=92_145,
    ccre_map=ccre_map,
    ccre_offset=4,
    return_token_ids=True,
)

joint_num_ccres = get_joint_ccre_count(
    ref_to_joint=ref_to_joint,
    new_to_joint=new_to_joint,
    ccre_offset=4,
    ids_include_offset=True,
)
```

## Sequence embedding extraction

```python
import torch
from torch.utils.data import DataLoader

from epizoo.models import SEAM, SEAMConfig
from epizoo.data import SEAMDataset, collate_fn_seam
from epizoo.inference import extract_seq_embeddings

seam = SEAM(
    SEAMConfig(
        dnabert_path="/path/to/dnabert2",
        emb_dim=512,
    )
)

seam.load_state_dict(torch.load("/path/to/seam_checkpoint.pt", map_location="cpu"))

dataset = SEAMDataset(
    sequences=ccre_sequences,
    dnabert_path="/path/to/dnabert2",
    max_length=512,
    return_index=True,
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_seam,
)

seq_embeddings = extract_seq_embeddings(
    model=seam,
    dataloader=loader,
    device="cuda",
    return_numpy=True,
)
```

## Checkpoint transfer

### Transfer to a new single-vocabulary species without cCRE mapping

```python
import torch
from epizoo.models import transfer_epizoox_state_dict

state_dict = torch.load("base_epizoo.pth", map_location="cpu")

new_state_dict = transfer_epizoox_state_dict(
    state_dict=state_dict,
    seq_embeddings=seq_embeddings,
)

torch.save(new_state_dict, "epizoox_new_species.pth")
```

### Transfer with cCRE mapping

```python
from epizoo.models import transfer_epizoox_state_dict_with_map

new_state_dict = transfer_epizoox_state_dict_with_map(
    state_dict=state_dict,
    seq_embeddings=seq_embeddings,
    ccre_map=ccre_map,       # {new_idx: reference_idx}
    source_species="human", # or "mouse"
    human_vocab_size=1_355_445,
)
```

### Transfer to a joint vocabulary

```python
from epizoo.models import transfer_epizoox_joint_state_dict

joint_state_dict = transfer_epizoox_joint_state_dict(
    state_dict=state_dict,
    new_seq_embeddings=new_seq_embeddings,
    ref_to_joint=ref_to_joint,
    new_to_joint=new_to_joint,
    source_species="human",
)
```

## Datasets and collators

### Base EpiZoo training

```python
from torch.utils.data import DataLoader
from epizoo.data import CellDataset, collate_fn

train_dataset = CellDataset(
    cell_sentences=adata.obs["cell_indices"].values,
    species=adata.obs["species"].values,
    human_num_ccres=1_355_445,
    mouse_num_ccres=1_341_077,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
)
```

### EpiZooX training

```python
from epizoo.data import CellDatasetX, collate_fn_x

dataset = CellDatasetX(
    cell_sentences=cell_sentences,
    num_ccres=92_145,
)
```

### Joint-vocabulary training

```python
from epizoo.data import CellDatasetJoint, collate_fn_x

dataset = CellDatasetJoint(
    cell_sentences=cell_sentences,
    species=species,         # 0/ref or 1/new
    ref_to_joint=ref_to_joint,
    new_to_joint=new_to_joint,
)
```

### Cancer-context training

```python
from epizoo.data import CellDatasetCancer, collate_fn_cancer

dataset = CellDatasetCancer(
    cell_sentences=adata.obs["cell_indices"].values,
    species=species,
    cancer_type=adata.obs["condition_id"].values,
    human_num_ccres=700_460,
    mouse_num_ccres=814_020,
)
```

### EpiZooSeq training data

`SEAMDataset` can also be used for EpiZooSeq when signals are provided:

```python
from epizoo.data import SEAMDataset, collate_fn_seam

dataset = SEAMDataset(
    sequences=train_sequences,
    signals=train_signals,   # [num_sequences, num_cell_types]
    dnabert_path="/path/to/dnabert2",
    max_length=512,
)
```

## Training

### Base EpiZoo / EpiZooDI fine-tuning

```python
from epizoo.models import EpiZoo, EpiZooConfig
from epizoo.train import EpiZooFinetuneTrainer, FineTuneConfig

model = EpiZoo(EpiZooConfig())

trainer = EpiZooFinetuneTrainer(
    model=model,
    train_loader=train_loader,
    cfg=FineTuneConfig(
        mode="sr_cca",
        output_dir="checkpoints/epizoo",
        max_steps=500_000,
        save_steps=10_000,
        log_steps=500,
        freeze_seq_emb=True,
    ),
)

trainer.train()
```

### EpiZooX post-training

```python
from epizoo.train import EpiZooXPostTrainer, EpiZooXPostTrainConfig

trainer = EpiZooXPostTrainer(
    model=model,
    train_loader=train_loader,
    cfg=EpiZooXPostTrainConfig(
        mode="sr_cca",
        freeze_seq_emb=True,
    ),
)
```

### Cancer-context fine-tuning

```python
from epizoo.models import EpiZooCancer, EpiZooCancerConfig
from epizoo.train import EpiZooCancerTrainer, CancerTrainConfig

model = EpiZooCancer(EpiZooCancerConfig(num_cancer_types=8))

trainer = EpiZooCancerTrainer(
    model=model,
    train_loader=train_loader,
    cfg=CancerTrainConfig(mode="sr_cca"),
)
```

### EpiZooSeq training

```python
from epizoo.models import EpiZooSeq, EpiZooSeqConfig
from epizoo.train import EpiZooSeqTrainer, EpiZooSeqTrainConfig

model = EpiZooSeq(
    cell_type_emb=cell_type_embeddings,
    cfg=EpiZooSeqConfig(
        dnabert_path="/path/to/dnabert2",
    ),
)

trainer = EpiZooSeqTrainer(
    model=model,
    train_loader=train_loader,
    cfg=EpiZooSeqTrainConfig(
        output_dir="checkpoints/epizoo_seq",
        epochs=20,
        warmup_epochs=1,
    ),
)

trainer.train()
```

## Inference

### Extract cell embeddings

```python
from epizoo.inference import extract_cell_embeddings

cell_embeddings = extract_cell_embeddings(
    model=model,
    dataloader=inference_loader,
    device="cuda",
)
```

### Compute cell type embeddings

```python
from epizoo.inference import compute_cell_type_embeddings

cell_type_embeddings, cell_type_order = compute_cell_type_embeddings(
    cell_embeddings=cell_embeddings,
    labels=cell_type_labels,
    return_labels=True,
)
```

### Predict reconstructed signals

```python
from epizoo.inference import predict_signals

out = predict_signals(
    model=model,
    dataloader=inference_loader,
    apply_sigmoid=True,
)
```

### Predict sequence accessibility

```python
from epizoo.inference import predict_seq_accessibility

out = predict_seq_accessibility(
    model=epizoo_seq,
    dataloader=seq_loader,
)

preds = out["preds"]
targets = out.get("targets")
```

### Score SNV loss-of-accessibility with EpiZooCancer

```python
from epizoo.inference import score_mutation_loa

result = score_mutation_loa(
    model=cancer_model,
    seam_model=seam,
    dataloader=cancer_loader,
    ref_sequence=ref_seq,
    alt_sequence=alt_seq,
    mut_idx=mut_ccre_idx,      # 0-based cCRE index, no special-token offset
    dnabert_path="/path/to/dnabert2",
    device="cuda",
)

loa_score = result["score"]
```

### Run EpiZooSeq attribution

```python
from transformers import AutoTokenizer
from epizoo.inference import run_seq_attribution
from epizoo.visualization import plot_attribution_logo

tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/dnabert2",
    trust_remote_code=True,
)

result = run_seq_attribution(
    model=epizoo_seq,
    tokenizer=tokenizer,
    sequence=sequence,
    target_cell_type="SstChodl",
    cell_type_names=cell_type_names,
    device="cuda",
    n_steps=50,
    smooth_sigma=1.0,
)

plot_attribution_logo(
    sequence=result["sequence"],
    scores=result["scores"],
    title="Attribution",
    save_path="attribution_logo.pdf",
)
```

## Metrics and visualization

### CCA metrics

```python
from epizoo.metrics import compute_cca_metrics

metrics = compute_cca_metrics(labels=labels, logits=logits)
```

### Imputation correlations

```python
from epizoo.metrics import compute_imputation_correlations

corr = compute_imputation_correlations(
    adata=adata_true,
    adata_imputed=adata_imputed,
    cell_type_key="cell_type",
)
```

### EpiZooSeq correlations

```python
from epizoo.metrics import compute_seq_correlations

corr = compute_seq_correlations(preds=preds, targets=targets)
```

### Density scatter plot

```python
from epizoo.visualization import plot_density_scatter

plot_density_scatter(
    preds=preds,
    targets=targets,
    title="EpiZoo",
    save_path="scatter.pdf",
)
```

## Notes for GitHub release

This repository intentionally does not include:

- raw `.h5ad` datasets
- generated TF-IDF matrices
- generated SEAM sequence embeddings
- model checkpoints (`.pt`, `.pth`)
- genome FASTA files
- liftOver chain files
- DNABERT-2 pretrained weights

These should be stored externally and downloaded or generated as part of the user workflow.

## License

Please add a license before public release. Recommended options include MIT, Apache-2.0, or BSD-3-Clause, depending on your intended distribution policy.

## Citation

If you use EpiZoo in your work, please cite the corresponding manuscript once available.
