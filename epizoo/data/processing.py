# epizoo/data/processing.py

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


def compute_tfidf(
    adata,
    df: Optional[np.ndarray] = None,
    cell_number: Optional[int] = None,
    scale_factor: float = 10_000.0,
    dtype=np.float32,
    store: Literal["X", "obsm"] = "X",
    obsm_key: str = "X_tfidf",
    verbose: bool = True,
):
    """
    Compute TF-IDF normalization for a cell-by-cCRE count AnnData.

    Parameters
    ----------
    adata:
        AnnData object with raw count matrix stored in `adata.X`.

    df:
        Document frequency for each cCRE.

        If provided, `df[j]` should be the number of cells in which cCRE j is accessible.
        Its order must match `adata.var_names`.

        If None, df is computed from `adata.X` as:
            df[j] = number of cells with X[:, j] > 0

    cell_number:
        Total number of cells used to compute `df`.

        If `df` is None, this is set to `adata.n_obs`.
        If `df` is provided but `cell_number` is None, this is also set to `adata.n_obs`.

    scale_factor:
        Scaling factor applied after TF-IDF normalization.

    dtype:
        Output dtype.

    store:
        Where to store the TF-IDF matrix.
        - "X": replace `adata.X` with TF-IDF.
        - "obsm": store TF-IDF in `adata.obsm[obsm_key]`.

    obsm_key:
        Key used when `store="obsm"`.

    Returns
    -------
    tfidf_adata:
        A copied AnnData object with TF-IDF stored in the selected location.

    Formula
    -------
        TF      = count / total_count_per_cell
        IDF     = log(1 + cell_number / (df + 1))
        TF-IDF  = TF * IDF * scale_factor
    """

    if store not in {"X", "obsm"}:
        raise ValueError("`store` must be either 'X' or 'obsm'.")

    tfidf_adata = adata.copy()

    if df is None:
        df = compute_document_frequency(adata.X, dtype=dtype)
        cell_number = adata.n_obs
    else:
        df = np.asarray(df, dtype=dtype).reshape(-1)
        if cell_number is None:
            raise ValueError("`cell_number` must be provided when `df` is given.")

    if df.size != adata.n_vars:
        raise ValueError(
            "`df` length must match the number of cCREs in `adata`. "
            f"Got len(df)={df.size}, adata.n_vars={adata.n_vars}."
        )

    if cell_number <= 0:
        raise ValueError("`cell_number` must be a positive integer.")

    idf = compute_idf(
        df=df,
        cell_number=cell_number,
        dtype=dtype,
    )

    tfidf = compute_tfidf_matrix(
        x=adata.X,
        idf=idf,
        scale_factor=scale_factor,
        dtype=dtype,
    ).tocsr()

    if store == "X":
        tfidf_adata.X = tfidf
    else:
        tfidf_adata.obsm[obsm_key] = tfidf

    if verbose:
        if store == "X":
            print("TF-IDF completed. TF-IDF matrix stored in adata.X")
        else:
            print(f"TF-IDF completed. TF-IDF matrix stored in adata.obsm['{obsm_key}']")
        
        print("=" * 50)
        print(f"Matrix shape: {tfidf.shape}")
        print(f"Matrix type: {type(tfidf)}")
        print(f"Data type: {tfidf.dtype}")

        if sp.issparse(tfidf):
            nnz = tfidf.nnz
            total = tfidf.shape[0] * tfidf.shape[1]
            values = tfidf.data
            features_per_cell = np.asarray((tfidf > 0).sum(axis=1)).flatten()

            print(f"Non-zero entries: {nnz:,}")
            print(f"Sparsity: {1 - nnz / total:.4%}")
            print(f"Non-zero value min: {values.min():.6f}")
            print(f"Non-zero value max: {values.max():.6f}")
            print(f"Non-zero value mean: {values.mean():.6f}")
            print(f"Non-zero value median: {np.median(values):.6f}")

        else:
            nnz = np.count_nonzero(tfidf)
            total = tfidf.size
            values = tfidf[tfidf > 0]

            print(f"Non-zero entries: {nnz:,}")
            print(f"Sparsity: {1 - nnz / total:.4%}")
            print(f"Non-zero value min: {values.min():.6f}")
            print(f"Non-zero value max: {values.max():.6f}")
            print(f"Non-zero value mean: {values.mean():.6f}")
            print(f"Non-zero value median: {np.median(values):.6f}")

            features_per_cell = np.sum(tfidf > 0,axis=1)

        print("-" * 50)
        print(f"Accessible cCREs per cell:")
        print(f"  Mean: {features_per_cell.mean():.2f}")
        print(f"  Median: {np.median(features_per_cell):.2f}")
        print(f"  Min: {features_per_cell.min()}")
        print(f"  Max: {features_per_cell.max()}")
        print("=" * 50)

    return tfidf_adata


def compute_document_frequency(
    x,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute document frequency for each cCRE.

    df[j] = number of cells with X[:, j] > 0
    """

    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(x)
    except ImportError:
        is_sparse = False

    if is_sparse:
        x = x.copy()
        x.eliminate_zeros()
        df = np.asarray(x.getnnz(axis=0)).reshape(-1)
    else:
        x = np.asarray(x)
        df = (x > 0).sum(axis=0)

    return df.astype(dtype, copy=False)


def compute_idf(
    df: np.ndarray,
    cell_number: int,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute inverse document frequency.

    IDF = log(1 + cell_number / (df + 1))
    """

    df = np.asarray(df, dtype=dtype).reshape(-1)

    idf = np.log1p(
        cell_number / (df + 1.0)
    )

    return idf.astype(dtype, copy=False)


def compute_tfidf_matrix(
    x,
    idf: np.ndarray,
    scale_factor: float,
    dtype=np.float32,
):
    """
    Compute TF-IDF matrix from count matrix and IDF vector.

    Supports dense and sparse matrices.
    """

    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(x)
    except ImportError:
        is_sparse = False

    if is_sparse:
        x = x.astype(dtype).tocsr()

        cell_sums = np.asarray(x.sum(axis=1)).reshape(-1).astype(dtype)

        inv_cell_sums = np.zeros_like(cell_sums, dtype=dtype)
        np.divide(
            1.0,
            cell_sums,
            out=inv_cell_sums,
            where=cell_sums > 0,
        )

        tf = x.multiply(inv_cell_sums[:, None])
        tfidf = tf.multiply(idf.reshape(1, -1)).multiply(scale_factor)

        return tfidf.astype(dtype)

    x = np.asarray(x, dtype=dtype)

    cell_sums = x.sum(axis=1, keepdims=True)

    tf = np.divide(
        x,
        cell_sums,
        out=np.zeros_like(x, dtype=dtype),
        where=cell_sums > 0,
    )

    tfidf = tf * idf.reshape(1, -1) * scale_factor

    return tfidf.astype(dtype, copy=False)


def generate_cell_sentences(
    adata,
    matrix_key: str = "X",
    obs_key: str = "cell_indices",
    species: Optional[int] = None,
    base_offset: int = 4,
    species_offset: int = 0,
):
    """
    Generate cell sentences from a TF-IDF AnnData.

    For each cell:
        1. find nonzero cCREs
        2. sort them by TF-IDF value in descending order
        3. add token offset
        4. store the resulting cCRE token ids in `adata.obs[obs_key]`

    Parameters
    ----------
    adata:
        AnnData object after TF-IDF normalization.

    matrix_key:
        Where to read the TF-IDF matrix.
        - "X": use `adata.X`
        - otherwise: use `adata.obsm[matrix_key]`

    obs_key:
        Key in `adata.obs` used to store generated cell sentences.

    species:
        Optional species id.
        - None: do not write species column and do not add species_offset
        - 0: human, write species=0 and do not add species_offset
        - 1: mouse, write species=1 and add species_offset

    base_offset:
        Offset reserved for special tokens.
        Default: 4.

    species_offset:
        Extra offset for mouse cCREs.
        Only applied when `species == 1`.

    Returns
    -------
    out:
        A copied AnnData object with generated cell sentences stored in `.obs`.
    """

    if species not in {None, 0, 1}:
        raise ValueError("`species` should be None, 0, or 1.")

    out = adata.copy()
    x = _get_matrix(out, matrix_key)

    token_offset = base_offset
    if species == 1:
        token_offset += species_offset

    cell_sentences = _build_cell_sentences(
        x=x,
        token_offset=token_offset,
    )

    out.obs[obs_key] = pd.Series(
        cell_sentences,
        index=out.obs_names,
        dtype="object",
    )

    if species is not None:
        out.obs["species"] = species

    return out


def _build_cell_sentences(
    x,
    token_offset: int,
):
    """
    Build sorted cCRE token ids for all cells.
    """

    if _is_sparse(x):
        x = x.tocsr()
        return [
            _build_sparse_cell_sentence(x.getrow(i), token_offset)
            for i in range(x.shape[0])
        ]

    x = np.asarray(x)
    return [
        _build_dense_cell_sentence(x[i], token_offset)
        for i in range(x.shape[0])
    ]


def _build_dense_cell_sentence(
    row: np.ndarray,
    token_offset: int,
):
    """
    Build one cell sentence from a dense TF-IDF row.
    """

    nonzero = np.flatnonzero(row)

    if len(nonzero) == 0:
        return []

    sorted_idx = nonzero[np.argsort(-row[nonzero])]

    return (sorted_idx + token_offset).astype(int).tolist()


def _build_sparse_cell_sentence(
    row,
    token_offset: int,
):
    """
    Build one cell sentence from a sparse TF-IDF row.
    """

    row = row.tocsr()
    indices = row.indices
    values = row.data

    if len(indices) == 0:
        return []

    sorted_idx = indices[np.argsort(-values)]

    return (sorted_idx + token_offset).astype(int).tolist()


def _get_matrix(
    adata,
    matrix_key: str,
):
    """
    Get matrix from adata.X or adata.obsm.
    """

    if matrix_key == "X":
        return adata.X

    if matrix_key not in adata.obsm:
        raise KeyError(f"`{matrix_key}` not found in `adata.obsm`.")

    return adata.obsm[matrix_key]


def _is_sparse(x) -> bool:
    """
    Check whether a matrix is scipy sparse.
    """

    try:
        import scipy.sparse as sp

        return sp.issparse(x)
    except ImportError:
        return False
    

def filter_cCREs(adata, filter_idx, species, verbose=True):
    """
    Filter cCRE features in an AnnData object using a predefined cCRE selection index.

    This function removes low-quality or uninformative cCREs according to the
    species-specific filtering strategy used in EpiZoo. The input AnnData object
    is expected to contain cells as rows and cCREs as columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Input single-cell chromatin accessibility dataset.
        The feature dimension (adata.n_vars) should correspond to the full
        species-specific cCRE vocabulary.

    filter_idx : array-like
        Boolean index or integer index array specifying the retained cCREs after
        filtering.

    species : int or None
        Species identifier:
        - 0: human
        - 1: mouse
        - None: skip species-specific length validation

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object containing only the selected cCRE features.

    Raises
    ------
    ValueError
        If `species` is not None, 0, or 1.
    AssertionError
        If the length of `filter_idx` does not match the expected cCRE vocabulary
        size for the specified species.

    Notes
    -----
    The expected cCRE vocabulary sizes are:
    - Human: 700,460 cCREs
    - Mouse: 814,020 cCREs
    """

    if species not in {None, 0, 1}:
        raise ValueError(
            "`species` should be None, 0 (human), or 1 (mouse)."
        )

    # Validate species-specific cCRE vocabulary size
    if species == 0:
        assert len(filter_idx) == 700460, (
            "Invalid human cCRE filter index: "
            "expected length 700,460."
        )

    elif species == 1:
        assert len(filter_idx) == 814020, (
            "Invalid mouse cCRE filter index: "
            "expected length 814,020."
        )

    # Subset cCRE features
    adata = adata[:, filter_idx].copy()

    if verbose:
        print(f"Filtered cCREs: {adata.n_vars} features retained.")

    return adata