# epizoo/data/ccre.py

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Dict, Mapping
import pandas as pd


OnError = Literal["raise", "skip"]


def extract_dna_sequences(
    fasta_path: str,
    regions: Sequence[str],
    uppercase: bool = True,
    fix_chrom_name: bool = True,
    on_error: OnError = "raise",
    show_progress: bool = True,
) -> List[str]:
    """
    Extract DNA sequences from a FASTA file using cCRE region strings.

    Parameters
    ----------
    fasta_path:
        Path to genome FASTA file.

    regions:
        List of region strings in the format:
            chr:start-end

        Example:
            chr1:1000-1200

    uppercase:
        Whether to convert sequences to uppercase.

    fix_chrom_name:
        Whether to try simple chromosome name alternatives when the original
        chromosome name is not found.

        Examples:
            chr1 -> Chr1
            Chr1 -> chr1
            1    -> chr1 / Chr1

    on_error:
        Error handling mode.

        - "raise":
            Raise an error when a region cannot be parsed or extracted.
            This is safer because the returned sequence order always matches
            the input region order.

        - "skip":
            Print a warning and skip failed regions.

    show_progress:
        Whether to show a tqdm progress bar.

    Returns
    -------
    sequences:
        List of extracted DNA sequences.
    """

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise ImportError(
            "`pyfaidx` is required for extracting sequences from FASTA. "
            "Please install it with `pip install pyfaidx`."
        ) from exc

    fasta = Fasta(
        fasta_path,
        sequence_always_upper=uppercase,
    )

    sequences = []

    iterator = regions
    if show_progress:
        iterator = tqdm(regions, desc="Extracting DNA sequences")

    for region in iterator:
        try:
            chrom, start, end = parse_region(region)
            chrom = resolve_chrom_name(
                chrom=chrom,
                fasta=fasta,
                fix_chrom_name=fix_chrom_name,
            )

            seq = str(fasta[chrom][start:end])

            if uppercase:
                seq = seq.upper()

            sequences.append(seq)

        except Exception as exc:
            if on_error == "raise":
                raise ValueError(f"Failed to extract sequence for region `{region}`.") from exc

            if on_error == "skip":
                print(f"Warning: failed to extract sequence for region `{region}`: {exc}")
                continue

            raise ValueError("`on_error` should be either 'raise' or 'skip'.")

    return sequences


def parse_region(region: str) -> Tuple[str, int, int]:
    """
    Parse a region string.

    Expected format:
        chr:start-end
    """

    try:
        chrom, coords = region.split(":")
        start, end = coords.split("-")

        start = int(start)
        end = int(end)

    except Exception as exc:
        raise ValueError(
            f"Invalid region format: `{region}`. "
            "Expected format is `chr:start-end`."
        ) from exc

    if start < 0:
        raise ValueError(f"Region start should be non-negative: `{region}`.")

    if end <= start:
        raise ValueError(f"Region end should be greater than start: `{region}`.")

    return chrom, start, end


def resolve_chrom_name(
    chrom: str,
    fasta,
    fix_chrom_name: bool = True,
) -> str:
    """
    Resolve chromosome name against FASTA keys.

    This keeps the original logic:
        chr1 -> Chr1

    and also tries a few common alternatives.
    """

    if chrom in fasta:
        return chrom

    if not fix_chrom_name:
        raise KeyError(f"Chromosome `{chrom}` not found in FASTA.")

    candidates = []

    if chrom.startswith("chr"):
        candidates.append(chrom.replace("chr", "Chr", 1))
        candidates.append(chrom[3:])

    elif chrom.startswith("Chr"):
        candidates.append(chrom.replace("Chr", "chr", 1))
        candidates.append(chrom[3:])

    else:
        candidates.append(f"chr{chrom}")
        candidates.append(f"Chr{chrom}")

    for candidate in candidates:
        if candidate in fasta:
            return candidate

    raise KeyError(
        f"Chromosome `{chrom}` not found in FASTA. "
        f"Tried candidates: {candidates}"
    )


def build_ccre_map(
    new_bed: str,
    ref_bed: str,
    chain_file: str,
    liftover_bin: str = "liftOver",
    bedtools_bin: str = "bedtools",
    min_overlap: int = 1,
    tmp_dir: Optional[str] = None,
    keep_tmp: bool = False,
) -> Dict[int, int]:
    """
    Build cCRE index mapping from a new species to a reference species.

    The returned mapping is:
        {new_ccre_idx: ref_ccre_idx}

    Both indices are 0-based positions in their original cCRE BED files.
    They do not include special-token offset.

    Workflow:
        1. Read and sort new species cCRE BED.
        2. Add a temporary integer index to each new cCRE.
        3. Run liftOver from new species to reference species.
        4. Add a temporary integer index to each reference cCRE.
        5. Run bedtools intersect -wao.
        6. For each new cCRE, keep the reference cCRE with the largest overlap.

    Parameters
    ----------
    new_bed:
        BED file of new species cCREs.

    ref_bed:
        BED file of reference species cCREs.

    chain_file:
        liftOver chain file from new species to reference species.

    liftover_bin:
        Path or command name for UCSC liftOver.

    bedtools_bin:
        Path or command name for bedtools.

    min_overlap:
        Minimum overlap length required to keep a pair.

    tmp_dir:
        Optional temporary directory.

    keep_tmp:
        If True, keep intermediate files for debugging.

    Returns
    -------
    ccre_map:
        Dict mapping new cCRE index to reference cCRE index.
    """

    _check_file(new_bed, "new_bed")
    _check_file(ref_bed, "ref_bed")
    _check_file(chain_file, "chain_file")
    _check_tool(liftover_bin, "liftOver")
    _check_tool(bedtools_bin, "bedtools")

    work_dir_obj = tempfile.TemporaryDirectory(dir=tmp_dir)
    work_dir = Path(work_dir_obj.name)

    try:
        new_indexed = work_dir / "new.indexed.sorted.bed"
        ref_indexed = work_dir / "ref.indexed.sorted.bed"
        lifted_bed = work_dir / "new.lifted.bed"
        unmapped_bed = work_dir / "new.unmapped.bed"
        overlap_bed = work_dir / "new_ref.overlap.wao.bed"

        new_n = _write_indexed_bed(
            input_bed=new_bed,
            output_bed=new_indexed,
            index_name="new_idx",
        )

        ref_n = _write_indexed_bed(
            input_bed=ref_bed,
            output_bed=ref_indexed,
            index_name="ref_idx",
        )

        _run_liftover(
            liftover_bin=liftover_bin,
            input_bed=new_indexed,
            chain_file=chain_file,
            output_bed=lifted_bed,
            unmapped_bed=unmapped_bed,
        )

        _run_bedtools_intersect(
            bedtools_bin=bedtools_bin,
            a_bed=lifted_bed,
            b_bed=ref_indexed,
            output_bed=overlap_bed,
        )

        ccre_map = _parse_overlap_map(
            overlap_bed=overlap_bed,
            min_overlap=min_overlap,
        )

        print(
            f"Built cCRE map: {len(ccre_map)} matched pairs "
            f"from {new_n} new cCREs to {ref_n} reference cCREs."
        )

        return ccre_map

    finally:
        if keep_tmp:
            print(f"Intermediate files kept at: {work_dir}")
        else:
            work_dir_obj.cleanup()


def _write_indexed_bed(
    input_bed: str,
    output_bed: Path,
    index_name: str,
) -> int:
    """
    Read BED, sort by chrom/start, and write a 4-column indexed BED.

    Output format:
        chrom start end index
    """

    bed = _read_bed(input_bed)

    if bed.shape[1] < 3:
        raise ValueError(f"BED file should have at least 3 columns: {input_bed}")

    bed = bed.iloc[:, :3].copy()
    bed.columns = ["chrom", "start", "end"]

    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)

    bed[index_name] = range(len(bed))

    bed = bed.sort_values(
        by=["chrom", "start", "end"],
        kind="mergesort",
    )

    bed.to_csv(
        output_bed,
        sep="\t",
        header=False,
        index=False,
    )

    return bed.shape[0]


def _read_bed(path: str) -> pd.DataFrame:
    """
    Read BED file with no header.
    """

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        compression="infer",
    )


def _run_liftover(
    liftover_bin: str,
    input_bed: Path,
    chain_file: str,
    output_bed: Path,
    unmapped_bed: Path,
) -> None:
    """
    Run UCSC liftOver.
    """

    cmd = [
        liftover_bin,
        str(input_bed),
        chain_file,
        str(output_bed),
        str(unmapped_bed),
    ]

    _run_cmd(cmd)


def _run_bedtools_intersect(
    bedtools_bin: str,
    a_bed: Path,
    b_bed: Path,
    output_bed: Path,
) -> None:
    """
    Run bedtools intersect -wao.
    """

    cmd = [
        bedtools_bin,
        "intersect",
        "-a",
        str(a_bed),
        "-b",
        str(b_bed),
        "-wao",
    ]

    with open(output_bed, "w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )


def _parse_overlap_map(
    overlap_bed: Path,
    min_overlap: int = 1,
) -> Dict[int, int]:
    """
    Parse bedtools -wao output and keep the best reference cCRE per new cCRE.

    Expected A BED:
        chrom start end new_idx

    Expected B BED:
        chrom start end ref_idx

    bedtools -wao output:
        A(4 columns) + B(4 columns) + overlap_length
    """

    overlap = pd.read_csv(
        overlap_bed,
        sep="\t",
        header=None,
    )

    if overlap.empty:
        return {}

    # Columns:
    # 0,1,2,3 = lifted new BED
    # 4,5,6,7 = reference BED
    # 8       = overlap length
    new_idx_col = 3
    ref_idx_col = 7
    overlap_col = 8

    overlap = overlap.rename(
        columns={
            new_idx_col: "new_idx",
            ref_idx_col: "ref_idx",
            overlap_col: "overlap",
        }
    )

    overlap = overlap[overlap["overlap"] >= min_overlap].copy()

    if overlap.empty:
        return {}

    overlap["new_idx"] = overlap["new_idx"].astype(int)
    overlap["ref_idx"] = overlap["ref_idx"].astype(int)
    overlap["overlap"] = overlap["overlap"].astype(int)

    # Keep the reference cCRE with the largest overlap for each new cCRE.
    overlap = overlap.sort_values(
        by=["new_idx", "overlap"],
        ascending=[True, False],
        kind="mergesort",
    )

    best = overlap.drop_duplicates(
        subset="new_idx",
        keep="first",
    )

    return dict(
        zip(
            best["new_idx"].astype(int),
            best["ref_idx"].astype(int),
        )
    )


def _run_cmd(cmd) -> None:
    """
    Run shell command safely.
    """

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )


def _check_file(path: str, name: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"`{name}` not found: {path}")


def _check_tool(tool: str, name: str) -> None:
    """
    Check whether an executable exists.

    Supports both:
        - executable name in PATH
        - explicit executable path
    """

    if os.path.isfile(tool) and os.access(tool, os.X_OK):
        return

    if shutil.which(tool) is not None:
        return

    raise FileNotFoundError(
        f"`{name}` executable not found: {tool}. "
        f"Please install it or provide the full path."
    )


def build_joint_ccre_maps(
    ref_num_ccres: int,
    new_num_ccres: int,
    ccre_map: Mapping[int, int],
    ccre_offset: int = 4,
    return_token_ids: bool = True,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build joint vocabulary maps for two species.

    The input cCRE map should be:

        {
            new_idx: ref_idx
        }

    where both indices are 0-based cCRE indices in their own original cCRE lists.
    They should not include special-token offset.

    Joint vocabulary order:
        1. overlap cCREs
        2. reference-only cCREs
        3. new-only cCREs

    Parameters
    ----------
    ref_num_ccres:
        Number of cCREs in the reference species.

    new_num_ccres:
        Number of cCREs in the new species.

    ccre_map:
        Mapping from new species cCRE index to reference species cCRE index.

    ccre_offset:
        Number of reserved special tokens.
        Default: 4.

    return_token_ids:
        If True, returned joint ids include `ccre_offset`.
        If False, returned ids are 0-based cCRE indices in the joint vocabulary.

    Returns
    -------
    ref_to_joint:
        Mapping from reference species original cCRE index to joint vocabulary id.

    new_to_joint:
        Mapping from new species original cCRE index to joint vocabulary id.
    """

    _check_num_ccres(ref_num_ccres, "ref_num_ccres")
    _check_num_ccres(new_num_ccres, "new_num_ccres")

    ccre_map = _normalize_ccre_map(
        ccre_map=ccre_map,
        ref_num_ccres=ref_num_ccres,
        new_num_ccres=new_num_ccres,
    )

    new_by_ref = defaultdict(list)
    for new_idx, ref_idx in ccre_map.items():
        new_by_ref[ref_idx].append(new_idx)

    ref_to_joint = {}
    new_to_joint = {}

    joint_idx = 0

    # 1. Overlap cCREs.
    for ref_idx in sorted(new_by_ref):
        ref_to_joint[ref_idx] = joint_idx

        for new_idx in sorted(new_by_ref[ref_idx]):
            new_to_joint[new_idx] = joint_idx

        joint_idx += 1

    # 2. Reference-only cCREs.
    for ref_idx in range(ref_num_ccres):
        if ref_idx not in ref_to_joint:
            ref_to_joint[ref_idx] = joint_idx
            joint_idx += 1

    # 3. New-only cCREs.
    for new_idx in range(new_num_ccres):
        if new_idx not in new_to_joint:
            new_to_joint[new_idx] = joint_idx
            joint_idx += 1

    if return_token_ids:
        ref_to_joint = {
            idx: joint_id + ccre_offset
            for idx, joint_id in ref_to_joint.items()
        }

        new_to_joint = {
            idx: joint_id + ccre_offset
            for idx, joint_id in new_to_joint.items()
        }

    return ref_to_joint, new_to_joint


def get_joint_ccre_count(
    ref_to_joint: Mapping[int, int],
    new_to_joint: Mapping[int, int],
    ccre_offset: int = 4,
    ids_include_offset: bool = True,
) -> int:
    """
    Get the number of cCREs in a joint vocabulary from joint maps.
    """

    joint_ids = list(ref_to_joint.values()) + list(new_to_joint.values())

    if len(joint_ids) == 0:
        return 0

    max_id = max(joint_ids)

    if ids_include_offset:
        return max_id - ccre_offset + 1

    return max_id + 1


def _normalize_ccre_map(
    ccre_map: Mapping[int, int],
    ref_num_ccres: int,
    new_num_ccres: int,
) -> Dict[int, int]:
    """
    Normalize and validate {new_idx: ref_idx} map.
    """

    out = {}

    for new_idx, ref_idx in ccre_map.items():
        new_idx = int(new_idx)
        ref_idx = int(ref_idx)

        if new_idx < 0 or new_idx >= new_num_ccres:
            raise IndexError(
                f"new_idx out of range: {new_idx}. "
                f"Expected [0, {new_num_ccres})."
            )

        if ref_idx < 0 or ref_idx >= ref_num_ccres:
            raise IndexError(
                f"ref_idx out of range: {ref_idx}. "
                f"Expected [0, {ref_num_ccres})."
            )

        out[new_idx] = ref_idx

    return out


def _check_num_ccres(
    value: int,
    name: str,
) -> None:
    if int(value) <= 0:
        raise ValueError(f"`{name}` should be a positive integer.")