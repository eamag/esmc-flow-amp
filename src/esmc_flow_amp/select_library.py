"""Greedy length-stratified selection of the 50,000-sequence library (config o2k0d1p97)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

from esmc_flow_amp.constants import (
    DBAASP_FASTA_PATH,
    GENERATED_LIBRARY_PATH,
    STANDARD_AMINO_ACIDS_SET,
    ShippedLibraryConfig,
)
from esmc_flow_amp.fasta_io import read_fasta_sequences, write_fasta_sequences


def standardize(x: np.ndarray) -> np.ndarray:
    """Compute standard score z = (x - mean) / std."""
    std = float(x.std())
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / std


def compute_length_allocations(reference_sequences: list[str], target_total: int) -> dict[int, int]:
    """Allocate exact integer quota per sequence length matching reference distribution."""
    valid_lengths = [len(s) for s in reference_sequences if 8 <= len(s) <= 50]
    total_valid = len(valid_lengths)
    counts = Counter(valid_lengths)

    exact = {l: target_total * counts.get(l, 0) / total_valid for l in range(8, 51)}
    alloc = {l: int(exact[l]) for l in exact}
    shortfall = target_total - sum(alloc.values())

    remainders = sorted(
        ((exact[l] - alloc[l], l) for l in exact),
        reverse=True,
    )
    for _, l in remainders[:shortfall]:
        alloc[l] += 1

    assert sum(alloc.values()) == target_total
    return alloc


def select_greedy_candidates(
    pool_sequences: list[str],
    lengths: np.ndarray,
    scores: np.ndarray,
    allocations: dict[int, int],
    target_size: int,
    cutoff: float = 90.0,
) -> list[int]:
    """Greedy selection stratified by length bin with near-duplicate elimination."""
    chosen_indices: list[int] = []

    for l in range(8, 51):
        quota = allocations.get(l, 0)
        if quota <= 0:
            continue
        bin_indices = np.where(lengths == l)[0]
        if len(bin_indices) == 0:
            continue
        sorted_bin_order = bin_indices[np.argsort(-scores[bin_indices])]

        # Progressive candidate windows for rapidfuzz deduplication
        for window_size in (
            max(6 * quota, quota + 2000),
            max(20 * quota, quota + 8000),
            len(sorted_bin_order),
        ):
            candidates = sorted_bin_order[:window_size]
            candidate_seqs = [pool_sequences[i] for i in candidates]
            dist_matrix = rf_process.cdist(
                candidate_seqs,
                candidate_seqs,
                scorer=fuzz.ratio,
                workers=-1,
                score_cutoff=cutoff,
                dtype=np.uint8,
            )
            dropped = np.zeros(len(candidates), dtype=bool)
            accumulated: list[int] = []
            for i in range(len(candidates)):
                if dropped[i]:
                    continue
                accumulated.append(candidates[i])
                if len(accumulated) == quota:
                    break
                dropped[i + 1 :] |= dist_matrix[i, i + 1 :] > 0

            if len(accumulated) == quota or window_size >= len(sorted_bin_order):
                break

        chosen_indices.extend(accumulated)

    # Top up shortfall if any bin lacked enough candidates
    if len(chosen_indices) < target_size:
        used_set = set(chosen_indices)
        top_up = [i for i in np.argsort(-scores) if i not in used_set][
            : target_size - len(chosen_indices)
        ]
        chosen_indices.extend(top_up)

    return chosen_indices[:target_size]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select final 50,000-sequence library from candidate pool."
    )
    parser.add_argument(
        "--features-dir",
        default="data/features",
        help="Directory containing pool_features.npz and prec20.npy.",
    )
    parser.add_argument(
        "--ref-fasta",
        default=str(DBAASP_FASTA_PATH),
        help="Path to DBAASP reference FASTA.",
    )
    parser.add_argument(
        "--out",
        default=str(GENERATED_LIBRARY_PATH),
        help="Target output path for selected library FASTA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ShippedLibraryConfig()
    features_dir = Path(args.features_dir)

    feat_path = features_dir / "pool_features.npz"
    if not feat_path.exists():
        feat_path = features_dir / "pool2.npz"
    prec_path = features_dir / "prec20.npy"

    data = np.load(feat_path)
    pool_seqs = data["seqs"].tolist()
    lengths = data["lens"]
    f1 = data["f1"].astype(float)
    mic = data["mic"]
    prec20 = np.load(prec_path)

    active = (mic <= 16.0).astype(float)

    z_conf = standardize(data["conf"])
    z_lw = standardize(data["lw"])
    z_div = standardize(data["div"])
    z_prec = standardize(prec20)

    # Multi-term selection score
    score = (
        config.omega * z_conf
        + config.active_weight * active
        + config.synthesizability_weight * f1
        + config.eta * z_lw
        + config.kappa * z_prec
        + config.delta * z_div
    )
    # Apply hard Precision filter threshold >= 0.97
    penalized_scores = np.where(prec20 >= config.precision_threshold, score, score - 1000.0)

    # Load DBAASP reference to calculate length histogram quota
    ref_all = read_fasta_sequences(Path(args.ref_fasta))
    ref_clean = [s for s in ref_all if set(s) <= STANDARD_AMINO_ACIDS_SET and 8 <= len(s) <= 50]
    allocations = compute_length_allocations(ref_clean, config.target_size)

    selected_indices = select_greedy_candidates(
        pool_seqs,
        lengths,
        penalized_scores,
        allocations,
        config.target_size,
        cutoff=config.near_duplicate_cutoff,
    )

    out_path = Path(args.out)
    selected_sequences = [pool_seqs[i] for i in selected_indices]
    write_fasta_sequences(
        selected_sequences,
        out_path,
        header_prefix=f"deep4_{config.name}",
    )
    print(
        f"Selected {len(selected_sequences):,} sequences (config {config.name}). Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
