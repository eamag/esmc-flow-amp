"""Rank top-100 candidates from 50k library with TabPFN oracle and hemolysis safety gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

from esmc_flow_amp.constants import (
    ANTIBACTERIAL_FASTA_PATH,
    DBAASP_FASTA_PATH,
    GENERATED_LIBRARY_PATH,
    GENERATED_TOP_PATH,
    STANDARD_AMINO_ACIDS_SET,
    ShippedTop100Config,
)
from esmc_flow_amp.fasta_io import read_fasta_sequences, write_fasta_sequences
from esmc_flow_amp.filters import is_synthetically_viable
from esmc_flow_amp.score_pool import load_ampredictor, predict_mic
from esmc_flow_amp.select_library import compute_length_allocations
from esmc_flow_amp.tabpfn_score import score_sequences

AMPREDICTOR_CHUNK = 4096


def ampredictor_mic(sequences: list[str], cache_path: Path, vendor_dir: Path) -> np.ndarray:
    """AMPredictor MIC per sequence: read from the pool scores when present, otherwise computed."""
    known: dict[str, float] = {}
    if cache_path.exists():
        cached = np.load(cache_path)
        known.update(zip(cached["seq"].tolist(), cached["mic"].tolist()))
    missing = [s for s in sequences if s not in known]
    if missing:
        print(f"Scoring {len(missing):,} sequences missing from {cache_path} with AMPredictor")
        model = load_ampredictor(vendor_dir)
        for start in range(0, len(missing), AMPREDICTOR_CHUNK):
            chunk = missing[start : start + AMPREDICTOR_CHUNK]
            known.update(zip(chunk, predict_mic(model, chunk).tolist()))
    return np.array([known[s] for s in sequences], dtype=float)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank top-100 candidate peptides from library.")
    parser.add_argument("--library", default=str(GENERATED_LIBRARY_PATH))
    parser.add_argument("--out", default=str(GENERATED_TOP_PATH))
    parser.add_argument("--hemo", type=float, default=0.35, help="Hemolysis probability threshold.")
    parser.add_argument("--shortlist", type=int, default=8000, help="Shortlist size for TabPFN.")
    parser.add_argument("--amp-gate", type=float, default=16.0, help="Max AMPredictor MIC (uM).")
    parser.add_argument(
        "--activity", default="data/pool_activity.npz", help="AMPredictor scores of the pool."
    )
    parser.add_argument("--vendor-dir", default="vendor/ampredictor")
    parser.add_argument(
        "--antibacterial-fasta",
        default=str(ANTIBACTERIAL_FASTA_PATH),
    )
    parser.add_argument(
        "--dbaasp-fasta",
        default=str(DBAASP_FASTA_PATH),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ShippedTop100Config(
        shortlist_size=args.shortlist,
        hemolysis_gate=args.hemo,
        ampredictor_gate=args.amp_gate,
    )

    lib_path = Path(args.library)
    library_seqs = read_fasta_sequences(lib_path)
    print(f"Loaded library with {len(library_seqs):,} sequences from {lib_path}")

    amp_scores = ampredictor_mic(library_seqs, Path(args.activity), Path(args.vendor_dir))

    # 1. Shortlist pre-filter: Synthesizability, AMPredictor gate, Novelty bound
    f1_mask = np.array([is_synthetically_viable(s) for s in library_seqs], dtype=bool)
    initial_order = [
        i for i in np.argsort(amp_scores) if f1_mask[i] and amp_scores[i] <= config.ampredictor_gate
    ][: 2 * config.shortlist_size]

    ref_seqs = read_fasta_sequences(Path(args.antibacterial_fasta))
    candidate_subset = [library_seqs[i] for i in initial_order]
    max_sim = (
        rf_process.cdist(
            candidate_subset,
            ref_seqs,
            scorer=fuzz.ratio,
            workers=-1,
            dtype=np.uint8,
        ).max(axis=1)
        / 100.0
    )

    shortlist = [
        library_seqs[i]
        for i, sim_val in zip(initial_order, max_sim)
        if sim_val < config.novelty_cutoff
    ][: config.shortlist_size]

    print(
        f"Filtered shortlist to {len(shortlist):,} candidates "
        f"(F1 pass, AMPredictor <= {config.ampredictor_gate}, similarity < {config.novelty_cutoff})"
    )

    # 2. TabPFN scoring and hemolysis gating
    scored_df = score_sequences(shortlist)
    gated_df = scored_df[scored_df["p_hemo"] < config.hemolysis_gate].copy()
    gated_df = gated_df.sort_values("mic_log10").reset_index(drop=True)
    gated_df["len"] = [len(s) for s in gated_df["sequence"]]

    print(f"Candidates passing hemolysis gate (< {config.hemolysis_gate}): {len(gated_df):,}")

    # 3. Stratified selection across length histogram
    dba_all = read_fasta_sequences(Path(args.dbaasp_fasta))
    dba_clean = [s for s in dba_all if set(s) <= STANDARD_AMINO_ACIDS_SET and 8 <= len(s) <= 50]
    quota = compute_length_allocations(dba_clean, config.target_count)

    picked_indices: list[int] = []

    def can_add(candidate_idx: int) -> bool:
        seq = gated_df["sequence"][candidate_idx]
        for p_idx in picked_indices:
            if fuzz.ratio(seq, gated_df["sequence"][p_idx]) >= config.near_duplicate_cutoff:
                return False
        picked_indices.append(candidate_idx)
        return True

    for length_val, target_q in sorted(quota.items()):
        got = 0
        bin_candidates = gated_df.index[gated_df["len"] == length_val]
        for idx in bin_candidates:
            if got == target_q:
                break
            if can_add(idx):
                got += 1

    # Fill remainder if quota left
    for idx in gated_df.index:
        if len(picked_indices) >= config.target_count:
            break
        if idx not in picked_indices:
            can_add(idx)

    final_top_seqs = [gated_df["sequence"][i] for i in picked_indices[: config.target_count]]
    out_path = Path(args.out)
    write_fasta_sequences(final_top_seqs, out_path, header_prefix="deep_top")
    print(f"Selected {len(final_top_seqs)} top candidates. Wrote {out_path}")


if __name__ == "__main__":
    main()
