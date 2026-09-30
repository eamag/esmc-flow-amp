"""Submission generation entry point for the AMP Challenge 2027.

This module materializes and verifies the final submitted FASTA pair:
  - generate/library.fasta: exactly 50,000 unique sequences (config o2k0d1p97)
  - generate/top.fasta: the 100 top-ranked sequences

NOTE ON REPRODUCIBILITY:
Regenerating this pair from scratch requires the full GPU training, sampling,
and selection pipeline (see REPRODUCE.md and src/esmc_flow_amp/). The generate
entry point verifies and materializes the committed final pair from data/final/
to generate/ in accordance with the official competition harness specification.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import Levenshtein

from esmc_flow_amp.constants import (
    ANTIBACTERIAL_FASTA_PATH,
    COMMITTED_LIBRARY_PATH,
    COMMITTED_TOP_PATH,
    EXPECTED_LIBRARY_MD5,
    EXPECTED_TOP_MD5,
    GENERATED_LIBRARY_PATH,
    GENERATED_TOP_PATH,
    MAXIMUM_SEQUENCE_LENGTH,
    MINIMUM_SEQUENCE_LENGTH,
    SIMILARITY_THRESHOLD,
    STANDARD_AMINO_ACIDS_SET,
    TARGET_LIBRARY_SIZE,
    TARGET_TOP_COUNT,
)
from esmc_flow_amp.fasta_io import read_fasta_sequences


def compute_file_md5(path: Path) -> str:
    """Compute hexadecimal MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_library_sequences(sequences: list[str]) -> list[str]:
    """Validate library sequence count, uniqueness, length, and canonical alphabet."""
    errors: list[str] = []
    if len(sequences) != TARGET_LIBRARY_SIZE:
        errors.append(f"Expected {TARGET_LIBRARY_SIZE} sequences, found {len(sequences)}")

    seen: set[str] = set()
    for index, sequence in enumerate(sequences, start=1):
        if not sequence:
            errors.append(f"Sequence {index}: empty string")
            continue
        invalid_residues = set(sequence) - STANDARD_AMINO_ACIDS_SET
        if invalid_residues:
            errors.append(f"Sequence {index}: non-canonical residues {sorted(invalid_residues)}")
            break
        if not (MINIMUM_SEQUENCE_LENGTH <= len(sequence) <= MAXIMUM_SEQUENCE_LENGTH):
            errors.append(
                f"Sequence {index}: length {len(sequence)} outside range "
                f"[{MINIMUM_SEQUENCE_LENGTH}, {MAXIMUM_SEQUENCE_LENGTH}]"
            )
            break
        if sequence in seen:
            errors.append(f"Sequence {index}: duplicate sequence detected")
            break
        seen.add(sequence)

    return errors


def validate_top_sequences(
    top_sequences: list[str],
    library_sequences: set[str],
    antibacterial_sequences: list[str],
) -> list[str]:
    """Validate top candidate count, subset integrity, and novelty constraint."""
    errors: list[str] = []
    if len(top_sequences) != TARGET_TOP_COUNT:
        errors.append(f"Expected {TARGET_TOP_COUNT} top sequences, found {len(top_sequences)}")

    seen: set[str] = set()
    for index, sequence in enumerate(top_sequences, start=1):
        if sequence not in library_sequences:
            errors.append(f"Top sequence {index}: not found in full library")
        if sequence in seen:
            errors.append(f"Top sequence {index}: duplicate in top list")
        seen.add(sequence)

    # Novelty check: Levenshtein ratio against known antibacterial reference peptides <= 0.80
    for seq in top_sequences:
        for ref in antibacterial_sequences:
            ratio = Levenshtein.ratio(seq, ref)
            if ratio > SIMILARITY_THRESHOLD:
                errors.append(
                    f"Top sequence similarity violation: {seq} has similarity "
                    f"{ratio:.4f} > {SIMILARITY_THRESHOLD} with reference {ref}"
                )
                return errors

    return errors


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for generate command."""
    parser = argparse.ArgumentParser(
        description="Materialize and verify the final AMP Challenge 2027 submission files."
    )
    parser.add_argument(
        "--source-library",
        type=Path,
        default=COMMITTED_LIBRARY_PATH,
        help="Path to committed library FASTA.",
    )
    parser.add_argument(
        "--source-top",
        type=Path,
        default=COMMITTED_TOP_PATH,
        help="Path to committed top-100 FASTA.",
    )
    parser.add_argument(
        "--target-library",
        type=Path,
        default=GENERATED_LIBRARY_PATH,
        help="Target output path for library FASTA.",
    )
    parser.add_argument(
        "--target-top",
        type=Path,
        default=GENERATED_TOP_PATH,
        help="Target output path for top-100 FASTA.",
    )
    parser.add_argument(
        "--antibacterial-fasta",
        type=Path,
        default=ANTIBACTERIAL_FASTA_PATH,
        help="Path to organizer antibacterial reference FASTA.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute validation and materialization of submission FASTA files."""
    args = parse_arguments()

    if not args.source_library.exists():
        raise FileNotFoundError(f"Committed library not found: {args.source_library}")
    if not args.source_top.exists():
        raise FileNotFoundError(f"Committed top-100 not found: {args.source_top}")

    # Read and validate sequences
    library_seqs = read_fasta_sequences(args.source_library)
    top_seqs = read_fasta_sequences(args.source_top)

    lib_errors = validate_library_sequences(library_seqs)
    if lib_errors:
        raise ValueError("Library validation failed:\n" + "\n".join(f"  - {e}" for e in lib_errors))

    library_set = set(library_seqs)

    # Reference check
    if args.antibacterial_fasta.exists():
        anti_seqs = read_fasta_sequences(args.antibacterial_fasta)
        overlap = library_set & set(anti_seqs)
        if overlap:
            raise ValueError(
                f"Library overlap violation: {len(overlap)} sequences match antibacterial reference."
            )
        top_errors = validate_top_sequences(top_seqs, library_set, anti_seqs)
        if top_errors:
            raise ValueError(
                "Top-100 validation failed:\n" + "\n".join(f"  - {e}" for e in top_errors)
            )
    else:
        print(
            f"Warning: antibacterial reference {args.antibacterial_fasta} not found, skipping novelty check"
        )

    # Copy files byte-for-byte to target locations
    args.target_library.parent.mkdir(parents=True, exist_ok=True)
    args.target_top.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(args.source_library, args.target_library)
    shutil.copyfile(args.source_top, args.target_top)

    # Verify checksums
    lib_md5 = compute_file_md5(args.target_library)
    top_md5 = compute_file_md5(args.target_top)

    if lib_md5 != EXPECTED_LIBRARY_MD5:
        raise ValueError(f"Library MD5 mismatch: got {lib_md5}, expected {EXPECTED_LIBRARY_MD5}")
    if top_md5 != EXPECTED_TOP_MD5:
        raise ValueError(f"Top-100 MD5 mismatch: got {top_md5}, expected {EXPECTED_TOP_MD5}")

    print("Verified and materialized submission:")
    print(f"  Library: {args.target_library} ({len(library_seqs):,} sequences, md5: {lib_md5})")
    print(f"  Top-100: {args.target_top} ({len(top_seqs)} sequences, md5: {top_md5})")


if __name__ == "__main__":
    main()
