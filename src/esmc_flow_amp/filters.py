"""Physicochemical and synthesizability sequence filters (HydrAMP Filtering-1)."""

from esmc_flow_amp.constants import (
    MAXIMUM_SEQUENCE_LENGTH,
    MINIMUM_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS,
    STANDARD_AMINO_ACIDS_SET,
)

HYDROPHOBIC_RESIDUES = frozenset("AILMFWV")
POSITIVE_RESIDUES = frozenset("KRH")


def contains_cysteine(sequence: str) -> bool:
    """Check if sequence contains any cysteine residues."""
    return "C" in sequence


def has_homopolymer_run(sequence: str, run_length: int = 3) -> bool:
    """Check if sequence contains a homopolymeric run of identical amino acids."""
    return any(amino_acid * run_length in sequence for amino_acid in STANDARD_AMINO_ACIDS)


def has_consecutive_hydrophobic_run(sequence: str, run_length: int = 3) -> bool:
    """Check if sequence contains consecutive hydrophobic residues of given run length."""
    sequence_length = len(sequence)
    for index in range(sequence_length - run_length + 1):
        window = sequence[index : index + run_length]
        if all(residue in HYDROPHOBIC_RESIDUES for residue in window):
            return True
    return False


def has_positive_charge_cluster(
    sequence: str, window_size: int = 5, minimum_positive_count: int = 3
) -> bool:
    """Check if sequence contains a dense cluster of positive residues in any window."""
    sequence_length = len(sequence)
    if sequence_length >= window_size:
        for index in range(sequence_length - window_size + 1):
            window = sequence[index : index + window_size]
            positive_count = sum(1 for residue in window if residue in POSITIVE_RESIDUES)
            if positive_count >= minimum_positive_count:
                return True
    else:
        positive_count = sum(1 for residue in sequence if residue in POSITIVE_RESIDUES)
        if positive_count >= minimum_positive_count:
            return True
    return False


def is_synthetically_viable(sequence: str) -> bool:
    """Evaluate full HydrAMP Filtering-1 rule for synthesizability.

    Requires:
      - No cysteine
      - No homopolymer runs of length >= 3
      - No consecutive hydrophobic runs of length >= 3
      - No >= 3 positive residues (K, R, H) in any 5-residue window
    """
    if contains_cysteine(sequence):
        return False
    if has_homopolymer_run(sequence, run_length=3):
        return False
    if has_consecutive_hydrophobic_run(sequence, run_length=3):
        return False
    return not has_positive_charge_cluster(sequence, window_size=5, minimum_positive_count=3)


def is_valid_canonical_sequence(sequence: str) -> bool:
    """Check whether sequence consists only of 20 canonical amino acids and within length bounds."""
    if not (MINIMUM_SEQUENCE_LENGTH <= len(sequence) <= MAXIMUM_SEQUENCE_LENGTH):
        return False
    return not (set(sequence) - STANDARD_AMINO_ACIDS_SET)
