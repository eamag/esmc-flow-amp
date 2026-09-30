"""Novelty evaluation against antibacterial reference sequences."""

from typing import cast

import Levenshtein
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process as rapidfuzz_process

from esmc_flow_amp.constants import SIMILARITY_THRESHOLD


def calculate_levenshtein_ratio(sequence_a: str, sequence_b: str) -> float:
    """Compute normalized Levenshtein similarity ratio between two sequences."""
    return float(Levenshtein.ratio(sequence_a, sequence_b))


def is_novel_against_reference_corpus(
    sequence: str,
    references_by_length: dict[int, list[str]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """Screen candidate against length-indexed reference database with length bounds pruning."""
    sequence_length = len(sequence)
    for reference_length, reference_list in references_by_length.items():
        theoretical_maximum_ratio = (
            2.0 * min(sequence_length, reference_length) / (sequence_length + reference_length)
        )
        if theoretical_maximum_ratio <= similarity_threshold:
            continue

        for reference in reference_list:
            if calculate_levenshtein_ratio(sequence, reference) > similarity_threshold:
                return False

    return True


def compute_maximum_similarity_against_references(
    candidate_sequences: list[str],
    reference_sequences: list[str],
) -> np.ndarray:
    """Compute max Levenshtein similarity ratio for each candidate against reference panel."""
    distance_matrix = rapidfuzz_process.cdist(
        candidate_sequences,
        reference_sequences,
        scorer=fuzz.ratio,
        workers=-1,
        dtype=np.uint8,
    )
    similarity_matrix = distance_matrix
    maximum_similarities = similarity_matrix.max(axis=1) / 100.0
    return cast(np.ndarray, maximum_similarities)
