"""Control token definitions, attribute computation, and bucketization for conditional flow."""

from collections.abc import Sequence

import numpy as np

# Condition token layout
CONDITIONS: tuple[str, ...] = ("charge", "length", "hydro")
N_BUCKET: int = 7
NULL_BUCKET: int = 7
CONTROL_STRIDE: int = 8  # 8 slots per condition: buckets 0-6 plus null token (7)

# Discretization boundaries
CHARGE_EDGES: list[float] = [0.0, 2.0, 4.0, 6.0, 8.0, 11.0]
LENGTH_EDGES: list[float] = [12.0, 16.0, 20.0, 26.0, 34.0, 44.0]
HYDRO_EDGES: list[float] = [-0.45, -0.25, -0.10, 0.05, 0.25, 0.50]

# Canonical amino acids
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(STANDARD_AA)}

# Bjellqvist pKa constants for net charge at pH 7.0
PKA_POSITIVE = {"K": 10.54, "R": 12.48, "H": 6.04}
PKA_NEGATIVE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07}
PKA_N_TERM = 9.69
PKA_C_TERM = 2.34

# Eisenberg consensus hydrophobicity scale
EISENBERG_SCALE = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}


def _charge_vector(pH: float = 7.0) -> np.ndarray:
    vector = np.zeros(20, dtype=np.float64)
    for residue, pka in PKA_POSITIVE.items():
        vector[AA_INDEX[residue]] = 1.0 / (1.0 + 10.0 ** (pH - pka))
    for residue, pka in PKA_NEGATIVE.items():
        vector[AA_INDEX[residue]] = -1.0 / (1.0 + 10.0 ** (pka - pH))
    return vector


def counts_matrix(sequences: list[str]) -> np.ndarray:
    """Compute (N, 20) uint8 residue count matrix for canonical sequences."""
    num_sequences = len(sequences)
    counts = np.zeros((num_sequences, 20), dtype=np.uint8)
    for row_idx, seq in enumerate(sequences):
        for residue in seq:
            idx = AA_INDEX.get(residue)
            if idx is not None:
                counts[row_idx, idx] += 1
    return counts


def compute_sequence_attributes(
    sequences: list[str],
    pH: float = 7.0,
    counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of (length, net_charge, eisenberg_hydrophobicity)."""
    counts_arr = counts_matrix(sequences) if counts is None else counts
    counts_float = counts_arr.astype(np.float32)
    lengths = counts_float.sum(axis=1)

    terminal_charge = (1.0 / (1.0 + 10.0 ** (pH - PKA_N_TERM))) - (
        1.0 / (1.0 + 10.0 ** (PKA_C_TERM - pH))
    )
    charges = counts_float @ _charge_vector(pH).astype(np.float32) + np.float32(terminal_charge)

    hydro_weights = np.array([EISENBERG_SCALE[a] for a in STANDARD_AA], dtype=np.float32)
    hydrophobicity = (counts_float @ hydro_weights) / np.maximum(lengths, 1.0)

    return lengths, charges, hydrophobicity


def bucketize(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Training-time bucket index: a value equal to an edge falls in the LOWER bucket.

    Same rule as ``torch.bucketize(values, edges)`` (float32), which built the shipped training tables.
    """
    edges_arr = np.asarray(edges, dtype=np.float32)
    return np.searchsorted(edges_arr, np.asarray(values, dtype=np.float32), side="left").astype(
        np.uint8
    )


def sampling_length_bucket(length: int) -> int:
    """Sampling-time length bucket: a length equal to an edge falls in the UPPER bucket.

    The sampler that drew the shipped pool used ``np.searchsorted(LENGTH_EDGES, L, side="right")``, whereas the
    training tables above use the lower-bucket rule, so a peptide of exactly 12, 16, 20, 26, 34 or 44 residues is
    requested one length bucket higher than it was trained under. This is kept as run, not corrected.
    """
    return int(np.searchsorted(LENGTH_EDGES, length, side="right"))


def compute_control_buckets(
    sequences: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (b_charge, b_length, b_hydro) discrete buckets for a list of sequences."""
    lengths, charges, hydros = compute_sequence_attributes(sequences)
    b_charge = bucketize(charges, CHARGE_EDGES)
    b_length = bucketize(lengths, LENGTH_EDGES)
    b_hydro = bucketize(hydros, HYDRO_EDGES)
    return b_charge, b_length, b_hydro


def apply_condition_dropout(
    controls: tuple[int, ...],
    drop_probability: float,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Apply classifier-free guidance condition dropout by replacing buckets with NULL_BUCKET."""
    return tuple(NULL_BUCKET if rng.random() < drop_probability else int(b) for b in controls)
