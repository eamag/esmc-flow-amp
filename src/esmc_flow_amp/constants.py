"""Named constants, configuration dataclasses, and path defaults for esmc-flow-amp."""

from dataclasses import dataclass
from pathlib import Path

# Amino acid alphabet
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AMINO_ACIDS_SET = frozenset(STANDARD_AMINO_ACIDS)

# Competition limits and targets
MINIMUM_SEQUENCE_LENGTH = 8
MAXIMUM_SEQUENCE_LENGTH = 50
SIMILARITY_THRESHOLD = 0.80

TARGET_LIBRARY_SIZE = 50_000
TARGET_TOP_COUNT = 100

# Expected MD5 checksums of the final submission pair
EXPECTED_LIBRARY_MD5 = "5cc81ec3b95b1765069b5fd86cbfac40"
EXPECTED_TOP_MD5 = "a00448984978e3d08c0fe4263e69f2eb"

# Standard paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COMMITTED_LIBRARY_PATH = PROJECT_ROOT / "data" / "final" / "library.fasta"
COMMITTED_TOP_PATH = PROJECT_ROOT / "data" / "final" / "top.fasta"
GENERATED_LIBRARY_PATH = PROJECT_ROOT / "generate" / "library.fasta"
GENERATED_TOP_PATH = PROJECT_ROOT / "generate" / "top.fasta"
ANTIBACTERIAL_FASTA_PATH = PROJECT_ROOT / "data" / "antibacterial.fasta"
DBAASP_FASTA_PATH = PROJECT_ROOT / "data" / "dbaasp.fasta"


@dataclass(frozen=True)
class ShippedLibraryConfig:
    """Fixed parameters for the shipped library selection (o2k0d1p97).

    Omega = 2 (Conformity), Eta = 4 (ESM-2 density ratio), Kappa = 0 (Precision proxy),
    Delta = 1 (Diversity proxy), with hard Precision filter >= 0.97.
    """

    name: str = "o2k0d1p97"
    omega: float = 2.0
    eta: float = 4.0
    kappa: float = 0.0
    delta: float = 1.0
    precision_threshold: float = 0.97
    active_weight: float = 1.8
    synthesizability_weight: float = 0.5
    near_duplicate_cutoff: float = 90.0
    target_size: int = TARGET_LIBRARY_SIZE


@dataclass(frozen=True)
class ShippedTop100Config:
    """Fixed parameters for the shipped top-100 selection.

    Shortlist of 8,000 filtered candidates scored by TabPFN with hemolysis gate < 0.35.
    """

    shortlist_size: int = 8000
    hemolysis_gate: float = 0.35
    ampredictor_gate: float = 16.0
    weight_amp: float = 0.0
    novelty_cutoff: float = 0.79
    near_duplicate_cutoff: float = 80.0
    target_count: int = TARGET_TOP_COUNT
