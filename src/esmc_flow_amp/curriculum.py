"""Curriculum tables for the three training stages.

Stage A (5.6M rows, broad peptide prior) and stage B (889k rows, AMP-like) are external tables: their sources and
licences are listed in data/plm/README.md and they are not rebuilt here. Stage C is small and derived from the committed
oracle table, so this module builds it: every measured peptide with a broad-spectrum MIC of at most 16 uM.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from esmc_flow_amp.constants import (
    MAXIMUM_SEQUENCE_LENGTH,
    MINIMUM_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS_SET,
)
from esmc_flow_amp.controls import compute_control_buckets

POTENT_MIC_UM = 16.0  # the competition's activity threshold


def build_stage_c(oracle_mic_path: Path, out_path: Path) -> int:
    """Write the measured-potent stage and return its row count."""
    table = pd.read_parquet(oracle_mic_path, columns=["sequence", "mic_broad_log10"])
    potent = table[
        table["mic_broad_log10"].notna() & (table["mic_broad_log10"] <= math.log10(POTENT_MIC_UM))
    ]
    sequences = sorted(
        {
            s
            for s in potent["sequence"]
            if MINIMUM_SEQUENCE_LENGTH <= len(s) <= MAXIMUM_SEQUENCE_LENGTH
            and set(s) <= STANDARD_AMINO_ACIDS_SET
        }
    )
    charge, length, hydro = compute_control_buckets(sequences)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sequence": sequences, "b_charge": charge, "b_length": length, "b_hydro": hydro}
    ).to_parquet(out_path, index=False)
    return len(sequences)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build curriculum stage C (measured potent peptides)."
    )
    parser.add_argument("--oracle-mic", default="data/oracle/peptide_mic_targets_v3.parquet")
    parser.add_argument("--out", default="data/plm/stage_c_potent.parquet")
    args = parser.parse_args()
    rows = build_stage_c(Path(args.oracle_mic), Path(args.out))
    print(f"Stage C: wrote {args.out} ({rows:,} rows)")


if __name__ == "__main__":
    main()
