"""Score candidate pool activity using AMPredictor."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from esmc_flow_amp.fasta_io import read_fasta_sequences


def load_ampredictor(vendor_dir: Path):
    """The organisers' AMPredictor (vendored), wrapped as a seqme model returning predicted MIC in uM."""
    import seqme as sm

    return sm.models.ThirdPartyModel(entry_point="ampredictor.predict:predict", path=vendor_dir)


def predict_mic(model, sequences: list[str]) -> np.ndarray:
    return np.asarray(model(sequences)).ravel().astype(float)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate sequences with AMPredictor.")
    parser.add_argument("--pool", required=True, help="Path to input pool FASTA file.")
    parser.add_argument("--out", required=True, help="Output path for activity .npz file.")
    parser.add_argument(
        "--vendor-dir",
        default="vendor/ampredictor",
        help="Path to vendored AMPredictor directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    sequences = read_fasta_sequences(Path(args.pool))
    unique_sequences = sorted(set(sequences))
    print(f"Loaded {len(unique_sequences):,} unique sequences from {args.pool}")

    model = load_ampredictor(Path(args.vendor_dir))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_path.parent / (out_path.stem + "_chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    mics: list[float] = []
    t0 = time.time()
    for start_idx in range(0, len(unique_sequences), args.chunk_size):
        chunk_file = chunks_dir / f"{start_idx:07d}.npy"
        if chunk_file.exists():
            part = np.load(chunk_file)
        else:
            batch = unique_sequences[start_idx : start_idx + args.chunk_size]
            part = predict_mic(model, batch)
            np.save(chunk_file, part)
        mics.extend(part.tolist())
        print(
            f"  Scored {len(mics):,}/{len(unique_sequences):,} sequences ({time.time() - t0:.1f}s)",
            flush=True,
        )

    np.savez(
        out_path,
        seq=np.array(unique_sequences),
        mic=np.array(mics, dtype=float),
    )
    print(f"Scoring complete. Saved {len(unique_sequences):,} predictions to {out_path}")


if __name__ == "__main__":
    main()
