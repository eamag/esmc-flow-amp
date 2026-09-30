"""Extract ESM-2 650M sequence representations for the candidate pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esmc_flow_amp.fasta_io import read_fasta_sequences


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 representations for candidate pool."
    )
    parser.add_argument("--pool", required=True, help="Input pool FASTA file.")
    parser.add_argument("--out", required=True, help="Output path for .npy embeddings.")
    parser.add_argument("--device", default="mps", help="Compute device (cuda, mps, cpu).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    import seqme as sm

    sequences = read_fasta_sequences(Path(args.pool))
    print(f"Loaded {len(sequences):,} sequences from {args.pool}")

    model = sm.models.ESM2(
        model_name=sm.models.ESM2Checkpoint.t33_650M,
        batch_size=args.batch_size,
        device=args.device,
        verbose=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shards_dir = out_path.parent / (out_path.stem + "_shards")
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_files: list[Path] = []
    for k in range(0, len(sequences), args.shard_size):
        shard_path = shards_dir / f"shard_{k // args.shard_size:04d}.npy"
        if not shard_path.exists():
            batch = sequences[k : k + args.shard_size]
            emb = np.asarray(model(batch), dtype=np.float32)
            np.save(shard_path, emb)
            print(f"Embedded {min(k + args.shard_size, len(sequences)):,}/{len(sequences):,}")
        shard_files.append(shard_path)

    merged = np.concatenate([np.load(s) for s in sorted(shard_files)], axis=0)
    np.save(out_path, merged)
    print(f"Saved {merged.shape} embeddings to {out_path}")


if __name__ == "__main__":
    main()
