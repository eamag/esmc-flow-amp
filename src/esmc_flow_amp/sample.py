"""Classifier-free guided masked flow sampler for generating peptide candidate pools."""

from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from esmc_flow_amp.constants import (
    ANTIBACTERIAL_FASTA_PATH,
    DBAASP_FASTA_PATH,
    MAXIMUM_SEQUENCE_LENGTH,
    MINIMUM_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS,
    STANDARD_AMINO_ACIDS_SET,
)
from esmc_flow_amp.controls import CONDITIONS, NULL_BUCKET, sampling_length_bucket
from esmc_flow_amp.fasta_io import read_fasta_sequences, write_fasta_sequences
from esmc_flow_amp.flow_model import ESMCFlowBackbone, adapt_state_dict


@functools.lru_cache(maxsize=1)
def load_dbaasp_lengths(path: Path = DBAASP_FASTA_PATH) -> list[int]:
    """Load empirical reference length distribution from DBAASP FASTA."""
    lengths: list[int] = []
    cur = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if cur > 0:
                    lengths.append(cur)
                    cur = 0
            else:
                cur += len(s)
        if cur > 0:
            lengths.append(cur)
    return [l for l in lengths if MINIMUM_SEQUENCE_LENGTH <= l <= MAXIMUM_SEQUENCE_LENGTH]


def residue_map(backbone: ESMCFlowBackbone) -> dict[int, str]:
    """Map residue token IDs back to 1-letter canonical amino acids."""
    inv: dict[int, str] = {}
    for aa in STANDARD_AMINO_ACIDS:
        core = backbone.encode_sequence(aa)
        if len(core) == 1:
            inv[core[0]] = aa
    return inv


def fp16_logits(backbone: ESMCFlowBackbone, device_type: str):
    """Forward pass under fp16 autocast, logits cast back to fp32 (how the shipped pool was drawn)."""
    forward = backbone.batch_logits

    def autocast_forward(padded: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type, dtype=torch.float16):
            return forward(padded).float()

    return autocast_forward


def draw_top_p(probs: torch.Tensor, top_p: float, rng: torch.Generator) -> torch.Tensor:
    """Sample one token ID per position using nucleus (top-p) sampling."""
    leading, width = probs.shape[:-1], probs.shape[-1]
    flat = probs.reshape(-1, width)
    if top_p >= 1.0:
        return torch.multinomial(flat, 1, generator=rng).reshape(leading)
    ordered, indices = torch.sort(flat, descending=True, dim=-1)
    keep = (ordered.cumsum(-1) - ordered) <= top_p
    ordered = (ordered * keep) / ordered.sum(-1, keepdim=True).clamp_min(1e-12)
    drawn = indices.gather(-1, torch.multinomial(ordered, 1, generator=rng))
    return drawn.reshape(leading)


def unmask_batch(
    backbone: ESMCFlowBackbone,
    lengths: list[int],
    ctrls_list: list[tuple[int, ...]],
    steps: int,
    temperature: float,
    top_p: float,
    rng: torch.Generator,
    cfg_weight: float = 1.0,
    residue_ids: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Unmask residues across steps in confidence order with CFG."""
    n_ctrl = len(backbone.conds)
    width = 2 + n_ctrl + max(lengths)
    dev = next(backbone.model.parameters()).device

    idx = torch.full((len(lengths), width), backbone.pad_id, dtype=torch.long)
    idx[:, 0] = 0
    for i, row_ctrls in enumerate(ctrls_list):
        for j, (b, c) in enumerate(zip(row_ctrls, backbone.conds)):
            idx[i, 1 + j] = backbone.control_id(c, b)
    first = 1 + n_ctrl
    for i, L in enumerate(lengths):
        idx[i, first : first + L] = backbone.mask_id
        idx[i, first + L] = 2
    idx = idx.to(dev)

    use_cfg = (
        cfg_weight != 1.0
        and n_ctrl > 0
        and not all(b == NULL_BUCKET for row_c in ctrls_list for b in row_c)
    )

    for s in range(steps):
        masked = idx == backbone.mask_id
        active = masked.any(-1)
        if not bool(active.any()):
            break

        if use_cfg:
            logits_cond = backbone.batch_logits(idx)
            idx_null = idx.clone()
            for j, c in enumerate(backbone.conds):
                idx_null[:, 1 + j] = backbone.control_id(c, NULL_BUCKET)
            logits_null = backbone.batch_logits(idx_null)
            logits = logits_null + cfg_weight * (logits_cond - logits_null)
        else:
            logits = backbone.batch_logits(idx)

        logits = logits / max(temperature, 1e-6)
        if residue_ids is not None:
            banned = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            banned[residue_ids.to(logits.device)] = False
            logits[:, :, banned] = -float("inf")
        else:
            logits[:, :, backbone.pad_id] = -float("inf")
            logits[:, :, 0] = -float("inf")
            logits[:, :, backbone.mask_id] = -float("inf")
            logits[:, :, backbone.base_vocab :] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        confidence, _ = probs.max(-1)
        confidence = confidence.masked_fill(~masked, -1.0)
        draw = draw_top_p(probs, top_p, rng)
        rem_steps = steps - s

        for b in range(idx.size(0)):
            if not bool(active[b]):
                continue
            m = int(masked[b].sum().item())
            n_unmask = (m + rem_steps - 1) // rem_steps
            order = torch.argsort(confidence[b], descending=True)[:n_unmask]
            idx[b, order] = draw[b, order]

    rows = [idx[i, first : first + L].clone() for i, L in enumerate(lengths)]
    for i, row in enumerate(rows):
        n_mask = int((row == backbone.mask_id).sum().item())
        assert n_mask == 0, f"Sequence {i} returned with {n_mask} leftover mask tokens"
    return rows


@torch.no_grad()
def sample_sequences(
    backbone: ESMCFlowBackbone,
    inv: dict[int, str],
    n: int,
    lengths: list[int],
    ctrls_list: Sequence[tuple[int, ...]],
    *,
    cfg_weight: float = 1.0,
    steps: int = 12,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    chunk: int = 256,
) -> list[str]:
    """Sample n peptide sequences in chunks."""
    backbone.model.eval()
    dev = next(backbone.model.parameters()).device
    rng = torch.Generator(device=dev).manual_seed(seed)
    residue_ids = torch.tensor(sorted(inv.keys()), dtype=torch.long, device=dev)

    out: list[str] = []
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        chunk_lengths = [lengths[(start + i) % len(lengths)] for i in range(size)]
        chunk_ctrls = [ctrls_list[(start + i) % len(ctrls_list)] for i in range(size)]
        rows = unmask_batch(
            backbone,
            chunk_lengths,
            chunk_ctrls,
            steps,
            temperature,
            top_p,
            rng,
            cfg_weight=cfg_weight,
            residue_ids=residue_ids,
        )
        for row in rows:
            out.append("".join(inv[int(t)] for t in row))
    return out


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample candidate peptide pool from ESM-C flow.")
    parser.add_argument("--checkpoint", required=True, help="Trained ESM-C checkpoint path.")
    parser.add_argument(
        "--spec",
        default="b0:900,b1:3600,b2:11100,b3:10800,b4:3000,b5:600",
        help="Charge mixture specification (e.g. 'b1:3600,b2:11100,...').",
    )
    parser.add_argument("--parts", type=int, default=10, help="Number of 30k draw parts to sample.")
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--antibacterial-fasta", default=str(ANTIBACTERIAL_FASTA_PATH))
    parser.add_argument("--out", required=True, help="Output path for merged pool FASTA.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dev_str = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(dev_str)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conds = tuple(ckpt.get("conds", CONDITIONS))
    backbone = ESMCFlowBackbone(conds=conds, device="cpu")
    state_dict = (
        ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    )
    state_dict = adapt_state_dict(state_dict, backbone.base_vocab, backbone.n_controls)
    backbone.model.load_state_dict(state_dict)
    backbone.model.to(device)
    backbone.model.eval()
    if device.type in ("cuda", "mps"):
        backbone.batch_logits = fp16_logits(backbone, device.type)

    inv = residue_map(backbone)
    reference_lengths = load_dbaasp_lengths()
    rng = np.random.default_rng(args.seed)

    spec_parts = []
    for item in args.spec.split(","):
        bucket_str, count_str = item.strip().split(":")
        spec_parts.append((int(bucket_str.lstrip("b")), int(count_str)))

    all_sequences: set[str] = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.parts} parts with spec: {args.spec} on {device}")
    for part_idx in range(args.parts):
        t0 = time.time()
        part_sequences: list[str] = []
        for b_charge, count in spec_parts:
            # Draw lengths from empirical distribution
            drawn_lengths = [int(rng.choice(reference_lengths)) for _ in range(count)]
            b_lengths = [sampling_length_bucket(l) for l in drawn_lengths]
            # Construct control tokens for this bucket
            ctrls = [(b_charge, b_l, NULL_BUCKET) for b_l in b_lengths]

            sampled = sample_sequences(
                backbone,
                inv,
                count,
                drawn_lengths,
                ctrls,
                cfg_weight=args.guidance_weight,
                steps=args.steps,
                seed=args.seed + part_idx * 100 + b_charge,
                chunk=args.chunk,
            )
            part_sequences.extend(sampled)

        before = len(all_sequences)
        all_sequences.update(part_sequences)
        print(
            f"Part {part_idx + 1}/{args.parts}: sampled {len(part_sequences):,} sequences "
            f"({len(all_sequences) - before:,} new unique, total {len(all_sequences):,}) "
            f"in {time.time() - t0:.1f}s"
        )

    # The pool is the candidate universe: unique, canonical, 8-50 aa, and not an exact reference match.
    reference = set(read_fasta_sequences(Path(args.antibacterial_fasta)))
    pool = sorted(
        s
        for s in all_sequences
        if MINIMUM_SEQUENCE_LENGTH <= len(s) <= MAXIMUM_SEQUENCE_LENGTH
        and set(s) <= STANDARD_AMINO_ACIDS_SET
        and s not in reference
    )
    write_fasta_sequences(pool, out_path, header_prefix="esmc_flow")
    print(f"Wrote {len(pool):,} unique pool sequences to {out_path}")


if __name__ == "__main__":
    main()
