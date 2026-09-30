"""Masked-LM curriculum training of ESM-C 300M with control-code conditioning.

Reads one curriculum stage (a parquet with sequence, b_charge, b_length, b_hydro), trains with Lightning Fabric and
writes best.pt / last.pt / snapshot_N.pt / history.json / report.json into --out. Stages are chained with --resume, so
each stage starts from the previous stage's weights (see scripts/run_pipeline.sh).

Sequence format: [BOS][charge][length][hydro][sequence][EOS]. Each control is a bucket id, or NULL under classifier-free
dropout. Only residue positions are masked and only they contribute to the loss. The control-token embeddings are a
separate parameter group (--ctrl-lr) because they start random while the rest of the model is pretrained.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from esmc_flow_amp.controls import CONDITIONS, apply_condition_dropout
from esmc_flow_amp.flow_model import ESMCFlowBackbone, adapt_state_dict

TRAIN_COLUMNS = ["sequence", "b_charge", "b_length", "b_hydro"]
VALIDATION_MASK_RATES = (0.15, 0.30, 0.50, 0.70, 0.90)

Pair = tuple[str, tuple[int, int, int]]  # (sequence, (charge, length, hydro) buckets)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one curriculum stage of the ESM-C masked flow."
    )
    parser.add_argument("--data", required=True, help="Curriculum stage parquet.")
    parser.add_argument("--out", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument(
        "--steps", type=int, default=100, help="Steps to run (added to the resumed step)."
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="Backbone learning rate.")
    parser.add_argument(
        "--ctrl-lr", type=float, default=1e-3, help="Control-embedding learning rate."
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--cond-drop", type=float, default=0.15, help="Per-control NULL dropout (enables CFG)."
    )
    parser.add_argument(
        "--mask-frac", type=float, default=0.15, help="Only used with --mask-schedule fixed."
    )
    parser.add_argument("--mask-schedule", choices=["fixed", "uniform"], default="uniform")
    parser.add_argument("--n-val", type=int, default=256, help="Rows held out for validation.")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--snapshot-every", type=int, default=250, help="Model-only snapshot period (0 = off)."
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="16-mixed", help="fp16 on a T4 (bf16 needs sm_80+).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Checkpoint of the previous stage.")
    return parser.parse_args()


def seed_everything(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.Generator(np.random.PCG64(seed))


def read_training_pairs(path: Path) -> list[Pair]:
    frame = pd.read_parquet(path)
    assert set(TRAIN_COLUMNS) <= set(frame.columns), frame.columns.tolist()
    return [
        (str(sequence), (int(charge), int(length), int(hydro)))
        for sequence, charge, length, hydro in zip(
            frame["sequence"], frame["b_charge"], frame["b_length"], frame["b_hydro"]
        )
    ]


def split_train_val(
    pairs: list[Pair], n_val: int, rng: np.random.Generator
) -> tuple[list[Pair], list[Pair]]:
    assert len(pairs) > n_val, f"{len(pairs)} rows cannot spare {n_val} for validation"
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order[n_val:]], [pairs[i] for i in order[:n_val]]


def draw_batch(pairs: list[Pair], size: int, rng: np.random.Generator) -> list[Pair]:
    picked = rng.choice(len(pairs), size=min(size, len(pairs)), replace=False)
    return [pairs[i] for i in picked]


def collate_masked_batch(
    backbone: ESMCFlowBackbone,
    sequences: list[str],
    controls_list: list[tuple[int, ...]],
    mask_frac: float,
    rng: np.random.Generator,
    mask_schedule: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad per-sample masked rows into one batch; pads never contribute to the loss."""
    masked = [
        backbone.mask_residues(
            backbone.ids_with_controls(sequence, controls), mask_frac, rng, mask_schedule
        )
        for sequence, controls in zip(sequences, controls_list)
    ]
    width = max(corrupted.size(0) for corrupted, _, _ in masked)
    inputs = torch.full((len(masked), width), backbone.pad_id, dtype=torch.long)
    labels = torch.full((len(masked), width), -100, dtype=torch.long)
    n_masked = 0
    for row, (corrupted, row_labels, count) in enumerate(masked):
        inputs[row, : corrupted.size(0)] = corrupted
        labels[row, : row_labels.size(0)] = row_labels
        n_masked += count
    return inputs, labels, n_masked


def masked_token_loss(
    backbone: ESMCFlowBackbone, inputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over masked positions (pads and unmasked tokens ignored)."""
    logits = backbone.batch_logits(inputs)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1).to(logits.device), ignore_index=-100
    )


def train_one_batch(
    backbone: ESMCFlowBackbone,
    optimizer: torch.optim.Optimizer,
    fabric: Any,
    batch: list[Pair],
    mask_frac: float,
    cond_drop: float,
    rng: np.random.Generator,
    mask_schedule: str,
) -> float:
    backbone.model.train()
    optimizer.zero_grad(set_to_none=True)
    controls_list = [apply_condition_dropout(buckets, cond_drop, rng) for _, buckets in batch]
    inputs, labels, _ = collate_masked_batch(
        backbone, [sequence for sequence, _ in batch], controls_list, mask_frac, rng, mask_schedule
    )
    loss = masked_token_loss(backbone, inputs, labels)
    fabric.backward(loss)
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
    optimizer.step()
    backbone.model.eval()
    return float(loss.detach())


def validation_loss(
    backbone: ESMCFlowBackbone, val: list[Pair], mask_frac: float, rng: np.random.Generator
) -> float:
    backbone.model.eval()
    total, n_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(val), 64):
            chunk = val[start : start + 64]
            inputs, labels, n_masked = collate_masked_batch(
                backbone, [s for s, _ in chunk], [b for _, b in chunk], mask_frac, rng, "fixed"
            )
            total += float(masked_token_loss(backbone, inputs, labels)) * n_masked
            n_tokens += n_masked
    return total / max(n_tokens, 1)


def evaluate_ladder(
    backbone: ESMCFlowBackbone, val: list[Pair], seed: int
) -> tuple[float, dict[str, float]]:
    """Validation loss at a ladder of mask rates, each with the same fixed mask seed."""
    per_rate = {}
    for rate in VALIDATION_MASK_RATES:
        rng = np.random.Generator(np.random.PCG64(seed))
        per_rate[f"{rate:.2f}"] = round(validation_loss(backbone, val, rate, rng), 4)
    return float(np.mean(list(per_rate.values()))), per_rate


def save_checkpoint(
    fabric: Any, path: Path, backbone: ESMCFlowBackbone, optimizer: Any, step: int, val: float
) -> None:
    fabric.save(
        path,
        {
            "model": backbone.model,
            "opt": optimizer,
            "step": step,
            "val": val,
            "conds": backbone.conds,
        },
    )


def load_checkpoint(path: Path, backbone: ESMCFlowBackbone, optimizer: Any) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint["model"]
    state_dict = raw.state_dict() if hasattr(raw, "state_dict") else raw
    backbone.model.load_state_dict(
        adapt_state_dict(state_dict, backbone.base_vocab, backbone.n_controls)
    )
    if "opt" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["opt"])
        except Exception as exc:  # noqa: BLE001  optimizer state is optional: a fresh stage may change the group sizes
            logger.debug("Could not load optimizer state: %s", exc)
    return int(checkpoint["step"]) + 1


def build_optimizer(backbone: ESMCFlowBackbone, args: argparse.Namespace) -> torch.optim.Optimizer:
    """AdamW with two groups: pretrained weights, and the freshly initialised control rows."""
    control = backbone.control_weight
    control_params = [p for p in backbone.parameters() if p is control]
    base_params = [p for p in backbone.parameters() if p is not control]
    groups: list[dict[str, Any]] = [
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}
    ]
    if control_params:
        groups.append({"params": control_params, "lr": args.ctrl_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def main() -> None:
    args = parse_arguments()
    rng = seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from lightning.fabric import Fabric

    train_pairs, val_pairs = split_train_val(read_training_pairs(Path(args.data)), args.n_val, rng)
    print(
        f"[train] {len(train_pairs):,} train / {len(val_pairs):,} val rows from {args.data}",
        flush=True,
    )

    fabric = Fabric(accelerator=args.accelerator, devices=1, precision=args.precision)
    fabric.launch()
    backbone = ESMCFlowBackbone(conds=CONDITIONS, device="cpu")
    optimizer = build_optimizer(backbone, args)
    backbone.model, optimizer = fabric.setup(backbone.model, optimizer)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), backbone, optimizer)
        print(f"[train] resumed from {args.resume} at step {start_step}", flush=True)

    # Control-embedding displacement from the start of this stage: distinguishes a learning run from a dead one.
    assert backbone.control_weight is not None
    control_start = backbone.control_weight.detach().clone()
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    n_steps = start_step + args.steps
    started = time.time()

    for step in range(start_step, n_steps):
        batch = draw_batch(train_pairs, args.batch, rng)
        train_loss = train_one_batch(
            backbone,
            optimizer,
            fabric,
            batch,
            args.mask_frac,
            args.cond_drop,
            rng,
            args.mask_schedule,
        )

        evaluated = step % args.eval_every == 0 or step == n_steps - 1
        val_loss, val_by_rate, control_norm, control_disp = float("nan"), None, None, None
        if evaluated:
            val_loss, val_by_rate = evaluate_ladder(backbone, val_pairs, args.seed)
            assert backbone.control_weight is not None
            control_now = backbone.control_weight.detach()
            control_norm = float(control_now.norm(dim=1).mean())
            control_disp = float((control_now - control_start).norm(dim=1).mean())
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(fabric, out_dir / "best.pt", backbone, optimizer, step, val_loss)
        if args.snapshot_every and (step + 1) % args.snapshot_every == 0:
            fabric.save(
                out_dir / f"snapshot_{step}.pt",
                {"model": backbone.model, "step": step, "conds": CONDITIONS},
            )

        history.append(
            {
                "step": step,
                "train": round(train_loss, 4),
                "val": round(val_loss, 4) if evaluated else None,
                "val_by_rate": val_by_rate,
                "ctrl_norm": control_norm,
                "ctrl_disp": control_disp,
            }
        )
        if evaluated:
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if evaluated or step % args.log_every == 0:
            controls = f" ctrl_disp {control_disp:.4f}" if control_disp is not None else ""
            print(
                f"  step {step}/{n_steps - 1} train {train_loss:.4f} val {val_loss:.4f}{controls} "
                f"{time.time() - started:.0f}s",
                flush=True,
            )

    save_checkpoint(
        fabric, out_dir / "last.pt", backbone, optimizer, n_steps - 1, history[-1]["val"]
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "report.json").write_text(json.dumps({**vars(args), "final": history[-1]}, indent=2))
    print(f"[train] best val {best_val:.4f}; checkpoints in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
