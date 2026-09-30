"""ESM-C 300M masked flow architecture with split control token embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from esmc_flow_amp.controls import (
    CONDITIONS,
    CONTROL_STRIDE,
    NULL_BUCKET,
)


def parent_module(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    """Retrieve parent submodule and target attribute name from dotpath."""
    node = model
    *pre, last = path.split(".")
    for part in pre:
        node = getattr(node, part)
    return node, last


class SplitEmbedding(nn.Module):
    """Token embedding whose control rows are a separate parameter.

    Keeps base weights (pretrained vocab) and control weights (control codes) separate
    so the optimizer can assign them distinct parameter groups and learning rates.
    """

    def __init__(self, base_weight: torch.Tensor, extra_rows: int, init_std: float):
        super().__init__()
        self.n_base = int(base_weight.size(0))
        self.embedding_dim = int(base_weight.size(1))
        self.num_embeddings = self.n_base + extra_rows
        self.weight = nn.Parameter(base_weight.detach().clone())
        self.control_weight = nn.Parameter(
            torch.empty(extra_rows, self.embedding_dim, dtype=base_weight.dtype)
        )
        with torch.no_grad():
            self.control_weight.normal_(std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, torch.cat([self.weight, self.control_weight], dim=0))


def adapt_state_dict(state_dict: dict[str, Any], base_size: int, extra_rows: int) -> dict[str, Any]:
    """Split unified embedding weights into pretrained and control parts if needed."""
    if any("control_weight" in k for k in state_dict):
        return state_dict
    if extra_rows <= 0:
        return state_dict
    out = dict(state_dict)
    target_rows = base_size + extra_rows
    for k, v in list(state_dict.items()):
        if (
            k.endswith("embed.weight")
            and hasattr(v, "shape")
            and len(v.shape) >= 1
            and v.shape[0] == target_rows
        ):
            prefix = k[: -len("weight")]
            out[k] = v[:base_size].clone()
            out[f"{prefix}control_weight"] = v[base_size:].clone()
    return out


def extend_vocab(model: nn.Module, base_size: int, extra_rows: int) -> None:
    """Expand token embedding and output head by extra_rows for control codes."""
    if any(isinstance(m, SplitEmbedding) for m in model.modules()):
        return
    embed_path = next(
        path
        for path, m in model.named_modules()
        if isinstance(m, nn.Embedding) and m.num_embeddings == base_size
    )
    head_path = next(
        path
        for path, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.out_features == base_size
    )
    old_embed = getattr(*parent_module(model, embed_path))
    base_rows = old_embed.weight[:base_size].detach()
    row_norms = base_rows.norm(dim=1)
    live_rows = base_rows[row_norms >= row_norms.median()]
    init_std = float(live_rows.std())
    new_embed = SplitEmbedding(old_embed.weight, extra_rows, init_std)
    setattr(*parent_module(model, embed_path), new_embed)

    old_head = getattr(*parent_module(model, head_path))
    new_head = nn.Linear(
        old_head.in_features,
        base_size + extra_rows,
        bias=old_head.bias is not None,
        dtype=old_head.weight.dtype,
    )
    with torch.no_grad():
        new_head.weight[:base_size] = old_head.weight
        new_head.weight[base_size:].normal_(std=0.02)
        if old_head.bias is not None:
            new_head.bias[:base_size] = old_head.bias
            new_head.bias[base_size:].zero_()
    setattr(*parent_module(model, head_path), new_head)


class ESMCFlowBackbone:
    """ESM-C 300M backbone equipped with discrete control tokens for masked flow transport."""

    def __init__(
        self,
        conds: tuple[str, ...] = CONDITIONS,
        device: torch.device | str = "cpu",
    ):
        from esm.models.esmc import ESMC

        self.conds = conds
        device = torch.device(device)
        self.model = ESMC.from_pretrained("esmc_300m", device=device)
        self.mask_id = self.model.tokenizer.mask_token_id
        self.pad_id = self.model.tokenizer.pad_token_id

        split_embed = next(
            (m for m in self.model.modules() if isinstance(m, SplitEmbedding)),
            None,
        )
        if split_embed is not None:
            self.base_vocab = split_embed.n_base
        else:
            self.base_vocab = next(
                m.num_embeddings for m in self.model.modules() if isinstance(m, nn.Embedding)
            )
        self.n_controls = CONTROL_STRIDE * len(conds)
        if self.n_controls:
            extend_vocab(self.model, self.base_vocab, self.n_controls)
        self.model.to(device)
        self.model.eval()

    @property
    def control_weight(self) -> nn.Parameter | None:
        """Access the separate control embedding parameter."""
        for m in self.model.modules():
            if isinstance(m, SplitEmbedding):
                return m.control_weight
        return None

    def control_id(self, cond: str, bucket: int) -> int:
        """Map condition name and bucket index to absolute token ID."""
        assert 0 <= bucket <= NULL_BUCKET, f"Bucket index {bucket} out of bounds"
        return self.base_vocab + CONTROL_STRIDE * self.conds.index(cond) + int(bucket)

    def encode_sequence(self, seq: str) -> list[int]:
        """Convert sequence string into list of residue token IDs."""
        raw = [int(i) for i in self.model._tokenize([seq])[0]]
        return [i for i in raw if i not in (0, 1, 2)]

    def ids_with_controls(self, seq: str, ctrls: tuple[int, ...]) -> list[int]:
        """Construct [BOS][controls][sequence][EOS] token ID representation."""
        controls = [self.control_id(c, b) for b, c in zip(ctrls, self.conds)]
        return [0, *controls, *self.encode_sequence(seq), 2]

    def batch_logits(self, padded: torch.Tensor) -> torch.Tensor:
        """Forward pass over padded token batch retaining autograd gradients."""
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = padded.device
        out: Any = self.model(sequence_tokens=padded.to(model_device))
        logits = getattr(out, "sequence_logits", getattr(out, "logits", None))
        assert logits is not None, "Logits tensor missing from ESM-C output"
        return logits

    def mask_residues(
        self,
        full: list[int],
        mask_frac: float,
        rng: np.random.Generator,
        mask_schedule: str = "uniform",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Randomly mask peptide residues while keeping control tokens and BOS/EOS visible."""
        rate = float(rng.uniform(0.0, 1.0)) if mask_schedule == "uniform" else mask_frac
        n_prefix = 1 + len(self.conds)
        ids = torch.tensor(full, dtype=torch.long)
        positions = torch.arange(n_prefix, ids.size(0) - 1)
        picked = torch.from_numpy(rng.random(len(positions)) < rate)
        if not bool(picked.any()):
            picked[rng.integers(0, len(positions))] = True
        masked_at = positions[picked]
        labels = torch.full_like(ids, -100)
        labels[masked_at] = ids[masked_at]
        corrupted = ids.clone()
        corrupted[masked_at] = self.mask_id
        return corrupted, labels, int(picked.sum())

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()
