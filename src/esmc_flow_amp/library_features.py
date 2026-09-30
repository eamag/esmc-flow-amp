"""Compute selection features using a held-out 50/50 reference split.

Extracts:
  - Property Conformity (via 5-fold CV KDE over charge and hydrophobic moment on fit half)
  - ESM-2 48-PCA kNN density ratio (fit half vs pool)
  - 20-draw Precision proxy against fit half
  - Diversity proxy (mean normalized Levenshtein distance to pool subsample)
  - Synthesizability (Filtering-1 pass rate) and activity (MIC <= 16 uM)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import process as rf_process
from rapidfuzz.distance import Levenshtein as Lev
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors

from esmc_flow_amp.constants import DBAASP_FASTA_PATH, STANDARD_AMINO_ACIDS_SET
from esmc_flow_amp.fasta_io import read_fasta_sequences
from esmc_flow_amp.filters import is_synthetically_viable

POS_PKS = {"Nterm": 9.38, "K": 10.67, "R": 12.10, "H": 6.04}
NEG_PKS = {"Cterm": 2.15, "D": 3.71, "E": 4.15, "C": 8.14, "Y": 10.10}
EISENBERG = {
    "I": 1.4,
    "F": 1.2,
    "V": 1.1,
    "L": 1.1,
    "W": 0.81,
    "M": 0.64,
    "A": 0.62,
    "G": 0.48,
    "C": 0.29,
    "Y": 0.26,
    "P": 0.12,
    "T": -0.05,
    "S": -0.18,
    "H": -0.4,
    "E": -0.74,
    "N": -0.78,
    "Q": -0.85,
    "D": -0.9,
    "K": -1.5,
    "R": -2.5,
}


def compute_charge_ph7(seq: str, ph: float = 7.0) -> float:
    pos_charge = 0.0
    for aa, pK in POS_PKS.items():
        c_r = 10.0 ** (pK - ph)
        count = 1.0 if aa == "Nterm" else seq.count(aa)
        pos_charge += count * (c_r / (c_r + 1.0))

    neg_charge = 0.0
    for aa, pK in NEG_PKS.items():
        c_r = 10.0 ** (ph - pK)
        count = 1.0 if aa == "Cterm" else seq.count(aa)
        neg_charge += count * (c_r / (c_r + 1.0))

    return round(pos_charge - neg_charge, 3)


def compute_hydrophobic_moment(seq: str, window: int = 11, angle: int = 100) -> float:
    n = len(seq)
    wdw = min(window, n)
    mtrx = [EISENBERG.get(aa, 0.0) for aa in seq]
    rads = angle * (np.pi / 180.0) * np.arange(wdw)
    cos_rads = np.cos(rads)
    sin_rads = np.sin(rads)
    n_windows = n - wdw + 1
    moms = []
    for i in range(n_windows):
        window_vals = np.array(mtrx[i : i + wdw])
        vcos = float(np.sum(window_vals * cos_rads))
        vsin = float(np.sum(window_vals * sin_rads))
        moms.append(np.sqrt(vcos**2 + vsin**2) / wdw)
    return float(np.mean(moms)) if moms else 0.0


class ReplicaConformityScore:
    """Evaluates conformity against reference using 5-fold CV KDE."""

    def __init__(self, reference_seqs: list[str], n_splits: int = 5, seed: int = 0):
        self.ref_arr = np.array(
            [[compute_charge_ph7(s), compute_hydrophobic_moment(s)] for s in reference_seqs]
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.folds: list[tuple[np.ndarray, KernelDensity]] = []
        for train_idx, val_idx in kf.split(self.ref_arr):
            train = self.ref_arr[train_idx]
            val = self.ref_arr[val_idx]
            kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
            kde.fit(train)
            log_prob_val = kde.score_samples(val)
            self.folds.append((log_prob_val, kde))

    def per_sequence_conformity(self, seq_arr: np.ndarray) -> np.ndarray:
        n_seqs = len(seq_arr)
        total = np.zeros(n_seqs, dtype=float)
        for log_prob_val, kde in self.folds:
            n_val = log_prob_val.shape[0]
            log_prob_gen = kde.score_samples(seq_arr)
            comp_matrix = log_prob_gen[:, None] >= log_prob_val[None, :]
            total += comp_matrix.sum(axis=1) / (n_val + 1.0)
        return total / len(self.folds)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute held-out selection features.")
    parser.add_argument("--pool", required=True, help="Input candidate pool FASTA.")
    parser.add_argument("--activity", required=True, help="Input activity .npz.")
    parser.add_argument("--embeddings", required=True, help="Input pool ESM-2 .npy embeddings.")
    parser.add_argument("--ref-fasta", default=str(DBAASP_FASTA_PATH))
    parser.add_argument(
        "--device", default=None, help="Device for the reference ESM-2 embeddings (default: auto)."
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save computed features.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_seqs = read_fasta_sequences(Path(args.pool))
    act_data = np.load(args.activity)
    act_dict = dict(zip(act_data["seq"].tolist(), act_data["mic"].tolist()))
    mic = np.array([act_dict[s] for s in pool_seqs], dtype=np.float32)
    assert np.isfinite(mic).all() and (mic > 0).all(), "AMPredictor MIC must be finite and positive"

    emb = np.load(args.embeddings, mmap_mode="r")
    lens = np.array([len(s) for s in pool_seqs], dtype=np.int32)
    f1 = np.array([is_synthetically_viable(s) for s in pool_seqs], dtype=bool)

    # The reference is every canonical DBAASP sequence up to 65 aa (8,841 of 8,967), in FASTA order, split 50/50.
    ref_all = read_fasta_sequences(Path(args.ref_fasta))
    ref_clean = [s for s in ref_all if set(s) <= STANDARD_AMINO_ACIDS_SET and len(s) <= 65]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ref_clean))
    fit_idx = np.sort(perm[: len(perm) // 2])
    val_idx = np.sort(perm[len(perm) // 2 :])
    np.savez(out_dir / "split.npz", fit=fit_idx, val=val_idx)

    fit_seqs = [ref_clean[i] for i in fit_idx]
    print(f"Reference split: {len(fit_seqs):,} fit, {len(val_idx):,} validation")

    # Conformity
    # One generator (seed 7) drives the Conformity sample and then the PCA sample below.
    conf_rng = np.random.default_rng(7)
    conformity_fit_sample = [
        fit_seqs[i] for i in conf_rng.choice(len(fit_seqs), 2000, replace=False)
    ]
    scorer = ReplicaConformityScore(conformity_fit_sample)
    charges = np.array([compute_charge_ph7(s) for s in pool_seqs], dtype=np.float32)
    muhs = np.array([compute_hydrophobic_moment(s) for s in pool_seqs], dtype=np.float32)

    conf_parts = []
    for i in range(0, len(pool_seqs), 20000):
        block = np.column_stack([charges[i : i + 20000], muhs[i : i + 20000]])
        conf_parts.append(scorer.per_sequence_conformity(block))
    conf = np.concatenate(conf_parts).astype(np.float32)

    # Density ratio (48-PCA on ESM-2 representations)
    print("Fitting ESM-2 PCA and density ratio...")
    pca_sample_idx = conf_rng.choice(len(emb), min(100000, len(emb)), replace=False)
    pca = PCA(n_components=48, random_state=0).fit(emb[pca_sample_idx].astype(np.float64))
    pool_pca = pca.transform(emb.astype(np.float64)).astype(np.float32)

    # Reference embeddings for fit split
    import seqme as sm

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    ref_model = sm.models.ESM2(
        model_name=sm.models.ESM2Checkpoint.t33_650M,
        batch_size=64,
        device=device,
        verbose=False,
    )
    ref_fit_emb = np.asarray(ref_model(fit_seqs), dtype=np.float32)
    ref_pca = pca.transform(ref_fit_emb.astype(np.float64)).astype(np.float32)

    nn_pool = NearestNeighbors(n_neighbors=26, n_jobs=-1).fit(pool_pca)
    rp = nn_pool.kneighbors(pool_pca)[0][:, -1]

    nn_ref = NearestNeighbors(n_neighbors=25, n_jobs=-1).fit(ref_pca)
    rr = nn_ref.kneighbors(pool_pca)[0][:, -1]

    lw = 48.0 * (np.log(np.maximum(rp, 1e-12)) - np.log(np.maximum(rr, 1e-12)))
    lo, hi = np.percentile(lw, [1, 99])
    lw = np.clip(lw, lo, hi).astype(np.float32)

    # 20-draw Precision proxy
    print("Computing 20-draw Precision proxy...")
    E = torch.from_numpy(np.asarray(emb, dtype=np.float32))
    prec20 = np.zeros(len(pool_seqs), dtype=np.float32)
    draw_rng = np.random.default_rng(23)
    for _ in range(20):
        draw_idx = draw_rng.choice(len(ref_fit_emb), 1000, replace=False)
        R = torch.from_numpy(ref_fit_emb[draw_idx])
        radii = torch.cdist(R, R).kthvalue(13, dim=1).values
        for i in range(0, len(pool_seqs), 16384):
            block_in = (torch.cdist(E[i : i + 16384], R) <= radii[None]).any(dim=1).float().numpy()
            prec20[i : i + 16384] += block_in / 20.0
    np.save(out_dir / "prec20.npy", prec20)

    # Diversity proxy (mean normalized Levenshtein distance to 200 random pool sequences)
    div_sample = [
        pool_seqs[i] for i in np.random.default_rng(5).choice(len(pool_seqs), 200, replace=False)
    ]
    div_parts = []
    for i in range(0, len(pool_seqs), 20000):
        chunk_div = rf_process.cdist(
            pool_seqs[i : i + 20000],
            div_sample,
            scorer=Lev.normalized_distance,
            dtype=np.float32,
            workers=-1,
        ).mean(axis=1)
        div_parts.append(chunk_div)
    div = np.concatenate(div_parts).astype(np.float32)

    # Save merged pool features
    np.savez(
        out_dir / "pool_features.npz",
        seqs=np.array(pool_seqs),
        lens=lens,
        f1=f1,
        mic=mic,
        conf=conf,
        lw=lw,
        div=div,
    )
    print(f"Features saved successfully to {out_dir}")


if __name__ == "__main__":
    main()
