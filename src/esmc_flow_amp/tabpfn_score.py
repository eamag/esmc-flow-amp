"""TabPFN oracle inference for MIC potency ranking and hemolysis safety gating.

Loads experimental MIC and hemolysis tables with precomputed descriptors,
fits TabPFNRegressor and TabPFNClassifier, and returns cached predictions.
TABPFN_TOKEN is read from environment by python-dotenv and never printed.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from esmc_flow_amp.constants import PROJECT_ROOT

STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

CTD_PROPERTIES = {
    "hydrophobicity": (
        {"R", "K", "E", "D", "Q", "N"},
        {"G", "A", "S", "T", "P", "H", "Y"},
        {"C", "V", "L", "I", "M", "F", "W"},
    ),
    "vdv_volume": (
        {"G", "A", "S", "C", "T", "P", "D"},
        {"N", "V", "E", "Q", "I", "L"},
        {"M", "H", "K", "F", "R", "Y", "W"},
    ),
    "polarity": (
        {"L", "I", "F", "W", "C", "M", "V", "Y"},
        {"P", "A", "T", "G", "S"},
        {"H", "Q", "R", "K", "N", "E", "D"},
    ),
    "polarizability": (
        {"G", "A", "S", "D", "T"},
        {"C", "P", "N", "V", "E", "Q", "I", "L"},
        {"K", "M", "H", "F", "R", "Y", "W"},
    ),
    "charge": (
        {"K", "R"},
        {"A", "N", "C", "Q", "G", "H", "I", "L", "M", "F", "P", "S", "T", "W", "Y", "V"},
        {"D", "E"},
    ),
    "secondary_struct": (
        {"E", "A", "L", "M", "Q", "K", "R", "H"},
        {"V", "I", "Y", "C", "W", "F", "T"},
        {"G", "N", "P", "S", "D"},
    ),
    "solvent_access": (
        {"A", "L", "F", "C", "G", "I", "V", "W"},
        {"R", "K", "Q", "E", "N", "D"},
        {"M", "S", "P", "T", "H", "Y"},
    ),
}

CTD_MAPPINGS: dict[str, dict[str, int]] = {}
for p_name, (g1, g2, g3) in CTD_PROPERTIES.items():
    m: dict[str, int] = {}
    for a in g1:
        m[a] = 1
    for a in g2:
        m[a] = 2
    for a in g3:
        m[a] = 3
    CTD_MAPPINGS[p_name] = m

AUTOCORR_SCALES = {
    "hydro": {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    },
    "charge": {
        "K": 1.0,
        "R": 1.0,
        "H": 0.1,
        "D": -1.0,
        "E": -1.0,
        "A": 0.0,
        "C": 0.0,
        "F": 0.0,
        "G": 0.0,
        "I": 0.0,
        "L": 0.0,
        "M": 0.0,
        "N": 0.0,
        "P": 0.0,
        "Q": 0.0,
        "S": 0.0,
        "T": 0.0,
        "V": 0.0,
        "W": 0.0,
        "Y": 0.0,
    },
    "vol": {
        "G": 60.1,
        "A": 88.6,
        "S": 89.0,
        "C": 108.5,
        "D": 111.1,
        "P": 112.7,
        "N": 114.1,
        "T": 116.1,
        "E": 138.4,
        "V": 140.0,
        "Q": 143.8,
        "H": 153.2,
        "M": 162.9,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "R": 173.4,
        "F": 189.9,
        "Y": 193.6,
        "W": 227.8,
    },
    "polar": {
        "A": 8.1,
        "R": 10.5,
        "N": 11.6,
        "D": 13.0,
        "C": 5.5,
        "E": 12.3,
        "Q": 10.5,
        "G": 9.0,
        "H": 10.4,
        "I": 5.2,
        "L": 4.9,
        "K": 11.3,
        "M": 5.7,
        "F": 5.2,
        "P": 8.0,
        "S": 9.2,
        "T": 8.6,
        "W": 5.4,
        "Y": 6.2,
        "V": 5.9,
    },
}


def compute_peptide_descriptors(seq: str) -> dict[str, float]:
    """Compute 300 physicochemical descriptors matching oracle training schema."""
    import peptides

    seq = str(seq).strip().upper()
    if not seq or not set(seq).issubset(STANDARD_AA_SET):
        raise ValueError(f"Invalid sequence: {seq}")

    n = len(seq)
    p = peptides.Peptide(seq)
    feats: dict[str, float] = {}

    counts = p.counts()
    for a in STANDARD_AA:
        feats[f"AAC_{a}"] = float(counts.get(a, 0)) / n

    feats["length"] = float(n)
    feats["molecular_weight"] = float(p.molecular_weight())
    q = float(p.charge(pH=7.4))
    feats["charge_pH7"] = q
    feats["charge_density"] = q / n
    feats["isoelectric_point"] = float(p.isoelectric_point())
    feats["aliphatic_index"] = float(p.aliphatic_index())
    feats["instability_index"] = float(p.instability_index())
    feats["boman_index"] = float(p.boman())
    feats["hydrophobicity_gravy"] = float(p.hydrophobicity(scale="KyteDoolittle"))
    feats["hydrophobic_moment_alpha"] = float(p.hydrophobic_moment(angle=100))
    feats["hydrophobic_moment_beta"] = float(p.hydrophobic_moment(angle=160))

    feats.update({k: float(v) for k, v in p.descriptors().items()})

    for p_name, mapping in CTD_MAPPINGS.items():
        encoded = [mapping.get(a, 0) for a in seq]
        c1 = sum(1 for x in encoded if x == 1)
        c2 = sum(1 for x in encoded if x == 2)
        c3 = sum(1 for x in encoded if x == 3)
        feats[f"CTD_{p_name}_C1"] = float(c1) / n
        feats[f"CTD_{p_name}_C2"] = float(c2) / n
        feats[f"CTD_{p_name}_C3"] = float(c3) / n

        denom = max(1, n - 1)
        t12 = sum(
            1
            for i in range(n - 1)
            if (encoded[i] == 1 and encoded[i + 1] == 2)
            or (encoded[i] == 2 and encoded[i + 1] == 1)
        )
        t13 = sum(
            1
            for i in range(n - 1)
            if (encoded[i] == 1 and encoded[i + 1] == 3)
            or (encoded[i] == 3 and encoded[i + 1] == 1)
        )
        t23 = sum(
            1
            for i in range(n - 1)
            if (encoded[i] == 2 and encoded[i + 1] == 3)
            or (encoded[i] == 3 and encoded[i + 1] == 2)
        )
        feats[f"CTD_{p_name}_T12"] = float(t12) / denom
        feats[f"CTD_{p_name}_T13"] = float(t13) / denom
        feats[f"CTD_{p_name}_T23"] = float(t23) / denom

        for g_idx, g_count in [(1, c1), (2, c2), (3, c3)]:
            positions = [i + 1 for i, x in enumerate(encoded) if x == g_idx]
            if g_count == 0:
                for pct in [0, 25, 50, 75, 100]:
                    feats[f"CTD_{p_name}_D{g_idx}_{pct}"] = 0.0
            else:
                for pct in [0, 25, 50, 75, 100]:
                    idx = math.ceil(pct * g_count / 100.0) - 1
                    idx = max(0, min(idx, g_count - 1))
                    feats[f"CTD_{p_name}_D{g_idx}_{pct}"] = float(positions[idx]) / n

    for s_name, s_dict in AUTOCORR_SCALES.items():
        vals = [s_dict.get(a, 0.0) for a in seq]
        mean_v = sum(vals) / n
        var_v = sum((x - mean_v) ** 2 for x in vals) / n
        std_v = math.sqrt(var_v) if var_v > 1e-9 else 1.0
        norm_v = [(x - mean_v) / std_v for x in vals]
        for lag in range(1, 6):
            if n > lag:
                ac = sum(norm_v[i] * norm_v[i + lag] for i in range(n - lag)) / (n - lag)
            else:
                ac = 0.0
            feats[f"AC_{s_name}_lag{lag}"] = float(ac)

    return feats


_DESCRIPTOR_NAMES: list[str] | None = None


def get_descriptor_names() -> list[str]:
    global _DESCRIPTOR_NAMES
    if _DESCRIPTOR_NAMES is None:
        sample = compute_peptide_descriptors("GLPRKILCAIAKKKGKCKGPLKLVCKC")
        _DESCRIPTOR_NAMES = sorted(sample.keys())
    return _DESCRIPTOR_NAMES


def featurize_many(sequences: list[str]) -> pd.DataFrame:
    names = get_descriptor_names()
    rows = []
    for s in sequences:
        d = compute_peptide_descriptors(s)
        rows.append([d[k] for k in names])
    return pd.DataFrame(rows, columns=names, dtype=np.float32)


CACHE_PATH = PROJECT_ROOT / "data" / "oracle" / "tabpfn_cache.parquet"
_STATE: dict[str, Any] = {}


def _fit_models() -> dict[str, Any]:
    if _STATE:
        return _STATE
    load_dotenv()
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN is missing from the environment")

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mic_path = PROJECT_ROOT / "data" / "oracle" / "peptide_mic_targets_v3.parquet"
    desc_mic_path = PROJECT_ROOT / "data" / "oracle" / "cache_desc_mic_v3.parquet"
    hemo_path = PROJECT_ROOT / "data" / "oracle" / "peptide_hemolysis_v3.parquet"
    desc_hemo_path = PROJECT_ROOT / "data" / "oracle" / "cache_desc_hemo_v3.parquet"

    mic_df = pd.read_parquet(mic_path)
    x_mic = pd.read_parquet(desc_mic_path).values.astype(np.float32)
    valid_mic = mic_df["mic_broad_log10"].notna().values

    hemo_df = pd.read_parquet(hemo_path)
    use_hemo = hemo_df["hc50_usable"] == True
    x_hemo = pd.read_parquet(desc_hemo_path).values.astype(np.float32)

    reg = TabPFNRegressor(device=dev, random_state=42, show_progress_bar=False)
    reg.fit(x_mic[valid_mic], mic_df.loc[valid_mic, "mic_broad_log10"].values.astype(np.float32))

    clf = TabPFNClassifier(device=dev, random_state=42, show_progress_bar=False)
    clf.fit(x_hemo, hemo_df.loc[use_hemo, "is_hemolytic"].values.astype(int))

    print(
        f"TabPFN fitted on {valid_mic.sum():,} MIC rows and {len(x_hemo):,} hemolysis rows ({dev})"
    )
    _STATE.update(reg=reg, clf=clf)
    return _STATE


def score_sequences(sequences: list[str], chunk_size: int = 1000) -> pd.DataFrame:
    """Score sequences with TabPFN predicting log10 MIC and hemolysis probability."""
    cache = (
        pd.read_parquet(CACHE_PATH)
        if CACHE_PATH.exists()
        else pd.DataFrame({"sequence": [], "mic_log10": [], "p_hemo": []})
    )
    have_set = set(cache["sequence"])
    pending_seqs = [s for s in dict.fromkeys(sequences) if s not in have_set]

    if pending_seqs:
        state = _fit_models()
        parts = [cache]
        t0 = time.time()
        for idx in range(0, len(pending_seqs), chunk_size):
            chunk = pending_seqs[idx : idx + chunk_size]
            features = featurize_many(chunk).values.astype(np.float32)
            pred_mic = np.asarray(state["reg"].predict(features), dtype=float)
            pred_hemo = np.asarray(state["clf"].predict_proba(features))[:, 1]
            parts.append(
                pd.DataFrame({"sequence": chunk, "mic_log10": pred_mic, "p_hemo": pred_hemo})
            )
            if (idx // chunk_size) % 5 == 4 or idx + chunk_size >= len(pending_seqs):
                cache = pd.concat(parts, ignore_index=True)
                parts = [cache]
                cache.to_parquet(CACHE_PATH)
                print(
                    f"  TabPFN scored {min(idx + chunk_size, len(pending_seqs)):,}/{len(pending_seqs):,} "
                    f"({time.time() - t0:.1f}s)"
                )
        cache = pd.concat(parts, ignore_index=True)

    return cache.drop_duplicates("sequence").set_index("sequence").loc[sequences].reset_index()
