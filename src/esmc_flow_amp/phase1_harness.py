"""
Phase-1 Evaluation Harness for AMP Challenge 2027.

Evaluates candidate peptide libraries against the competition's Phase-1 metric families
using seqme and the organizers' official AMPredictor surrogate model:
  1. Sequence-level metrics (Count, Uniqueness, Length, Diversity)
  2. Novelty & Authenticity (exact matches against reference, similarity distribution)
  3. Surrogate activity prediction (AMPredictor MIC vs E. coli: median, % <= 16 uM)
  4. PLM embedding space similarity (ESM-2: FBD, MMD, Precision, Recall vs AMPs/negatives)
  5. Physicochemical property conformity & synthesizability (charge, moment, Filtering-1 pass rate)

This is the harness that produced the numbers in metrics/PHASE1_DEEP3.md. Any FASTA files can be compared by name; the
generic-peptide (UniProt) and MarLys references are optional and are skipped when not given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import seqme as sm

if TYPE_CHECKING:
    from esm.models.esmc import ESMC

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = set("AILMFWV")
POSITIVE_RESIDUES = set("KRH")


def read_fasta_sequences(path: Path) -> list[str]:
    """Read sequences from a FASTA file, stripping headers and invalid characters."""
    result = sm.read_fasta(path)
    # sm.read_fasta returns a (headers, sequences) tuple only when return_headers=True.
    raw_seqs = result[1] if isinstance(result, tuple) else result
    return [s.strip().upper() for s in raw_seqs if s.strip()]


def subsample_sequences(sequences: list[str], n_samples: int, seed: int) -> list[str]:
    """Deterministically subsample; unwraps seqme's union return (tuple only if return_indices=True)."""
    result = sm.utils.subsample(sequences, n_samples=n_samples, seed=seed)
    return result[0] if isinstance(result, tuple) else result


def filter_standard_aa(sequences: list[str], max_len: int = 65) -> list[str]:
    """Filter to standard 20 amino acids and <= max_len residues."""
    return [s for s in sequences if set(s) <= STANDARD_AMINO_ACIDS and len(s) <= max_len]


def compute_synthesizability(sequences: list[str]) -> dict[str, float]:
    """
    Check HydrAMP Filtering-1 biological synthesizability constraints:
      1. No cysteine (C)
      2. No 3 identical consecutive residues
      3. No 3 consecutive hydrophobic residues
      4. No >= 3 positive residues (K, R, H) in any 5-residue sliding window
    """
    total = len(sequences)
    if total == 0:
        return {"pass_rate": 0.0, "no_cys_rate": 0.0}

    no_cys_count = 0
    passed_all = 0

    for seq in sequences:
        # Check Cys
        has_cys = "C" in seq
        if not has_cys:
            no_cys_count += 1

        # Check triple repeats
        has_triple_repeat = any(seq[i] == seq[i + 1] == seq[i + 2] for i in range(len(seq) - 2))

        # Check triple hydrophobic
        has_triple_hydro = any(
            seq[i] in HYDROPHOBIC_RESIDUES
            and seq[i + 1] in HYDROPHOBIC_RESIDUES
            and seq[i + 2] in HYDROPHOBIC_RESIDUES
            for i in range(len(seq) - 2)
        )

        # Check charge clusters: >= 3 positive residues in any 5-window
        has_charge_cluster = False
        if len(seq) >= 5:
            for i in range(len(seq) - 4):
                window = seq[i : i + 5]
                if sum(r in POSITIVE_RESIDUES for r in window) >= 3:
                    has_charge_cluster = True
                    break
        else:
            if sum(r in POSITIVE_RESIDUES for r in seq) >= 3:
                has_charge_cluster = True

        if (
            not has_cys
            and not has_triple_repeat
            and not has_triple_hydro
            and not has_charge_cluster
        ):
            passed_all += 1

    return {
        "pass_rate": float(passed_all / total),
        "no_cys_rate": float(no_cys_count / total),
    }


def compute_authenticity_and_novelty(
    sequences: list[str],
    reference_set: set[str],
) -> dict[str, Any]:
    """Compute exact matches and reference overlap."""
    total = len(sequences)
    if total == 0 or len(reference_set) == 0:
        return {"exact_matches": 0, "exact_match_pct": 0.0}

    exact_matches = sum(1 for s in sequences if s in reference_set)
    return {
        "exact_matches": exact_matches,
        "exact_match_pct": float(100.0 * exact_matches / total),
    }


class ESMCEmbedder:
    """EvolutionaryScale ESM Cambrian (ESM-C) embedder for seqme metrics."""

    def __init__(self, model_name: str = "esmc_300m", device: str = "cpu", verbose: bool = False):
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self._model: ESMC | None = None

    def _ensure_loaded(self):
        if self._model is None:
            from esm.models.esmc import ESMC

            print(f"Loading {self.model_name} on {self.device}...")
            model = ESMC.from_pretrained(self.model_name).to(self.device)
            model.eval()
            self._model = model

    def __call__(self, sequences: list[str]) -> np.ndarray:
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig

        self._ensure_loaded()
        assert self._model is not None
        embeddings = []
        with torch.inference_mode():
            for s in sequences:
                prot = ESMProtein(sequence=s)
                t = self._model.encode(prot)
                out = self._model.logits(t, LogitsConfig(sequence=True, return_embeddings=True))
                assert out.embeddings is not None
                res_emb = out.embeddings[0, 1 : len(s) + 1, :].mean(dim=0)
                embeddings.append(res_emb.cpu().float().numpy())
        return np.array(embeddings, dtype=np.float32)

    embed = __call__


def run_mmseqs_screen(
    candidate_fasta: Path,
    marlys_fasta: Path,
    min_identity: float = 0.80,
    min_query_coverage: float = 0.80,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Run MMseqs2 easy-search against MarLys database with short-peptide calibrated settings:
    -s 7.5 -k 5 -e 10000 --max-seqs 300
    Format: query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov

    Returns:
        (violations, exact_rediscoveries)
        where:
        - violations: query -> dict of match info with fident > min_identity AND qcov >= min_query_coverage
        - exact_rediscoveries: query -> dict of match info with fident >= 0.999 AND qcov >= 0.999
    """
    if not shutil.which("mmseqs"):
        print("WARNING: mmseqs executable not found on PATH. Skipping MMseqs2 alignment screen.")
        return {}, {}
    if not marlys_fasta.exists():
        print(
            f"WARNING: MarLys reference FASTA not found: {marlys_fasta}. Skipping MMseqs2 alignment screen."
        )
        return {}, {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        out_m8 = tmp_p / "aln.m8"
        cmd = [
            "mmseqs",
            "easy-search",
            str(candidate_fasta),
            str(marlys_fasta),
            str(out_m8),
            str(tmp_p / "tmp"),
            "-s",
            "7.5",
            "-k",
            "5",
            "-e",
            "10000",
            "--max-seqs",
            "300",
            "--format-output",
            "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov",
            "-v",
            "0",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"WARNING: MMseqs2 search failed: {e}")
            return {}, {}

        violations: dict[str, dict[str, Any]] = {}
        exact_rediscoveries: dict[str, dict[str, Any]] = {}
        if out_m8.exists():
            for line in out_m8.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 14:
                    q, t = parts[0], parts[1]
                    fident = float(parts[2])
                    alnlen = int(parts[3])
                    evalue = float(parts[10])
                    bitscore = float(parts[11])
                    qcov = float(parts[12])
                    tcov = float(parts[13])

                    # Exact full-query rediscovery: fident >= 0.999 & qcov >= 0.999
                    if (fident >= 0.999 and qcov >= 0.999) and (
                        q not in exact_rediscoveries or fident > exact_rediscoveries[q]["identity"]
                    ):
                        exact_rediscoveries[q] = {
                            "target": t,
                            "identity": fident,
                            "alnlen": alnlen,
                            "evalue": evalue,
                            "bitscore": bitscore,
                            "qcov": qcov,
                            "tcov": tcov,
                        }

                    # Novelty violation: fident > min_identity & qcov >= min_query_coverage
                    if (fident > min_identity and qcov >= min_query_coverage) and (
                        q not in violations or fident > violations[q]["identity"]
                    ):
                        violations[q] = {
                            "target": t,
                            "identity": fident,
                            "alnlen": alnlen,
                            "evalue": evalue,
                            "bitscore": bitscore,
                            "qcov": qcov,
                            "tcov": tcov,
                        }
        return violations, exact_rediscoveries


def run_harness(
    candidate_paths: dict[str, Path],
    ref_amp_path: Path,
    ref_bg_path: Path | None,
    ref_antibacterial_path: Path | None,
    marlys_path: Path | None,
    thirdparty_dir: Path,
    sample_size: int = 1000,
    seed: int = 42,
    skip_ampredictor: bool = False,
    skip_esmc: bool = False,
    esm2_checkpoint: str = "t33_650M",
    esmc_model: str = "esmc_300m",
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full Phase-1 evaluation harness across all candidates."""
    print("=" * 80)
    print("AMP CHALLENGE 2027: PHASE-1 EVALUATION HARNESS")
    print(
        f"Seed: {seed} | Sample size: {sample_size} | ESM-2: {esm2_checkpoint} | ESM-C: {esmc_model} | Device: {device}"
    )
    print("=" * 80)

    # 1. Load candidate sequences
    raw_datasets: dict[str, list[str]] = {}
    clean_datasets: dict[str, list[str]] = {}
    sub_datasets: dict[str, list[str]] = {}

    for name, path in candidate_paths.items():
        if not path.exists():
            print(f"WARNING: File not found: {path} (skipping {name})")
            continue
        seqs = read_fasta_sequences(path)
        clean = filter_standard_aa(seqs)
        raw_datasets[name] = seqs
        clean_datasets[name] = clean
        sub_datasets[name] = (
            subsample_sequences(clean, n_samples=sample_size, seed=seed)
            if len(clean) > sample_size
            else clean
        )
        print(f"Loaded '{name}': {len(seqs):,} raw seqs, {len(clean):,} clean standard-AA seqs")

    # 2. Load references
    ref_amp_clean = filter_standard_aa(read_fasta_sequences(ref_amp_path))
    ref_amp_sub = subsample_sequences(ref_amp_clean, n_samples=sample_size, seed=seed)
    ref_amp_set = set(ref_amp_clean)
    print(f"Loaded Reference AMPs ({ref_amp_path.name}): {len(ref_amp_clean):,} seqs")

    ref_antibacterial_set: set[str] = set()
    if ref_antibacterial_path and ref_antibacterial_path.exists():
        ref_ab_clean = filter_standard_aa(read_fasta_sequences(ref_antibacterial_path))
        ref_antibacterial_set = set(ref_ab_clean)
        print(
            f"Loaded Reference Antibacterial ({ref_antibacterial_path.name}): {len(ref_ab_clean):,} seqs"
        )

    ref_bg_sub: list[str] | None = None
    if ref_bg_path and ref_bg_path.exists():
        ref_bg_clean = filter_standard_aa(read_fasta_sequences(ref_bg_path))
        ref_bg_sub = subsample_sequences(ref_bg_clean, n_samples=sample_size, seed=seed)
        print(f"Loaded Background Reference ({ref_bg_path.name}): {len(ref_bg_clean):,} seqs")

    marlys_clean = (
        filter_standard_aa(read_fasta_sequences(marlys_path))
        if marlys_path and marlys_path.exists()
        else []
    )
    marlys_set = set(marlys_clean)
    if marlys_path is not None and marlys_set:
        print(f"Loaded MarLys Reference AMPs ({marlys_path.name}): {len(marlys_clean):,} seqs")

    # 3. Setup models and cache
    print(f"\nInitializing PLM embedding model ({esm2_checkpoint}) and physicochemical models...")
    esm2_checkpoint_enum = getattr(sm.models.ESM2Checkpoint, esm2_checkpoint)
    esm2_batch_size = (
        32 if "650M" in esm2_checkpoint else (64 if "150M" in esm2_checkpoint else 256)
    )
    esm2_model = sm.models.ESM2(
        model_name=esm2_checkpoint_enum,
        batch_size=esm2_batch_size,
        device=device,
        verbose=False,
    )

    models_dict: dict[str, Any] = {
        "esm2": esm2_model,
        "charge": sm.models.Charge(),
        "moment": sm.models.HydrophobicMoment(),
        "hydro": sm.models.Hydrophobicity(),
    }
    if not skip_esmc:
        models_dict["esmc"] = ESMCEmbedder(model_name=esmc_model, device=device)

    cache = sm.Cache(models=models_dict)

    # 4. Compute Full-Library Sequence and Synthesizability Metrics
    print("\nComputing Sequence-level, Synthesizability, and Novelty metrics...")
    results: dict[str, dict[str, Any]] = {}

    for name, seqs in clean_datasets.items():
        raw_count = len(raw_datasets[name])
        clean_count = len(seqs)
        uniqueness = float(len(set(seqs)) / clean_count) if clean_count > 0 else 0.0

        lengths = [len(s) for s in seqs]
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        std_len = float(np.std(lengths)) if lengths else 0.0

        # Synthesizability on full library
        synth = compute_synthesizability(seqs)

        # Authenticity / exact match vs references
        exact_ab = compute_authenticity_and_novelty(seqs, ref_antibacterial_set)
        exact_dbaasp = compute_authenticity_and_novelty(seqs, ref_amp_set)
        exact_marlys = compute_authenticity_and_novelty(seqs, marlys_set)

        results[name] = {
            "raw_count": raw_count,
            "clean_count": clean_count,
            "uniqueness": uniqueness,
            "mean_length": mean_len,
            "std_length": std_len,
            "filtering1_pass_pct": synth["pass_rate"] * 100.0,
            "no_cys_pct": synth["no_cys_rate"] * 100.0,
            "exact_antibacterial_matches": exact_ab["exact_matches"],
            "exact_antibacterial_pct": exact_ab["exact_match_pct"],
            "exact_dbaasp_matches": exact_dbaasp["exact_matches"],
            "exact_dbaasp_pct": exact_dbaasp["exact_match_pct"],
            "exact_marlys_matches": exact_marlys["exact_matches"],
            "exact_marlys_pct": exact_marlys["exact_match_pct"],
        }

        # MMseqs2 novelty screening and exact rediscovery against MarLys
        if marlys_path and marlys_path.exists():
            cand_fasta = candidate_paths[name]
            mmseqs_viol, mmseqs_exact = run_mmseqs_screen(
                cand_fasta, marlys_path, min_identity=0.80, min_query_coverage=0.80
            )
            violating = len(mmseqs_viol)
            exact_count = len(mmseqs_exact)
            compliance_pct = (
                100.0 * (clean_count - violating) / clean_count if clean_count > 0 else 0.0
            )
            exact_pct = 100.0 * exact_count / clean_count if clean_count > 0 else 0.0
            results[name]["marlys_novelty_compliant_pct"] = compliance_pct
            results[name]["marlys_violating_count"] = violating
            results[name]["marlys_exact_matches"] = exact_count
            results[name]["marlys_exact_pct"] = exact_pct
            results[name]["marlys_violating_hits"] = mmseqs_viol
            results[name]["marlys_exact_hits"] = mmseqs_exact
            print(
                f"  MarLys Screen for '{name}': {compliance_pct:.1f}% novel ({violating} violations >80% id & >=80% cov), "
                f"{exact_count} exact rediscoveries ({exact_pct:.2f}%)"
            )
        else:
            results[name]["marlys_novelty_compliant_pct"] = None
            results[name]["marlys_violating_count"] = None
            results[name]["marlys_exact_matches"] = exact_marlys["exact_matches"]
            results[name]["marlys_exact_pct"] = exact_marlys["exact_match_pct"]
            results[name]["marlys_violating_hits"] = {}
            results[name]["marlys_exact_hits"] = {}

    # 5. Compute seqme metrics on subsamples
    print("\nComputing seqme Diversity, PLM Embeddings (ESM-2 & ESM-C), and Conformity metrics...")

    def cached_model(model_name: str) -> Callable[[list[str]], np.ndarray]:
        return cast(Callable[[list[str]], np.ndarray], cache.model(model_name))

    seqme_metrics: list[Any] = [
        sm.metrics.Diversity(k=200, name="Diversity"),
        sm.metrics.FBD(reference=ref_amp_sub, embedder=cached_model("esm2"), name="FBD (AMPs)"),
        sm.metrics.MMD(reference=ref_amp_sub, embedder=cached_model("esm2"), name="MMD (AMPs)"),
        sm.metrics.Precision(
            n_neighbors=12,
            reference=ref_amp_sub,
            embedder=cached_model("esm2"),
            name="Precision (AMPs)",
            strict=False,
        ),
        sm.metrics.Recall(
            n_neighbors=12,
            reference=ref_amp_sub,
            embedder=cached_model("esm2"),
            name="Recall (AMPs)",
            strict=False,
        ),
        sm.metrics.ConformityScore(
            reference=ref_amp_sub,
            predictors=[cached_model("charge"), cached_model("moment")],
            name="Conformity",
        ),
    ]

    if not skip_esmc:
        seqme_metrics.extend(
            [
                sm.metrics.FBD(
                    reference=ref_amp_sub, embedder=cached_model("esmc"), name="ESMC FBD (AMPs)"
                ),
                sm.metrics.MMD(
                    reference=ref_amp_sub, embedder=cached_model("esmc"), name="ESMC MMD (AMPs)"
                ),
            ]
        )

    if ref_bg_sub is not None:
        seqme_metrics.append(
            sm.metrics.FBD(
                reference=ref_bg_sub, embedder=cached_model("esm2"), name="FBD (UniProt)"
            )
        )

    seqme_df = sm.evaluate(sub_datasets, seqme_metrics)

    assert isinstance(seqme_df.columns, pd.MultiIndex)
    for name in sub_datasets:
        for col in seqme_df.columns.levels[0]:
            val = seqme_df.loc[name, (col, "value")]
            dev = seqme_df.loc[name, (col, "deviation")]
            results[name][f"seqme_{col}"] = float(val) if not np.isnan(val) else None
            results[name][f"seqme_{col}_std"] = float(dev) if not np.isnan(dev) else None

    # Physicochemical distribution metrics
    for name, sub_seqs in sub_datasets.items():
        charges = cache.model("charge")(sub_seqs)
        moments = cache.model("moment")(sub_seqs)
        hydros = cache.model("hydro")(sub_seqs)
        results[name]["mean_charge"] = float(np.mean(charges))
        results[name]["mean_moment"] = float(np.mean(moments))
        results[name]["mean_hydrophobicity"] = float(np.mean(hydros))

    # 6. AMPredictor surrogate MIC prediction
    if not skip_ampredictor:
        ampredictor_dir = thirdparty_dir / "ampredictor"
        if not ampredictor_dir.exists():
            print(f"\nSetting up AMPredictor plugin in {ampredictor_dir}...")
            ampredictor_model = sm.models.ThirdPartyModel(
                entry_point="ampredictor.predict:predict",
                path=ampredictor_dir,
                url="https://github.com/szczurek-lab/seqme-ampredictor",
            )
        else:
            ampredictor_model = sm.models.ThirdPartyModel(
                entry_point="ampredictor.predict:predict",
                path=ampredictor_dir,
            )

        print("\nEvaluating AMPredictor MIC on candidate samples...")
        for name, sub_seqs in sub_datasets.items():
            print(f"  Scoring '{name}' (n={len(sub_seqs)})...")
            t0 = time.time()
            mics = ampredictor_model(sub_seqs)
            elapsed = time.time() - t0
            valid_mics = mics[~np.isnan(mics)]
            med_mic = float(np.median(valid_mics))
            mean_mic = float(np.mean(valid_mics))
            pct_active_16 = float(100.0 * np.mean(valid_mics <= 16.0))
            pct_active_32 = float(100.0 * np.mean(valid_mics <= 32.0))
            q25 = float(np.percentile(valid_mics, 25))
            q75 = float(np.percentile(valid_mics, 75))

            results[name]["ampredictor_median_mic"] = med_mic
            results[name]["ampredictor_mean_mic"] = mean_mic
            results[name]["ampredictor_pct_le_16um"] = pct_active_16
            results[name]["ampredictor_pct_le_32um"] = pct_active_32
            results[name]["ampredictor_q25_mic"] = q25
            results[name]["ampredictor_q75_mic"] = q75
            print(
                f"    -> Median MIC: {med_mic:.2f} uM | "
                f"Active <=16 uM: {pct_active_16:.1f}% | "
                f"Active <=32 uM: {pct_active_32:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )
    else:
        print("\nSkipping AMPredictor evaluation as requested.")

    # 7. Convert to DataFrame
    df_summary = pd.DataFrame.from_dict(results, orient="index")
    return df_summary, results


def format_markdown_report(df: pd.DataFrame, sample_size: int | None = None) -> str:
    """Format the summary table: one row per candidate, columns are the Phase 1 metric families."""
    lines = [
        "# Phase 1 evaluation harness",
        "",
        (
            "Columns map onto the four Phase 1 metric families. Not implemented: the MBC-Attention and DeepAMP surrogates and "
            "the clustering-based coverage metric; activity is AMPredictor only."
        ),
        "",
    ]

    # Build primary table
    table_cols = {
        "clean_count": "Count",
        "mean_length": "Length",
        "uniqueness": "Uniqueness↑",
        "seqme_Diversity": "Diversity↑",
        "ampredictor_median_mic": "AMPredictor Median MIC (µM)↓",
        "ampredictor_pct_le_16um": "Active ≤16µM (%)↑",
        "seqme_FBD (AMPs)": "ESM2 FBD↓",
        "seqme_MMD (AMPs)": "ESM2 MMD↓",
        "seqme_ESMC FBD (AMPs)": "ESMC FBD↓",
        "seqme_ESMC MMD (AMPs)": "ESMC MMD↓",
        "seqme_Precision (AMPs)": "Precision↑",
        "seqme_Recall (AMPs)": "Recall↑",
        "seqme_Conformity": "Conformity↑",
        "filtering1_pass_pct": "Filtering-1 pass (%)↑",
        "marlys_exact_matches": "MarLys Exact Rediscovery↓",
        "marlys_novelty_compliant_pct": "Novelty Compliance (≤80%)↑",
    }

    headers = ["Model"] + [v for v in table_cols.values()]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model_name, row in df.iterrows():
        row_vals = [f"**{model_name}**"]
        row_dict = dict(row)
        for col_key in table_cols:
            val = row_dict.get(col_key)
            if val is None or (pd.isna(val) is True):
                row_vals.append("—")
            elif col_key == "uniqueness":
                row_vals.append(f"{val:.3f}")
            elif col_key == "mean_length":
                std_val = row_dict.get("std_length", 0.0)
                row_vals.append(f"{val:.1f}±{std_val:.1f}")
            elif col_key in [
                "seqme_Diversity",
                "seqme_Conformity",
                "seqme_Precision (AMPs)",
                "seqme_Recall (AMPs)",
            ]:
                row_vals.append(f"{val:.3f}")
            elif col_key == "clean_count":
                row_vals.append(f"{int(val):,}" if isinstance(val, (int, float, str)) else f"{val}")
            elif col_key == "marlys_exact_matches":
                pct = row_dict.get("marlys_exact_pct", 0.0)
                int_part = int(val) if isinstance(val, (int, float, str)) else val
                row_vals.append(f"{int_part} ({pct:.2f}%)")
            elif col_key == "marlys_novelty_compliant_pct":
                viol = row_dict.get("marlys_violating_count", 0)
                if isinstance(viol, (int, float)) and viol > 0:
                    row_vals.append(f"{val:.1f}% ({int(viol)} fail)")
                else:
                    row_vals.append(f"{val:.1f}%")
            elif "pct" in col_key or col_key == "ampredictor_pct_le_16um":
                row_vals.append(f"{val:.1f}%")
            elif "mic" in col_key or "FBD" in col_key or "MMD" in col_key:
                row_vals.append(f"{val:.2f}")
            else:
                row_vals.append(f"{val}")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 Evaluation Harness for AMP Challenge 2027"
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["Ours=generate/library.fasta", "Ours-Top100=generate/top.fasta"],
        help="List of Name=Path pairs for candidate fasta files.",
    )
    parser.add_argument(
        "--ref-amp",
        type=Path,
        default=Path("data/dbaasp.fasta"),
        help="FASTA path for known AMP reference (default: data/dbaasp.fasta).",
    )
    parser.add_argument(
        "--ref-bg",
        type=Path,
        default=None,
        help="Optional FASTA of generic UniProt peptides (8-50 aa) for the background-reference metrics.",
    )
    parser.add_argument(
        "--ref-antibacterial",
        type=Path,
        default=Path("data/antibacterial.fasta"),
        help="FASTA path for competition antibacterial reference (default: data/antibacterial.fasta).",
    )
    parser.add_argument(
        "--marlys-fasta",
        type=Path,
        default=None,
        help="Optional MarLys FASTA for the diagnostic MMseqs2 screen (needs mmseqs on PATH).",
    )
    parser.add_argument(
        "--thirdparty-dir",
        type=Path,
        default=Path("vendor"),
        help="Directory holding the vendored AMPredictor plugin (default: vendor).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for computationally heavy PLM/GNN evaluations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling (default: 42).",
    )
    parser.add_argument(
        "--esm2-checkpoint",
        choices=["t33_650M", "t30_150M", "t6_8M"],
        default="t33_650M",
        help="ESM2 checkpoint to use for embedding metrics (default: t33_650M).",
    )
    parser.add_argument(
        "--esmc-model",
        default="esmc_300m",
        help="ESM-C checkpoint/model name (default: esmc_300m).",
    )
    parser.add_argument(
        "--skip-ampredictor",
        action="store_true",
        help="Skip running the AMPredictor surrogate model.",
    )
    parser.add_argument(
        "--skip-esmc",
        action="store_true",
        help="Skip running the ESM-C embedder.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for PyTorch models (e.g. cpu, mps, cuda; default: cpu).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("metrics/phase1_results.json"),
        help="Output JSON file path (default: metrics/phase1_results.json).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("metrics/PHASE1_REPORT.md"),
        help="Output Markdown report path (default: metrics/PHASE1_REPORT.md).",
    )

    args = parser.parse_args()

    candidate_paths: dict[str, Path] = {}
    for item in args.candidates:
        if "=" in item:
            name, path_str = item.split("=", 1)
        else:
            path = Path(item)
            name = path.stem
            path_str = item
        candidate_paths[name] = Path(path_str)

    df_summary, raw_results = run_harness(
        candidate_paths=candidate_paths,
        ref_amp_path=args.ref_amp,
        ref_bg_path=args.ref_bg,
        ref_antibacterial_path=args.ref_antibacterial,
        marlys_path=args.marlys_fasta,
        thirdparty_dir=args.thirdparty_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        skip_ampredictor=args.skip_ampredictor,
        skip_esmc=args.skip_esmc,
        esm2_checkpoint=args.esm2_checkpoint,
        esmc_model=args.esmc_model,
        device=args.device,
    )

    # Ensure output directory exists
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\nWrote full metrics JSON to: {args.output_json}")

    md_report = format_markdown_report(df_summary, sample_size=args.sample_size)
    with open(args.output_md, "w") as f:
        f.write(md_report)
    print(f"Wrote Markdown report to: {args.output_md}")

    print("\n" + md_report)


if __name__ == "__main__":
    main()
