# Phase-1 Evaluation Harness: Benchmark & Comparison

Benchmark of the baseline pair against the shipped submission library.

## 1. Summary table

Columns map onto the four Phase-1 metric families defined in the competition proposal §1.5.

| Model | Count | Length | Uniqueness↑ | Diversity↑ | AMPredictor Median MIC (µM)↓ | Active ≤16µM (%)↑ | ESM2 FBD↓ | ESM2 MMD↓ | ESMC FBD↓ | ESMC MMD↓ | Precision↑ | Recall↑ | Conformity↑ | Filtering-1 pass (%)↑ | MarLys Exact Rediscovery↓ | Novelty Compliance (≤80%)↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **FINAL_o2k0d1p97 (Ours)** | 50,000 | 22.3±9.9 | 1.000 | 0.826 | 9.07 | 79.8% | 1.55 | 2.81 | 0.05 | 0.08 | 0.996 | 0.865 | 0.808 | 20.2% | 1 (0.00%) | 99.4% (301 fail) |
| **Baseline (DiffTabPFN)** | 50,000 | 18.0±9.0 | 1.000 | 0.828 | 13.80 | 54.9% | 2.38 | 8.75 | 0.11 | 0.58 | 0.995 | 0.845 | 0.752 | 9.2% | 358 (0.72%) | 73.1% (13456 fail) |

## 2. Findings

### A. Published against measured numbers

- The "5.5-point gap":
  - The published `seqme` paper (Table 1 and Cell 52 of `1_benchmark.ipynb`) reports only the
    median predicted MIC (11.8 µM for HydrAMP, 12.4 µM for AMP-Diffusion). It does not
    publish an active-fraction table.
  - The ~64.6% and ~63.4% figures in earlier draft prose were ungrounded estimates, not literature targets.
  - Measured on the candidate fastas in this run, under the competition's MIC ≤ 16.0 µM threshold:
    - FINAL_o2k0d1p97 (Ours): 79.8% active (median 9.07 µM)
    - Baseline (DiffTabPFN): 54.9% active (median 13.80 µM)
  - Sampled with `--sample-size` = 1000; on a ~50% proportion that is ±3.10 pp at 95%,
    so differences below a few points are not resolvable in a single run.

### B. Embedding spaces: ESM-2 (650M) and ESM-C (300M)

- Phase 1 computes embedding metrics in two PLM spaces: ESM-2 (`t33_650M`) and ESM-C (`esmc_300m`).
- Both models evaluate distributional divergence (FBD, MMD, Precision, Recall) against
  the DBAASP reference set.

### C. Novelty: the criterion that decides the submission

- The competition's novelty rule is implemented in `scripts/verify_submission.py` and is
  not an alignment screen:
  - `generate/top.fasta` fails if `Levenshtein.ratio(seq, ref) > 0.80` against any of
    the 39,448 sequences in `data/antibacterial.fasta`.
  - `generate/library.fasta` must have zero exact set intersection with that same reference;
    near-duplicates are permitted in the 50k library.
- Novelty must therefore be enforced with `Levenshtein.ratio` against `data/antibacterial.fasta`,
  and at selection time for the top-100.
- MMseqs2 is not used in any `generate` code path. Earlier reports framed novelty as
  MMseqs2 identity/coverage violations against MarLys (103,200 sequences); that framing is
  withdrawn. It measured a different reference set with a different metric and did not
  predict verifier outcomes.
- `run_mmseqs_screen` is retained only as an optional diagnostic, and its output is not
  a submission criterion.

### D. Surrogate potency and reward hacking (Baseline-KP)

- Baseline-KP scores an artificial 7.96 µM median MIC (100% active ≤16 µM) by outputting
  repetitive poly-cationic K/P strings that exploit charge bias in the GNN/ESM head.
- However, Baseline-KP collapses on PLM embedding fidelity (ESM-2 FBD >90, ESM-C FBD collapsed)
  and biological property conformity (0.002 vs 0.439–0.871).
- Design rule: AMPredictor median MIC is a floor or gate, not an objective to maximise.
  Optimizing median MIC alone generates toxic poly-K garbage. With Phase 1 aggregation weights
  held out until closing, submissions must maintain balanced strength across all four families.

### E. Where the submission stands

- Current submission pool `generate/library.fasta`: 79.8% active, median 9.07 µM.
- Current submission top-100 `generate/top.fasta`: 100% active, median 5.34 µM.
- Ranking a pool lifts the top-100 well above its parent distribution, but selection
  cannot fix an unconditioned generator: the ceiling is set by what the pool contains.
- AMPredictor is not credible at the extreme, which is where top-100 selection operates.
  Treat its median MIC as a floor/gate, never as an objective to maximise, and prefer
  calibrated or uncertainty-aware ranking for the top-100.
