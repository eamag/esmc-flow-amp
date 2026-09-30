# Submission Writeup: AMP Challenge 2027

## Title (63/80)

```text
Charge-steered ESM-C masked flow with held-out library selection
```

## Subtitle (103/140)

```text
A conditioned peptide flow model, a 396k-draw pool, and a 50k library chosen against held-out references
```

## Submission track

`Computational track`.

## Project description

### One line

We fine-tuned ESM-C 300M into a masked flow model that can be steered by charge, length and
hydrophobicity codes, drew 396k peptides from it, and selected the 50,000-sequence library and the
top-100 against references it never saw.

### What we found first

Steering our earlier diffusion pair by net charge helps, but a single high-charge target hits a
geometric wall: the Filtering-1 rule rejects any 5-residue window with ≥ 3 K/R, which caps a
20-mer near charge +7.

### The model

- **Backbone:** ESM-C 300M with a masked-flow objective. The sequence is `[BOS][controls][seq][EOS]`;
  the controls are charge, length and hydrophobicity buckets, dropped to a NULL token 15% of the
  time during training so classifier-free guidance is available. The calibrated guidance weight
  is w = 1.0, i.e. plain conditional sampling.
- **Curriculum (three stages, Tesla T4, fp16):** broad peptide prior (5.6M rows) → AMP-like (889k)
  → measured-potent only (4,460 rows).
- **Conditioning check:** Spearman(requested charge bucket, realised charge) went from +0.40 to
  +1.00, and the realised charge spread from 4.8 to 26.6 units. The control embeddings are trained
  (displacement 0.52–1.20 against a random-walk bound of 0.107)

### Depth and selection

1. **Depth:** ten parts of 30,000 draws on an Apple M5 Pro (Metal, fp16, one process per charge
   bucket), plus the initial pool: a merged pool of 396,334 unique sequences, 66% predicted
   active, 10.6% Filtering-1 pass rate.
2. **Selection score:**
   `2·z(Conformity) + 1.8·active + 0.5·Filtering-1 + 4·z(ESM-2 density ratio) + 1·z(Diversity proxy)`,
   with a hard Precision filter (≥ 0.97), greedy and stratified by the DBAASP length histogram,
   near-duplicates (Levenshtein ratio ≥ 0.90) eliminated.
3. **Held-out references:** DBAASP is split 50/50. Conformity KDE, density ratio and the Precision
   proxy are fitted on one half; every evaluation scores draws from the other half. Selecting 50k
   from 396k (a 13% ratio) closes the Conformity gap; from an earlier 94k pool (a 53% ratio) the
   same selection could not reach this Conformity level.

### Results

| | Div | median MIC (µM) | active ≤16 µM | ESM-2 FBD / MMD | ESM-C FBD / MMD | Prec | Recall | Conformity | Filtering-1 | novelty compliance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ours** | 0.826 | **9.07** | **79.8%** | **1.55 / 2.81** | **0.05 / 0.08** | **0.996** | **0.865** | **0.808** | **20.2%** | **99.4%** |
| DiffTabPFN baseline | 0.828 | 13.80 | 54.9% | 2.38 / 8.75 | 0.11 / 0.58 | 0.995 | 0.845 | 0.752 | 9.2% | 73.1% |

On held-out reference draws (10 paired draws), we beat the earlier pair in 100% of draws
on FBD and MMD in both embedders against the AMP reference, and on FBD/MMD against generic peptides
in ESM-C; Conformity 0.805 against 0.749 (100% win rate).
But: ESM-C AMP Precision (0.990 against 0.992) and ESM-2 generic-peptide MMD (21.1 against 20.8).
Sequence diversity ties at 0.827.

### The top-100

Ranked by TabPFN with measured MIC in context, because on 752 held-out measured MICs it is the better
ranker (top-100: 85% ≤ 16 µM and 60% ≤ 4 µM, against 74% and 42% for AMPredictor, which the
Phase 1 harness uses). Gates: Filtering-1, novelty (max Levenshtein ratio 0.783 against the 39,448
antibacterial references, under the 0.80 limit), AMPredictor ≤ 16 µM as an agreement check,
and predicted hemolysis (TabPFN probability p < 0.35). It is a strict subset of the library.

| | TabPFN active ≤16 µM | AMPredictor median MIC | hemolysis p mean | p > 0.55 |
| --- | --- | --- | --- | --- |
| **Ours** | **100%** | **5.34 µM** | **0.286** | **0%** |
| Earlier pair's top-100 | 100% | 6.63 µM | 0.313 | 9% |

### What we got wrong

- The first flow model lost badly (28.2% active) and we wrote it off. Problems: a sampler that
  deleted 30–60% of each sequence, no guidance at sampling time, no control embeddings, and a
  finetune set that was 0.39% real AMPs.
- We tuned guidance to w = 8 against a bad conditioning signal; the calibrated value is 1.0.

### Limitations

- **The library is predicted more hemolytic than the earlier pair's** (TabPFN p mean 0.43 against
  0.33; 26% above 0.55 against 17%). Only the top-100 is gated for it. This is the biggest risk to
  Phase 2.
- **Potency is predicted.** AMPredictor selected the library, so its numbers are optimistic;
  TabPFN (not used in library selection) says 77% against 59% active ≤ 16 µM. The oracle is a
  ranker with Spearman 0.36 on measured MIC. No wet-lab data for either pair.
- **Clustering coverage of the reference is lower than the earlier pair's** (DBAASP clusters
  covered at identity 0.6: 7.9% against 14.1%; at 0.7: 1.6% against 5.6%), the flip side of being
  far more novel. Our library forms more clusters of its own (855 against 566 at 0.6). The rules
  give no formula, so we do not know which reading is scored.
- Two Phase 1 surrogates (MBC-Attention, DeepAMP) and the clustering-based coverage metric are not
  implemented by us
- Precision and Diversity were optimised through proxies

### Disclosure

AI coding assistants were used in preparing this submission, per the NeurIPS Main Track
Handbook. The team is responsible for all submitted content. No non-public data was used.

- **Generator:** ESM-C 300M (`EvolutionaryScale/esmc-300m-2024-12`, Cambrian Open licence, which
  covers fine-tuning), fine-tuned by us in three stages. The stage C weights are published at
  <https://huggingface.co/eamag/esmc-flow-amp>, not in this repository.
- **Training tables** (sources, row counts and licences in `data/plm/README.md`): stage A 5.6M
  rows, UniProt slices, protein tiling, ToxProt, AMPSphere (CC-BY-4.0) and the curated databases
  below; stage B 889,293 rows, the AMPSphere and curated-database subset of stage A: MarLys (CC0),
  OmegAMP positives (MIT), NeuroPep, ConoServer, CPPsite 2.0 (academic terms, cited), DBAASP
  (redistribution with acknowledgement), GRAMPA (no upstream licence file), DRAMP general and
  clinical (CC-BY-4.0); stage C 4,460 rows of measured potent
  peptides from `data/oracle/peptide_mic_targets_v3.parquet` (DBAASP and GRAMPA measurements).
  The organisers' `data/antibacterial.fasta` was used only to exclude exact matches and to
  measure novelty, never as a training label.
- **DBAASP acknowledgement:** "Data were obtained from the DBAASP (<https://dbaasp.org>) which is
  an open-access AMP data resource supported by I. Beritashvili Center of Experimental Biomedicine
  (IBCEB), Tbilisi, Georgia and NIAID OCICB, Bethesda, MD."
- **Scorers and embedders (not trained by us):** ESM-2 650M (`t33_650M`) and ESM-C 300M as Phase 1
  embedders; AMPredictor (organiser harness; the vendored copy carries no LICENSE file), which
  selected the library; TabPFN, trained on the measured-MIC and hemolysis tables in
  `data/oracle/`, which ranked the top-100 and gated hemolysis.

### Reproducibility

The submitted pair is committed at `generate/{library,top}.fasta` (also `data/final/`).
`uv sync && uv run generate` validates both against the competition rules and copies them byte for
byte, in seconds and with no network access. The single-path pipeline that produces such a pair
(data → training → sampling → scoring → held-out selection → top-100) is in `src/esmc_flow_amp/`,
driven by `scripts/run_pipeline.sh` and described step by step in `REPRODUCE.md`.

Re-running it will not reproduce the files exactly. The shipped pool combines an earlier scale-up
pool with ten new parts of 30,000 draws, sampling is device- and seed-dependent, and the TabPFN
ranker is a hosted model. A rerun of the top-100 step on the shipped library shared 73 of the 100
sequences and scored slightly worse on AMPredictor median MIC (6.29 against 5.34 µM).
Stage A and B tables (5.6M and 889k rows) are downloaded from
<https://huggingface.co/datasets/eamag/amp-plm>; stage C is rebuilt here row for row.
`REPRODUCE.md` lists what is and is not reproducible.

## Project links

One public repository link (organisers need read access: `@RasmusML`, `@szymczakpau`):

```text
https://github.com/eamag/esmc-flow-amp
```
