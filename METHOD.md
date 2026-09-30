# Method, selection and data disclosure

How the submitted pair was made. Code paths are in `src/esmc_flow_amp/`; commands are in [`REPRODUCE.md`](REPRODUCE.md).

## 1. Generator

ESM-C 300M (`EvolutionaryScale/esmc-300m-2024-12`) fine-tuned as a masked-flow
(discrete unmasking) generator. Each training example is

```text
[BOS] [charge] [length] [hydro] [residues ...] [EOS]
```

The three control tokens are bucket ids (0-6) or a learned NULL id (7). Only residue positions are
masked and only they contribute to the cross-entropy loss; the mask rate is drawn uniformly per
sequence.

| Control | Attribute | Edges (bucket = number of edges the value exceeds) |
| --- | --- | --- |
| `charge` | Bjellqvist net charge at pH 7 | 0, 2, 4, 6, 8, 11 |
| `length` | residues | 12, 16, 20, 26, 34, 44 |
| `hydro` | mean Eisenberg hydrophobicity | -0.45, -0.25, -0.10, 0.05, 0.25, 0.50 |

So charge bucket 0 is charge ≤ 0, bucket 1 is (0, 2], bucket 2 is (2, 4], and so on;
bucket 6 is above 11.

- **Control dropout:** each control is replaced by NULL with probability 0.15 during training,
  which makes classifier-free guidance available:
  `logits = logits_null + w · (logits_cond - logits_null)`.
- **Guidance weight:** the calibrated value is w = 1.0, where the guided logits equal the
  conditional logits, so sampling needs one forward pass per step. An earlier w = 8 was tuned
  against a conditioning signal that was noise.
- **Two learning rates:** the control embeddings are a separate parameter (`SplitEmbedding`)
  trained at 1e-3 while the pretrained weights use 1e-5. The controls start random, so a shared low
  rate leaves them at initialisation. Their displacement from initialisation was 0.52 to 1.20
  against a random-walk bound of 0.107, and the rank correlation between requested and realised
  charge bucket reached +1.00.
- **Sampling:** confident unmasking in 12 steps (all residues start masked, the most confident are
  fixed first), temperature 1.0, top-p 0.95, restricted to the 20 canonical residues,
  fp16 autocast on Metal or CUDA.
- **Known quirk, kept as run:** training tables bucket a value equal to an edge into the lower
  bucket (`torch.bucketize`). The sampler that drew the pool requested the length bucket with
  `np.searchsorted(..., side="right")`, so a peptide of exactly 12, 16, 20, 26, 34 or 44 residues
  was requested one length bucket above its training bucket. See `controls.sampling_length_bucket`.

## 2. Curriculum

Three stages on a Tesla T4 (fp16 mixed precision, batch 64, AdamW, weight decay 0.01,
gradient-norm clip 1.0), each resumed from the previous one:

| Stage | Table | Rows | Steps | Content |
| --- | --- | --- | --- | --- |
| A, prior | `stage_a_prior.parquet` | 5,634,791 | 4,000 | UniProt slices, protein tiling, ToxProt, AMPSphere |
| B, AMP-like | `stage_b_amp.parquet` | 889,293 | 6,000 | AMPSphere plus curated peptide databases |
| C, potent | `stage_c_potent.parquet` | 4,460 | 2,000 | every measured peptide with broad-spectrum MIC ≤ 16 µM |

Stages A and B are published as the Hugging Face dataset
[`eamag/amp-plm`](https://huggingface.co/datasets/eamag/amp-plm); sources and licences are in
[`data/plm/README.md`](data/plm/README.md). Stage C is rebuilt from the committed oracle table
by `esmc_flow_amp.curriculum`, and matches the table used for the run row for row and bucket for
bucket. No novelty screen was applied to stage C.

## 3. Candidate pool

The pool is the union of two sets of draws from the stage C model: an earlier scale-up pool,
and ten parts of 30,000 draws made on an Apple M5 Pro (Metal, fp16, one process per charge bucket).
After removing duplicates, sequences outside 8-50 residues or the canonical alphabet, exact matches
of `data/antibacterial.fasta`, and sequences without an AMPredictor score, 396,334 unique sequences
remain (66% predicted active, 10.6% Filtering-1).

Each part draws a charge mixture rather than one bucket, because the Filtering-1 rule rejects any
5-residue window with three or more K/R/H and so caps a 20-mer near charge +7 whatever the
guidance weight:

```text
b0:900, b1:3600, b2:11100, b3:10800, b4:3000, b5:600      (3%, 12%, 37%, 36%, 10%, 2% of 30,000)
```

Lengths are drawn from the DBAASP length distribution (8-50). The length control follows the
drawn length; the hydrophobicity control is NULL.

## 4. Filters and oracles

- **Filtering-1 (synthesizability):** no cysteine, no homopolymer run of three,
  no three consecutive hydrophobic residues (A, I, L, M, F, W, V), and no 5-residue window with
  three or more K, R or H.
- **Novelty:** the library has no exact overlap with `data/antibacterial.fasta` (39,448 sequences).
  The top-100 has a maximum Levenshtein ratio of 0.783 against it (limit 0.80); the shortlist
  uses 0.79.
- **AMPredictor** (organisers' starter kit, vendored): predicted MIC in µM. It scores the pool and
  gates the top-100, so its numbers are optimistic for anything it selected.
- **TabPFN**, fitted on the committed tables in `data/oracle/` (300 descriptors): a regressor on
  measured broad-spectrum log10 MIC (about 7.5k rows) and a classifier on `is_hemolytic` (rows with
  a usable HC50). The MIC head is a ranker, not a regressor (Spearman 0.36 on held-out measured
  MIC): compare sets by relative potency and do not read its values as µM. On 752 held-out measured
  MICs its top-100 is 85% ≤ 16 µM and 60% ≤ 4 µM, against 74% and 42% for AMPredictor.

## 5. Library selection (`o2k0d1p97`)

**Held-out references.** The 8,841 canonical DBAASP sequences of up to 65 residues are split 50/50
(seed 0) into a fit half and a validation half. Everything fitted to the reference (Conformity KDE,
ESM-2 density ratio, Precision proxy) sees the fit half only; the comparisons in the writeup
score draws from the validation half.

**Per-sequence features.**

| Feature | Definition |
| --- | --- |
| Conformity | Five-fold cross-validated KDE over (net charge, hydrophobic moment) on 2,000 fit-half sequences, as a percentile against held-out log-densities |
| Density ratio (`lw`) | 48-component PCA of ESM-2 650M embeddings; 25-nearest-neighbour radius in the pool over that in the fit half; clipped at the 1st and 99th percentile |
| Precision proxy (`prec20`) | Fraction of 20 fit-half draws (n = 1,000) whose 12-nearest-neighbour balls contain the sequence |
| Diversity proxy | Mean normalised Levenshtein distance to 200 random pool sequences |
| Active | AMPredictor MIC ≤ 16 µM |
| Filtering-1 | pass or fail |

**Score.** With `z(·)` the standard score over the pool,

```text
score = 2·z(Conformity) + 1.8·Active + 0.5·Filtering-1 + 4·z(density ratio) + 0·z(Precision proxy) + 1·z(Diversity proxy)
```

and a hard filter: sequences with `prec20` < 0.97 are penalised by 1000.

**Assembly.** Length quotas match the DBAASP length histogram over 8-50. Within each length,
sequences are taken in score order and any within Levenshtein ratio 90 of one already taken is
skipped, widening the candidate window if a quota is not filled. The result is exactly 50,000
sequences.

## 6. Top-100

From the library: (1) keep Filtering-1 passes with AMPredictor ≤ 16 µM, take the 16,000 best by
AMPredictor, and keep the first 8,000 of those that are novel (ratio < 0.79 to the reference);
(2) score them with TabPFN; (3) drop those with predicted hemolysis p ≥ 0.35; (4) rank by TabPFN
predicted MIC; (5) fill the DBAASP length histogram for 100 slots, skipping any sequence within
ratio 80 of one already chosen, then fill the remainder by rank. The 100 are a strict subset of
the library, with mean hemolysis p 0.286 and none above 0.55.

## 7. Provenance and licences

Licences are recorded, not resolved: several sources are academic-use or carry no licence file,
and that list is the input to a later clean-up pass. It does not change what the submitted
pair contains.

| Resource | Origin | Licence / terms | Used for |
| --- | --- | --- | --- |
| ESM-C 300M | EvolutionaryScale (2024) | Cambrian Open License (covers fine-tuning) | Generator backbone |
| ESM-2 650M (`t33`) | Lin et al. (2023) | MIT | Embedding space, density ratio, Precision proxy |
| AMPredictor | AMP Challenge starter kit | no `LICENSE` file in the upstream copy | Activity surrogate |
| TabPFN | Prior Labs | Prior Labs terms (API client, personal token) | Oracle ranking, hemolysis gate |
| AMPSphere | Santos-Júnior et al. (2022) | CC-BY-4.0 | Stages A, B |
| UniProt slices, ToxProt | UniProt Consortium | CC-BY-4.0 | Stage A |
| MarLys | MarLys database | CC0 1.0 | Stage B |
| OmegAMP positives | Soares et al. | MIT | Stage B |
| DBAASP | Pirtskhalava et al. (2021) | open access, redistribution with acknowledgement | Reference set, stages B, C, oracle MIC |
| GRAMPA | Witten and Witten (2019) | no upstream licence file | Stages B, C, oracle MIC |
| DRAMP general and clinical | Kang et al. (2019) | CC-BY-4.0 | Stage B |
| NeuroPep, ConoServer, SmProt, CPPsite 2.0 | database publications | academic terms, cited | Stage B |
| `antibacterial.fasta` | competition organisers | competition reference | Novelty measurement; exact matches excluded from the pool |

> Data were obtained from the DBAASP (<https://dbaasp.org>) which is an open-access AMP data
> resource supported by I. Beritashvili Center of Experimental Biomedicine (IBCEB), Tbilisi,
> Georgia and NIAID OCICB, Bethesda, MD.
