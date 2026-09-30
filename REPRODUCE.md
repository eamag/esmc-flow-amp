# Reproduction guide

## 1. Check the submitted pair (seconds, CPU, no network)

```bash
uv sync
uv run generate
```

`generate` validates `data/final/{library,top}.fasta` against the competition rules (50,000 unique
canonical sequences of 8-50 residues, a 100-sequence top list that is a subset, no exact overlap
with `data/antibacterial.fasta`, Levenshtein ratio ≤ 0.80 for the top-100), copies both to
`generate/` byte for byte, and checks the MD5s:

- `generate/library.fasta`: `5cc81ec3b95b1765069b5fd86cbfac40`
- `generate/top.fasta`: `a00448984978e3d08c0fe4263e69f2eb`

The organisers' check is
`uv run python -P scripts/verify_submission.py <repo-url> --antibacterial-fasta data/antibacterial.fasta`.

## 2. What is and is not reproducible

| Step | Status |
| --- | --- |
| Submitted files, `generate` | exact, byte-identical on every run |
| Stage C table | exact: `esmc_flow_amp.curriculum` rebuilds all 4,460 rows and their buckets |
| Stage A and B tables | exact: downloaded from `eamag/amp-plm` (5.6M and 889k rows); sources and licences in `data/plm/README.md` |
| Training | same recipe and hyper-parameters; not bit-identical (GPU nondeterminism, spot-instance restarts) |
| Sampling | same model, mixture and sampler; draws depend on device and seed |
| Pool | the shipped pool of 396,334 is an earlier scale-up pool plus ten 30,000-draw parts; `sample` makes the ten-part mixture only |
| Library selection | deterministic given a pool, its AMPredictor scores and ESM-2 embeddings; the shipped library came from the shipped pool |
| Top-100 | needs a Prior Labs TabPFN token; a rerun on the shipped library shared 73 of the 100 sequences (section 5) |

So the pipeline below reproduces the method and the recipe, and the committed files are
the reproducible artefact.

## 3. Hardware

| Step | Hardware used |
| --- | --- |
| Training, three stages | Tesla T4 (16 GB), fp16 mixed precision |
| Sampling | Apple M5 Pro (Metal, fp16), or CUDA |
| AMPredictor scoring, ESM-2 embedding | GPU or Metal; CPU works, slowly |
| Selection | multi-core CPU, about 8 GB RAM |
| Top-100 | CPU or Metal, plus the TabPFN API |

## 4. Pipeline

```bash
git clone https://github.com/eamag/esmc-flow-amp.git && cd esmc-flow-amp
uv sync --extra pipeline
cp .env.example .env          # set TABPFN_TOKEN for step 7 only
```

`scripts/run_pipeline.sh` runs everything below in order and finishes with `generate`;
the commands are spelled out here.

**1. Curriculum tables.** Download the stage A and B tables from the Hugging Face dataset
[`eamag/amp-plm`](https://huggingface.co/datasets/eamag/amp-plm), then rebuild stage C:

```bash
hf download eamag/amp-plm --repo-type dataset --include "*.parquet" --local-dir data/plm
bash scripts/prepare_data.sh
```

**2. Training.** Each stage resumes from the previous one; `--steps` counts additional steps.

```bash
python -m esmc_flow_amp.train --data data/plm/stage_a_prior.parquet  --steps 4000 --batch 64 --out runs/run1/a_prior
python -m esmc_flow_amp.train --data data/plm/stage_b_amp.parquet    --steps 6000 --batch 64 --resume runs/run1/a_prior/last.pt --out runs/run1/b_amp
python -m esmc_flow_amp.train --data data/plm/stage_c_potent.parquet --steps 2000 --batch 64 --resume runs/run1/b_amp/last.pt  --out runs/run1/c_potent
```

Everything else is the default and matches the run: lr 1e-5, control lr 1e-3, control dropout 0.15,
uniform mask rate, `16-mixed`. The generator is `runs/run1/c_potent/best.pt`.

**3. Sampling.** Ten parts of the charge mixture at guidance weight 1.0. The output is
filtered to unique canonical 8-50-residue sequences that are not exact matches of the reference.

```bash
python -m esmc_flow_amp.sample --checkpoint runs/run1/c_potent/best.pt \
  --spec "b0:900,b1:3600,b2:11100,b3:10800,b4:3000,b5:600" --parts 10 --out data/pool.fasta
```

**4. Scoring and embedding.**

```bash
python -m esmc_flow_amp.score_pool --pool data/pool.fasta --out data/pool_activity.npz
python -m esmc_flow_amp.embed_pool --pool data/pool.fasta --out data/pool_esm2.npy
```

**5. Held-out features.** Splits DBAASP 50/50 and computes Conformity, density ratio,
the Precision proxy and the Diversity proxy for every pool sequence.

```bash
python -m esmc_flow_amp.library_features --pool data/pool.fasta --activity data/pool_activity.npz \
  --embeddings data/pool_esm2.npy --out-dir data/features
```

**6. Library.** Configuration `o2k0d1p97`.

```bash
python -m esmc_flow_amp.select_library --features-dir data/features --out generate/library.fasta
```

**7. Top-100.**

```bash
python -m esmc_flow_amp.rank_top100 --library generate/library.fasta --hemo 0.35 --shortlist 8000 --out generate/top.fasta
```

**8. Evaluate** (optional). `esmc_flow_amp.phase1_harness` is the Phase 1 harness that produced `metrics/PHASE1_DEEP3.md`:

```bash
python -m esmc_flow_amp.phase1_harness --candidates Ours=generate/library.fasta --ref-bg <uniprot_8_50.fasta> --device mps
```

The generic-peptide UniProt reference (167 MB) is not committed; without `--ref-bg`
those metrics are skipped.

## 5. What was run for this repository

- `generate`, twice, byte-identical, and the organisers' `verify_submission.py` on a clean clone.
- Stage C rebuild: identical to the table used for training (4,460 rows, all three buckets).
- TabPFN descriptors: identical to the research code on 300 library sequences and on 100 cached
  training rows.
- Training: two steps and a resume on Metal (fp32) from the stage C table; sampling from the real
  stage C checkpoint; then scoring, embedding, features and selection on a 2,000-sequence pool.
- Top-100 rerun on the shipped library (Apple M5 Pro, Metal, TabPFN via the API): AMPredictor
  re-scored the 50,000 sequences (about 25 minutes), 7,050 passed the filters, TabPFN scored them
  (about 33 minutes), 2,626 passed the hemolysis gate, and the top 100 were selected. It shares
  73 of 100 sequences with the shipped `generate/top.fasta`, so the MD5 differs. In the Phase 1
  harness both sets are 100% active; AMPredictor median MIC is 6.29 µM for the rerun against
  5.34 µM shipped, and TabPFN median MIC is 6.25 against 6.42 µM. Which input drifted was not
  tested (hosted TabPFN, or AMPredictor scores that were recomputed because `data/pool_activity.npz`
  is not committed).
- Not run here: the full training, and the 396k-sequence pool and its selection. The selection and
  top-100 code was compared line by line with the research scripts that produced the shipped files.

## 6. Not committed

`data/pool.fasta` and its scores and embeddings (about 2 GB), the stage A and B tables,
the checkpoints (3.7 GB each with optimizer state), and `data/oracle/tabpfn_cache.parquet`.
The oracle tables in `data/oracle/` and the AMPredictor weights are committed.
