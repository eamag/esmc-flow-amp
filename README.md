# esmc-flow-amp

AMP Challenge 2027 submission: a charge-steered ESM-C 300M masked flow generator,
drawing a 396k-candidate pool, with held-out library selection (o2k0d1p97) and
a TabPFN-ranked, hemolysis-gated top-100.

Generates the two files the competition requires:

```text
generate/library.fasta   50,000 unique, legal, reference-free sequences
generate/top.fasta       the 100 best of them, a strict subset
```

The Kaggle submission writeup, its card image, and the upload files are staged
in [`writeup/`](writeup/). Start at
[`writeup/WRITEUP.md`](writeup/WRITEUP.md).

## Who this is for

- **Organisers and reviewers.** Verify the submitted pair in seconds: `generate` validates it
  against the competition rules and checks the MD5s, with no network or GPU.
- **Anyone who wants a controllable AMP generator.** The published ESM-C weights sample peptides
  steered by net charge, length and hydrophobicity. The main use is drawing a candidate pool that
  you then score and filter with your own activity model.
- **People building on the method.** Held-out library selection, the TabPFN-ranked top-100 and the
  Phase 1 harness are separate modules you can reuse on any pool of peptides.
- **Anyone retraining.** The recipe, the stage A and B tables and the stage C rebuild are here.

It does not give you wet-lab validated peptides: activity is predicted (AMPredictor, TabPFN), and
none of the generated sequences was tested in the lab.

## Headline results

Official Phase 1 harness benchmark (n = 1,000, seed 42), comparing our final submission against
our earlier diffusion pair (DiffTabPFN):

| Metric Family | Metric | Ours (ESM-C Flow) | Earlier pair |
| --- | --- | --- | --- |
| **Activity** | AMPredictor Median MIC | **9.07 µM** | 13.80 µM |
| | Active fraction (MIC ≤ 16 µM) | **79.8%** | 54.9% |
| **Embeddings** | ESM-2 FBD / MMD | **1.55 / 2.81** | 2.38 / 8.75 |
| | ESM-C FBD / MMD | **0.05 / 0.08** | 0.11 / 0.58 |
| | Precision / Recall | **0.996 / 0.865** | 0.995 / 0.845 |
| **Biophysics** | Property Conformity | **0.808** | 0.752 |
| | Sequence Diversity | 0.826 | **0.828** |
| **Filters** | Filtering-1 pass rate | **20.2%** | 9.2% |
| | Novelty compliance (≤80% similarity) | **99.4%** | 73.1% |

Top-100 candidate ranking (TabPFN oracle trained on experimental MIC):

| Set | TabPFN active ≤ 16 µM | AMPredictor median MIC | Hemolysis mean p | Hemolysis p > 0.55 |
| --- | --- | --- | --- | --- |
| **Ours (top-100)** | **100%** | **5.34 µM** | **0.286** | **0%** |
| Earlier pair (top-100) | 100% | 6.63 µM | 0.313 | 9% |

Every number comes from recorded harness evaluations in [`metrics/`](metrics/).

## Quick start

**1. Verify the submission.** Validates length bounds (8-50 aa), canonical alphabet,
duplicate-freedom, exact antibacterial overlap and Levenshtein novelty against
`data/antibacterial.fasta`. It runs in seconds on CPU, with no network calls or downloads.

```bash
uv sync
uv run generate
```

**2. Sample peptides with the published weights.** The checkpoint needs `esm==3.2.1.post1`, which
the lock file pins. This draws 200 peptides, 100 each from charge buckets 1 and 5 (about 13 s on an
Apple M5 Pro). The sampler uses CUDA, then Metal, then CPU.

```bash
uv run --extra pipeline --locked hf download eamag/esmc-flow-amp checkpoint.pt --local-dir weights
uv run --extra pipeline --locked python -m esmc_flow_amp.sample \
  --checkpoint weights/checkpoint.pt --spec "b1:100,b5:100" --parts 1 --out pool.fasta
```

`--spec` is a list of `b<charge bucket>:<draws>`. The bucket edges, the Python API for setting
length and hydrophobicity, and the use cases are in the
[model card](https://huggingface.co/eamag/esmc-flow-amp).

**3. Retrain the generator.** Download the stage A and B tables, rebuild stage C and train the three
stages. A T4 was used for the real run; the commands and hyper-parameters are in
[`REPRODUCE.md`](REPRODUCE.md).

```bash
uv run --extra pipeline --locked hf download eamag/amp-plm --repo-type dataset \
  --include "*.parquet" --local-dir data/plm
uv run --extra pipeline --locked bash scripts/prepare_data.sh
```

## Weights and data

The fine-tuned weights are not in this repository (3.7 GB with optimizer state). The stage C
generator is published without optimizer state at
[`eamag/esmc-flow-amp`](https://huggingface.co/eamag/esmc-flow-amp) on Hugging Face, and the stage A
and B training tables at [`eamag/amp-plm`](https://huggingface.co/datasets/eamag/amp-plm). The
submitted pair does not depend on them.

The shipped submission files (`generate/library.fasta` and `generate/top.fasta`) are committed
and validated deterministically by `generate`.

## Further documentation

- **[`METHOD.md`](METHOD.md)**: the masked flow objective, control conditioning,
  the three curriculum stages, the held-out reference split, library selection,
  the top-100 procedure, and the data and licence table.
- **[`REPRODUCE.md`](REPRODUCE.md)**: hardware (T4 for training,
  Apple Silicon for sampling), what is and is not reproducible,
  the command for each pipeline step, and the expected MD5 checksums.
- **[`writeup/WRITEUP.md`](writeup/WRITEUP.md)**: the text for each field of the Kaggle submission form.
