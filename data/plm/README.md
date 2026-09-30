# Curriculum tables (`data/plm/`)

Three stages train the ESM-C generator. Columns in every stage: `sequence`, `b_charge`, `b_length`,
`b_hydro` (control buckets, see `METHOD.md` §1). The parquet files are not committed. Stages A and B
are published as the Hugging Face dataset
[`eamag/amp-plm`](https://huggingface.co/datasets/eamag/amp-plm):

```bash
hf download eamag/amp-plm --repo-type dataset --include "*.parquet" --local-dir data/plm
```

| Stage | File | Rows | Content | Where it comes from |
| --- | --- | --- | --- | --- |
| A, prior | `stage_a_prior.parquet` | 5,634,791 | Broad peptide prior: UniProt slices, protein tiling, ToxProt, AMPSphere, plus the curated peptide databases below | `eamag/amp-plm` |
| B, AMP-like | `stage_b_amp.parquet` | 889,293 | The stage A rows from AMPSphere and the curated peptide databases; explicit negatives, generic protein, ToxProt and SmProt removed | `eamag/amp-plm` |
| C, potent | `stage_c_potent.parquet` | 4,460 | Every measured peptide with broad-spectrum MIC ≤ 16 µM, canonical, 8-50 residues; no novelty screen | `python -m esmc_flow_amp.curriculum`, from `data/oracle/peptide_mic_targets_v3.parquet` |

Stage C is rebuilt row for row and bucket for bucket. Stages A and B are the tables used for the
run. They are downloaded, not rebuilt by this repository; what they contain is listed here so the
provenance is checkable.

## Sources and licences (recorded, not resolved)

| Source | Terms | Used in |
| --- | --- | --- |
| AMPSphere | CC-BY-4.0, attribution | A, B |
| UniProt slices, protein tiling, ToxProt | CC-BY-4.0, attribution | A |
| OmegAMP positives (negatives: A only) | MIT (keep notice) | A, B |
| MarLys | CC0 | A, B (39,767 rows) |
| NeuroPep, ConoServer, CPPsite 2.0 | academic research, cite the papers | A, B |
| SmProt | academic research, cite the paper | A |
| DBAASP | redistribution with acknowledgement | A, B, C (286 rows in the training corpus, 5,338 in the oracle table) |
| GRAMPA | open benchmark, no upstream licence file | A, B, C (6,060 rows in the training corpus, 6,305 in the oracle table) |
| DRAMP general (246), DRAMP clinical (4) | CC-BY-4.0 | A, B |

DBAASP acknowledgement: "Data were obtained from the DBAASP (<https://dbaasp.org>) which is an
open-access AMP data resource supported by I. Beritashvili Center of Experimental Biomedicine
(IBCEB), Tbilisi, Georgia and NIAID OCICB, Bethesda, MD."
