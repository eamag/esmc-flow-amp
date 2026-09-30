#!/usr/bin/env bash
# Prepare the curriculum tables. Stage C is rebuilt from the committed oracle table; stages A and B come from the Hugging Face dataset eamag/amp-plm.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m esmc_flow_amp.curriculum

for stage in stage_a_prior stage_b_amp; do
    if [ ! -f "data/plm/${stage}.parquet" ]; then
        echo "Missing data/plm/${stage}.parquet: run: hf download eamag/amp-plm --repo-type dataset --include \"*.parquet\" --local-dir data/plm (see data/plm/README.md)." >&2
        exit 1
    fi
done
echo "Curriculum tables ready in data/plm/."
