#!/usr/bin/env bash
# End-to-end execution pipeline for esmc-flow-amp.
# Rebuilds datasets, trains curriculum stages, samples candidate pool,
# computes held-out features, selects the 50k library, and ranks the top-100.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo "AMP Challenge 2027: ESM-C Masked Flow Pipeline"
echo "=========================================================="

# 1. Data Preparation
echo "[1/8] Preparing curriculum training datasets..."
bash scripts/prepare_data.sh

# 2. Curriculum Training (Requires GPU/CUDA)
echo "[2/8] Multi-stage curriculum training..."
if [ -f "runs/run1/c_potent/best.pt" ]; then
    echo "  Checkpoint runs/run1/c_potent/best.pt already exists, skipping training."
else
    echo "  Training Stage A (Prior)..."
    python -m esmc_flow_amp.train \
        --data data/plm/stage_a_prior.parquet \
        --steps 4000 --batch 64 --precision 16-mixed \
        --out runs/run1/a_prior

    echo "  Training Stage B (AMP-like)..."
    python -m esmc_flow_amp.train \
        --data data/plm/stage_b_amp.parquet \
        --steps 6000 --batch 64 --precision 16-mixed \
        --resume runs/run1/a_prior/last.pt \
        --out runs/run1/b_amp

    echo "  Training Stage C (Potent)..."
    python -m esmc_flow_amp.train \
        --data data/plm/stage_c_potent.parquet \
        --steps 2000 --batch 64 --precision 16-mixed \
        --resume runs/run1/b_amp/last.pt \
        --out runs/run1/c_potent
fi

# 3. Candidate Pool Sampling (Metal / CUDA)
echo "[3/8] Sampling candidate pool..."
if [ -f "data/pool.fasta" ]; then
    echo "  data/pool.fasta already exists, skipping sampling."
else
    python -m esmc_flow_amp.sample \
        --checkpoint runs/run1/c_potent/best.pt \
        --spec "b0:900,b1:3600,b2:11100,b3:10800,b4:3000,b5:600" \
        --parts 10 \
        --out data/pool.fasta
fi

# 4. Activity Scoring
echo "[4/8] Scoring pool activity with AMPredictor..."
if [ -f "data/pool_activity.npz" ]; then
    echo "  data/pool_activity.npz already exists, skipping scoring."
else
    python -m esmc_flow_amp.score_pool \
        --pool data/pool.fasta \
        --out data/pool_activity.npz
fi

# 5. ESM-2 Embeddings
echo "[5/8] Extracting ESM-2 650M representations..."
if [ -f "data/pool_esm2.npy" ]; then
    echo "  data/pool_esm2.npy already exists, skipping embedding."
else
    python -m esmc_flow_amp.embed_pool \
        --pool data/pool.fasta \
        --out data/pool_esm2.npy
fi

# 6. Held-Out Selection Features
echo "[6/8] Computing held-out 50/50 selection features..."
if [ -f "data/features/pool_features.npz" ]; then
    echo "  data/features/pool_features.npz already exists, skipping feature computation."
else
    python -m esmc_flow_amp.library_features \
        --pool data/pool.fasta \
        --activity data/pool_activity.npz \
        --embeddings data/pool_esm2.npy \
        --out-dir data/features
fi

# 7. Library Selection (o2k0d1p97)
echo "[7/8] Selecting 50,000-sequence library..."
python -m esmc_flow_amp.select_library \
    --features-dir data/features \
    --out generate/library.fasta

# 8. Top-100 Candidate Ranking
echo "[8/8] Ranking top-100 candidates with TabPFN..."
python -m esmc_flow_amp.rank_top100 \
    --library generate/library.fasta \
    --hemo 0.35 \
    --shortlist 8000 \
    --out generate/top.fasta

echo "Pipeline execution complete. Running validation..."
python -m esmc_flow_amp.generate
echo "All steps finished successfully."
