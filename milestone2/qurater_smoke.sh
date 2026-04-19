#!/bin/bash
#SBATCH --job-name=qurater_smoke
#SBATCH --partition=sc-loprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/qurater_smoke_%j.out
#SBATCH --error=logs/qurater_smoke_%j.err

# Smoke test: score the FIRST 50 docs of shard 431 only.
# Writes to data/scores_qurater_smoke/ (keeps real outputs separate).
set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone2
mkdir -p logs

uv run 04_qurater_scores.py \
    --shards 431 \
    --max-docs 50 \
    --batch-size 8 \
    --output-dir data/scores_qurater_smoke
