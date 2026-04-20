#!/bin/bash
#SBATCH --job-name=qurater
#SBATCH --partition=jag-standard,jag-lo
#SBATCH --exclude=jagupard19,jagupard20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-99%30
#SBATCH --account=nlp
#SBATCH --output=logs/qurater_%A_%a.out
#SBATCH --error=logs/qurater_%A_%a.err

# jagupard19/20 have older GPUs whose compute capability isn't in the
# installed PyTorch wheel (CUDA error: no kernel image is available)


set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone2

SHARD_IDX=$(uv run python -c "
import json
shards = json.load(open('data/sampled_shards.json'))['shards']
print(shards[${SLURM_ARRAY_TASK_ID}])
")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD_IDX}"
uv run 04_qurater_scores.py --shards "${SHARD_IDX}" --batch-size 32
