#!/bin/bash
#SBATCH --job-name=m4_aidet
#SBATCH --partition=jag-standard,jag-lo
#SBATCH --exclude=jagupard19,jagupard20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-99%15
#SBATCH --time=02:00:00
#SBATCH --account=nlp
#SBATCH --output=logs/m4_aidet_%A_%a.out
#SBATCH --error=logs/m4_aidet_%A_%a.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4
mkdir -p logs

SHARD=$(../.venv/bin/python3 -c "
import json
shards = json.load(open('../milestone2/data/sampled_shards.json'))['shards']
print(shards[${SLURM_ARRAY_TASK_ID}])
")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}"
../.venv/bin/python3 08_score_ai_detector.py --shards "${SHARD}" --batch-size 64
