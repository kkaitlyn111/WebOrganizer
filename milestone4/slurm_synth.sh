#!/bin/bash
#SBATCH --job-name=m4_synth
#SBATCH --partition=jag-standard,jag-lo
#SBATCH --array=0-99%30
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=nlp
#SBATCH --output=logs/m4_synth_%A_%a.out
#SBATCH --error=logs/m4_synth_%A_%a.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4
mkdir -p logs

SHARD=$(../.venv/bin/python3 -c "
import json
shards = json.load(open('../milestone2/data/sampled_shards.json'))['shards']
print(shards[${SLURM_ARRAY_TASK_ID}])
")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}"
../.venv/bin/python3 02_extract_synth_features.py --shards "${SHARD}" --n-workers 1
