#!/bin/bash
#SBATCH --job-name=m4_pretok
#SBATCH --partition=jag-standard,jag-lo
#SBATCH --array=0-99%30
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/pretok_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/pretok_%A_%a.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/datasets

SHARD=$(.venv/bin/python3 -c "
import json
shards = json.load(open('milestone2/data/sampled_shards.json'))['shards']
print(shards[${SLURM_ARRAY_TASK_ID}])
")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}"
.venv/bin/python3 milestone4/m4/scripts/01_pretokenize_shard.py --shard "${SHARD}"
