#!/bin/bash
#SBATCH --job-name=m4_ent_fast
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo
#SBATCH --array=0-99%40
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/ent_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/ent_%A_%a.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

SHARD=$(.venv/bin/python3 -c "
import json
shards = json.load(open('milestone2/data/sampled_shards.json'))['shards']
print(shards[${SLURM_ARRAY_TASK_ID}])
")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}"
.venv/bin/python3 milestone4/m4/scripts/entities_fast.py --shard "${SHARD}" --batch-size 128
