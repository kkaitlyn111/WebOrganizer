#!/bin/bash
#SBATCH --job-name=m4_pretok_fresh
#SBATCH --partition=john,john-lo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/pretok_fresh_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/pretok_fresh_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=true

IDX=${SLURM_ARRAY_TASK_ID}
SHARD=$(.venv/bin/python3 -c "import json; print(json.load(open('milestone4/m4/data/fresh_shards.json'))['shards'][$IDX])")
echo "=== pretokenize task=$IDX shard=$SHARD on $(hostname) ==="

.venv/bin/python3 milestone4/m4/scripts/01_pretokenize_shard.py --shard $SHARD
