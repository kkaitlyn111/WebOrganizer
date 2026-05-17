#!/bin/bash
#SBATCH --job-name=m4_fresh_ai
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/fresh_ai_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/fresh_ai_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

IDX=${SLURM_ARRAY_TASK_ID}
SHARD=$(.venv/bin/python3 -c "import json; print(json.load(open('milestone4/m4/data/fresh_shards.json'))['shards'][$IDX])")
echo "=== AI detector task=$IDX shard=$SHARD on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

cd milestone4 && ../.venv/bin/python3 08_score_ai_detector.py --shards $SHARD --batch-size 64
