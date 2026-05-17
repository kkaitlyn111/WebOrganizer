#!/bin/bash
#SBATCH --job-name=m4_fresh_qr
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/fresh_qr_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/fresh_qr_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

IDX=${SLURM_ARRAY_TASK_ID}
SHARD=$(.venv/bin/python3 -c "import json; print(json.load(open('milestone4/m4/data/fresh_shards.json'))['shards'][$IDX])")
echo "=== QuRater task=$IDX shard=$SHARD on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

cd milestone2 && ../.venv/bin/python3 04_qurater_scores.py --shards $SHARD --batch-size 16
