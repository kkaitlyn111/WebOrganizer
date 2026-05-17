#!/bin/bash
#SBATCH --job-name=m4_smoke
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/smoke_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/smoke_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/torchinductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== smoke task=$SLURM_ARRAY_TASK_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# Train only 60M tokens (~few min) per smoke task — just for OOM/speed check
.venv/bin/python3 milestone4/m4/scripts/run_diagnostic.py \
    --run-id $((20 + SLURM_ARRAY_TASK_ID)) --no-wandb \
    --configs milestone4/m4/data/diagnostic_run_configs_trimmed.json \
    --total-tokens 60000000 --unique-tokens 150000000
