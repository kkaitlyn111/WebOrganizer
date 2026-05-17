#!/bin/bash
#SBATCH --job-name=m4_bs32
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --exclude=jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/bs32_%j.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/bs32_%j.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/torchinductor

echo "=== bs32 smoke on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# Train just 800 steps (enough for steady-state measurement) with micro=32 to check if OOM
.venv/bin/python3 milestone4/m4/scripts/run_diagnostic.py \
    --run-id 16 --no-wandb \
    --configs milestone4/m4/data/diagnostic_run_configs.json \
    --total-tokens 100000000 --unique-tokens 150000000 \
    --micro-batch-size 32
