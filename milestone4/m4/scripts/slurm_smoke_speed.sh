#!/bin/bash
#SBATCH --job-name=m4_smoke
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --exclude=jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/smoke_%j.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/smoke_%j.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

echo "=== smoke on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# Use a small selection run_id (0016 was 1.4M docs / 1.4B tokens; perfect).
# Train on 600M to compare apples-to-apples with previous 42min baseline.
.venv/bin/python3 milestone4/m4/scripts/run_diagnostic.py --run-id 16 --no-wandb \
    --total-tokens 300000000 --unique-tokens 150000000
