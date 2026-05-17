#!/bin/bash
#SBATCH --job-name=m4_diag
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --exclude=jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/diag_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/diag_%A_%a.err

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/wandb
export WANDB_DIR=/juice5b/scr5b/kaitwang/research/cache/wandb
export WANDB_API_KEY=c81a92dcbd1ae5d8a46f8920c86c48f0d0b8e93f
export WANDB_SILENT=true
# Reuse torch.compile cache across array tasks on the same node
export TORCHINDUCTOR_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/torchinductor

RUN_ID=${SLURM_ARRAY_TASK_ID}
echo "=== run_id=$RUN_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python3 milestone4/m4/scripts/run_diagnostic.py --run-id "$RUN_ID"
