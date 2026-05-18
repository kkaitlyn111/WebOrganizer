#!/bin/bash
#SBATCH --job-name=m4_val
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/val_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/val_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export WANDB_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/wandb
export WANDB_DIR=/juice5b/scr5b/kaitwang/research/cache/wandb
export WANDB_API_KEY=c81a92dcbd1ae5d8a46f8920c86c48f0d0b8e93f
export WANDB_SILENT=true
export TORCHINDUCTOR_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/torchinductor

RID="V${SLURM_ARRAY_TASK_ID}"
echo "=== validation $RID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python3 milestone4/m4/scripts/09_run_validation.py \
    --fresh-master  milestone4/m4/data/fresh_master.parquet \
    --fresh-tokens  milestone4/m4/data/tokens/fresh_token_ids.bin \
    --fresh-offsets milestone4/m4/data/tokens/fresh_doc_offsets.npy \
    --ebm-scores         milestone4/m4/data/fresh_ebm_scores.npy \
    --ebm-interp-scores  milestone4/m4/data/fresh_ebm_interp_scores.npy \
    --lasso-scores       milestone4/m4/data/fresh_lasso_scores.npy \
    --run "$RID" \
    --total-tokens 150000000 \
    --wandb-project m4-validation
