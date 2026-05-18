#!/bin/bash
# Equal-budget validation that TRAINS + SAVES CHECKPOINT (no benchmarks).
# Benchmarks are run separately after via eval_checkpoint_bench.py.
#
# Env vars:
#   MODEL_PRESET=15m|60m|120m (default 15m)
#   TOTAL_TOKENS=int unique tokens to select per strategy (default 150000000)
#   EPOCHS=int (default 1; total_train_tokens = TOTAL_TOKENS * EPOCHS)
#   OUT_SUFFIX=str (default _MODELPRESET_BUDGETM)
#
# Usage:
#   MODEL_PRESET=60m TOTAL_TOKENS=150000000 EPOCHS=2 OUT_SUFFIX=_60m_2ep \
#       sbatch --array=0-9%10 milestone4/m4/scripts/slurm_validation_v2.sh

#SBATCH --job-name=m4_val_v2
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/val_v2_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/val_v2_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=c81a92dcbd1ae5d8a46f8920c86c48f0d0b8e93f
export WANDB_SILENT=true
export TORCHINDUCTOR_CACHE_DIR=/juice5b/scr5b/kaitwang/research/cache/torchinductor

: "${MODEL_PRESET:=15m}"
: "${TOTAL_TOKENS:=150000000}"
: "${EPOCHS:=1}"
: "${OUT_SUFFIX:=_${MODEL_PRESET}_${EPOCHS}ep_$((TOTAL_TOKENS/1000000))M}"

RID="V${SLURM_ARRAY_TASK_ID}"

echo "=== val $RID  preset=$MODEL_PRESET  unique=$TOTAL_TOKENS  epochs=$EPOCHS  out_suffix=$OUT_SUFFIX ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python3 milestone4/m4/scripts/09b_run_validation_equal_tokens.py \
    --run "$RID" --budget "$TOTAL_TOKENS" \
    --epochs "$EPOCHS" \
    --model-preset "$MODEL_PRESET" \
    --save-checkpoint \
    --out-suffix "$OUT_SUFFIX" \
    --wandb-project m4-val-v2
