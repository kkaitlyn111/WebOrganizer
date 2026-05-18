#!/bin/bash
#SBATCH --job-name=m4_bench
#SBATCH --partition=jag-standard,jag-lo,sphinx-lo,sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/bench_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/bench_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false

: "${OUT_SUFFIX:=_60m_2ep}"
RID="V${SLURM_ARRAY_TASK_ID}"

echo "=== bench $RID  suffix=$OUT_SUFFIX  on $(hostname) ==="
.venv/bin/python3 milestone4/m4/scripts/eval_checkpoint_bench.py \
    --ckpt milestone4/m4/data/validation_results_eq${OUT_SUFFIX}/${RID}_ckpt.pt \
    --rid "$RID" \
    --out-dir milestone4/m4/data/validation_results_eq${OUT_SUFFIX}
