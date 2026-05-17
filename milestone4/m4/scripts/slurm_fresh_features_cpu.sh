#!/bin/bash
# Per-shard CPU feature extraction for fresh shards.
# Array task id => index into fresh_shards.json["shards"].
#
# Usage:
#   N=$(.venv/bin/python3 -c 'import json; print(len(json.load(open("milestone4/m4/data/fresh_shards.json"))["shards"]))')
#   sbatch --array=0-$((N-1))%20 milestone4/m4/scripts/slurm_fresh_features_cpu.sh

#SBATCH --job-name=m4_freshfeat
#SBATCH --partition=john,john-lo
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=nlp
#SBATCH --output=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/freshfeat_%A_%a.out
#SBATCH --error=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone4/m4/logs/freshfeat_%A_%a.err

set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/research/cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

IDX=${SLURM_ARRAY_TASK_ID}
SHARD=$(.venv/bin/python3 -c "import json; print(json.load(open('milestone4/m4/data/fresh_shards.json'))['shards'][$IDX])")
echo "=== task=$IDX shard=$SHARD on $(hostname) ==="

PY=/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/.venv/bin/python3

# 1. m2 heuristics (16)
( cd milestone2 && $PY 02_extract_jsonl_features.py --shards $SHARD --n-workers 4 ) || echo "step1 failed"

# 2. m4 synth heuristics (~18)
( cd milestone4 && $PY 02_extract_synth_features.py --shards $SHARD --n-workers 4 ) || echo "step2 failed"

# 3. textbook quality fastText
( cd milestone4 && $PY 07_score_textbook_quality.py --shards $SHARD --n-workers 2 ) || echo "step3 failed"

# 4. perplexity-corr fastText x 4
( cd milestone2 && $PY 03_fasttext_scores.py --model all --shards $SHARD --n-workers 2 ) || echo "step4 failed"

# 5. entities (spaCy, slow - 1-2hr depending on shard size)
$PY milestone4/m4/scripts/entities_fast.py --shard $SHARD --batch-size 128 || echo "step5 failed"

echo "DONE shard $SHARD"
