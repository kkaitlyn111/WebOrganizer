#!/bin/bash
# Phase 4: extract features on the 8 fresh shards, pretokenize, score with EBM,
# run V0-V9 validation. Fresh shards were downloaded by 07b_download_fresh.py.

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

if [ ! -f milestone4/m4/data/fresh_shards.json ]; then
  echo "ERROR: fresh_shards.json missing. Run 07b_download_fresh.py --download first."
  exit 1
fi

SHARDS=$(.venv/bin/python3 -c "
import json
print(' '.join(str(s) for s in json.load(open('milestone4/m4/data/fresh_shards.json'))['shards']))
")
echo "Fresh shards: $SHARDS"

echo "[1/6] Heuristics (m2) ..."
cd milestone2
../.venv/bin/python3 02_extract_jsonl_features.py --shards $SHARDS --n-workers 4

echo "[2/6] Synth heuristics ..."
cd ../milestone4
../.venv/bin/python3 02_extract_synth_features.py --shards $SHARDS --n-workers 4

echo "[3/6] Entities (fast) ..."
for S in $SHARDS; do
  ../.venv/bin/python3 m4/scripts/entities_fast.py --shard $S --batch-size 128
done

echo "[4/6] Textbook + AI-gen + fastText perplexity ..."
../.venv/bin/python3 07_score_textbook_quality.py --shards $SHARDS --n-workers 2
../.venv/bin/python3 08_score_ai_detector.py --shards $SHARDS --batch-size 64

cd ../milestone2
for AX in arc_easy piqa sciq lambada; do
  ../.venv/bin/python3 03_fasttext_scores.py --filter $AX --shards $SHARDS --n-workers 2 || true
done

echo "[5/6] Pretokenize fresh shards ..."
cd ..
for S in $SHARDS; do
  .venv/bin/python3 milestone4/m4/scripts/01_pretokenize_shard.py --shard $S
done

echo "[6/6] Build fresh master.parquet + score with EBM + run V0-V9 ..."
.venv/bin/python3 milestone4/m4/scripts/09_validation_prepare.py
.venv/bin/python3 milestone4/m4/scripts/09_run_validation.py \
  --fresh-master milestone4/m4/data/fresh_master.parquet \
  --fresh-tokens milestone4/m4/data/tokens/fresh_token_ids.bin \
  --fresh-offsets milestone4/m4/data/tokens/fresh_doc_offsets.npy

echo
echo "DONE phase 4. Next: bash milestone4/m4/scripts/RUN_PHASE_6_FIGURES.sh"
