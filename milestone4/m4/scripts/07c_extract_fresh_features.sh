#!/bin/bash
# Extract all features for fresh shards by re-using milestone2/milestone4 scripts.
# Run sequentially (8 shards, fast).

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

SHARDS=$(.venv/bin/python3 -c "
import json
print(' '.join(str(s) for s in json.load(open('milestone4/m4/data/fresh_shards.json'))['shards']))
")
echo "Fresh shards: $SHARDS"

cd milestone2
# 1. Heuristic features (16 cols)
.venv/bin/python3 02_extract_jsonl_features.py --shards $SHARDS --n-workers 4

cd ../milestone4
# 2. Synth heuristics (18 cols)
../.venv/bin/python3 02_extract_synth_features.py --shards $SHARDS --n-workers 4

# 3. Entities (3 cols) - fast version
for S in $SHARDS; do
  ../.venv/bin/python3 m4/scripts/entities_fast.py --shard $S --batch-size 128
done

# 4. Textbook quality
../.venv/bin/python3 07_score_textbook_quality.py --shards $SHARDS --n-workers 2

# 5. AI-generated (GPU)
../.venv/bin/python3 08_score_ai_detector.py --shards $SHARDS --batch-size 64

# 6. Perplexity-corr fastText (4 axes)
cd ../milestone2
for AX in arc_easy piqa sciq lambada; do
  ../.venv/bin/python3 03_fasttext_scores.py --filter $AX --shards $SHARDS --n-workers 2
done

# 7. QuRater (optional, GPU heavier - skip if time tight)
# ../.venv/bin/python3 04_qurater_scores.py --shards $SHARDS

echo "DONE fresh-shard feature extraction"
