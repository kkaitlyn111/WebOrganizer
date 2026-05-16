# Milestone 4 — Additional Features

Scripts here extend the milestone 2 pipeline with:
- **A3** textbook-quality fastText (CPU)
- **A4** AI-generated text detector (GPU, RoBERTa-base)
- **B1** spaCy entity features (3 columns)
- **B2–B5, C** synthetic-data-inspired heuristics + readability (18 columns, regex/textstat, CPU)

All outputs are written under `milestone2/data/` so the existing join script picks them up.

## Layout

| Script | Output | Compute |
| --- | --- | --- |
| `heuristics_synth.py` | (module) | — |
| `02_extract_synth_features.py` | `data/features_synth/<stem>.parquet` | CPU pool |
| `06_extract_entities.py` | `data/entities/<stem>.npz` | CPU, spaCy (slurm array) |
| `07_score_textbook_quality.py` | `data/scores_textbook_quality/<stem>.npy` | CPU pool, fastText |
| `08_score_ai_detector.py` | `data/scores_ai_generated/<stem>.npy` | single GPU |

All scripts accept `--shards N [N ...]`, `--max-docs N`, `--overwrite`.

## Smoke tests

```bash
../.venv/bin/python3 test_heuristics_synth.py
../.venv/bin/python3 02_extract_synth_features.py --shards 431 --max-docs 50 --n-workers 1 --overwrite
../.venv/bin/python3 06_extract_entities.py        --shards 431 --max-docs 30 --overwrite
../.venv/bin/python3 07_score_textbook_quality.py  --shards 431 --max-docs 30 --n-workers 1 --overwrite
../.venv/bin/python3 08_score_ai_detector.py       --shards 431 --max-docs 20 --batch-size 16 --overwrite
```

## Full runs

```bash
# heuristics: ~20 docs/sec/worker; 8 workers; ~100 shards
../.venv/bin/python3 02_extract_synth_features.py --n-workers 8

# textbook quality fastText: ~1k docs/sec/worker; 4 workers
../.venv/bin/python3 07_score_textbook_quality.py --n-workers 4

# entities: slurm array (one shard per task; spaCy doesn't multiprocess well)
sbatch slurm_entities.sh

# ai detector: single GPU job
../.venv/bin/python3 08_score_ai_detector.py --batch-size 64
```

## Notes

- `heuristics_synth.py` skips `content_word_ratio` because it equals `1 - stopword_fraction` already in `milestone2/heuristics.py`.
- The textbook classifier has labels High/Mid/Low. We collapse to a continuous score `P(High) + 0.5 * P(Mid)`.
- The AI detector's label map is `{0: Fake, 1: Real}`; the script auto-detects the FAKE index from `id2label`.
- All shard outputs are length-aligned with the existing `.npy` annotations in `Corpus-200B/`.
