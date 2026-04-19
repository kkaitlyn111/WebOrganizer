# Milestone 2 — Implementation Notes

**Purpose:** Hand-off doc so any future Claude (or me) can resume work on the Stats 305C Milestone 2 pipeline without re-deriving context. Read this FIRST.

See `CLAUDE.md` (one dir up, `WebOrganizer/CLAUDE.md`) for the scientific framing and the full project spec, and `/afs/cs.stanford.edu/u/kaitwang/.claude/plans/help-me-review-and-snazzy-aho.md` for the approved execution plan.

## Status at last session (2026-04-18)

| Stage | Script | Status |
|---|---|---|
| 0 | — | HF LFS pull of Corpus-200B mostly complete (tokens, dclm, fwedu_rounded all 100/100; fwedu_score raw still partial 13/100; topic/format "choice" files partial — see "Known data quirks") |
| 1 | `00_subsample_shards.py` | ✓ Ran. Output: `data/sampled_shards.json` (seed=42, 100 shards). |
| 1 | `01_load_annotations.py` | ✓ Debug-ran on 3 shards (59,044 docs). NOT yet run on all 100 (do this once fwedu_score finishes downloading). |
| 2 | `02_extract_jsonl_features.py` + `heuristics.py` + `test_heuristics.py` | ✓ Unit tests pass, full run on 100 shards done. 2,263,456 docs, 368 MB of parquet under `data/features/`. 3 NaN cells total (ngram_rep_{3,4} on ultra-short docs). |
| 3 | `03_fasttext_scores.py` + `fasttext_filters.py` | ✓ arc_easy, piqa, sciq done; **lambada_es still running (34/100 when last checked)**. Outputs under `data/scores_pc/{model}/*.npy` as float32 `p_include`. |
| 3 | `04_qurater_scores.py` + sbatch files | **Scaffolded but NOT smoke-tested.** Login node has no GPU. User plans to grab interactive GPU and run `04_qurater_scores.py --shards 431 --max-docs 50 --batch-size 8 --output-dir data/scores_qurater_smoke` before we submit `qurater_array.sbatch`. |
| 4 | `06_within_topic_percentiles.py` | NOT WRITTEN YET. Derived percentiles for each quality score within topic / within format. |
| 5 | Notebooks | `nb_explore_features.ipynb` exists (just a feature-explorer). The three Milestone-2 notebooks (sanity/bivariate/hierarchy) NOT WRITTEN YET. |

## Directory layout under `WebOrganizer/milestone2/`

```
milestone2/
├── NOTES.md                          ← you are here
├── config.py                         # paths, SEED=42, N_SHARDS=100, annot path builders
├── _shared_io.py                     # iter_docs() — streams .jsonl.zst
├── heuristics.py                     # feature extractor (URL + 16 Gopher-style features)
├── test_heuristics.py                # 6 unit tests, all pass
├── 00_subsample_shards.py            # writes data/sampled_shards.json + availability report
├── 01_load_annotations.py            # loads .npy annotations into data/annotations.parquet
├── 02_extract_jsonl_features.py      # main JSONL single-pass extractor → data/features/*.parquet
├── 03_fasttext_scores.py             # perplexity-correlations filters → data/scores_pc/{model}/*.npy
├── 04_qurater_scores.py              # QuRater-1.3B adapter (direct transformers, no flash-attn)
├── fasttext_filters.py               # model registry + numpy-2 monkey-patch for fasttext
├── smoke_fasttext.py                 # 5-doc smoke on the fastText filters
├── qurater_smoke.sbatch              # 1-shard, 50-doc GPU smoke test
├── qurater_array.sbatch              # full 100-shard array job, 15 concurrent
├── nb_explore_features.ipynb         # quick Stage-2 feature explorer
├── models/                           # downloaded fastText .bin files (~11 GB)
│   ├── arc_easy/arc_easy_target.bin
│   ├── piqa/piqa_target.bin
│   ├── sciq/sciq_target.bin
│   └── lambada_es/lambada_es_target.bin
└── data/
    ├── sampled_shards.json           # {seed, n_shards, total_shards, shards}
    ├── features/CC_shard_NNNNNNNN_processed.parquet   # 100 files, 20 cols
    ├── annotations.parquet           # (not yet built for all 100 shards)
    └── scores_pc/{arc_easy,piqa,sciq,lambada_es}/CC_shard_NNNNNNNN_processed.npy
```

Also cloned under `WebOrganizer/QuRating/` (the princeton-nlp repo). **Not on our import path** — we deliberately do NOT use their `qurater_annotate.py`. See "QuRater" section below.

## Environment

- Python lives in `/juice5b/scr5b/kaitwang/stats305c/WebOrganizer/.venv/bin/python3` (Python 3.12).
- Always invoke as `../.venv/bin/python3 <script.py>` from inside `milestone2/`.
- Installed this session: `numpy pandas zstandard tqdm pyarrow tldextract nltk fasttext huggingface-hub torch transformers safetensors accelerate`.
- `uv pip install <pkg>` in the repo root installs into the `.venv` automatically.
- `datatools` (referenced by `WebOrganizer/annotate_data/`) is NOT public; we re-implemented what we needed with `zstandard` + `multiprocessing`.

## Pipeline conventions

### Shard naming

All Corpus-200B files follow `CC_shard_{i:08d}_processed` with per-subdir suffixes:
- `documents/<stem>.jsonl.zst`
- `tokens/<stem>.npy`
- `scores_dclm-fasttext/<stem>.npy`
- `scores_fineweb-edu/<stem>.npy`
- `scores_fineweb-edu__rounded/<stem>__rounded.npy`
- `domains_topics/<stem>__choice.npy`, `domains_topics__logits/<stem>__logits.npy`
- `domains_formats/<stem>__choice.npy`, `domains_formats__logits/<stem>__logits.npy`
- `domains_clusters-k24/<stem>.npy`

All encoded in `config.ANNOT_SPECS`. Use `annot_path(shard_idx, key)` and `doc_path(shard_idx)` — do NOT hand-build these paths.

### Feature parquet schema (`data/features/*.parquet`)

Columns (from `heuristics.URL_COLS + heuristics.FEATURE_COLS` plus `doc_idx`):
- `doc_idx` (int)
- `url` (raw), `url_netloc` (lowercased, www. stripped), `url_registered_domain` (via tldextract)
- `char_count`, `word_count`, `mean_word_length`, `frac_alpha`, `frac_digit`, `frac_punctuation`, `frac_uppercase`, `frac_lines_terminal_punct`, `frac_lines_bullet`, `type_token_ratio`, `stopword_fraction`, `ngram_rep_2`, `ngram_rep_3`, `ngram_rep_4`, `num_lines`, `mean_line_length`

Parquet has a variant string type (`str`, not `object`) — pandas prints `str` instead of `object`. Not a bug.

### fastText score naming

Positive label for all 4 perplexity-correlations models is `__label__include`. `03_fasttext_scores.py` always saves `p(__label__include)` regardless of which label is top-1. Monotone in "quality" across all 4.

### QuRater score layout

Per `04_qurater_scores.py`: saves `(n_docs, 4)` float32 arrays. **Logit order from the model README** (do not reorder):
- axis 0 = writing_style
- axis 1 = required_expertise
- axis 2 = facts_and_trivia
- axis 3 = educational_value

Chunking: model trained on ≤512 tokens. We split each doc's token_ids into 512-length windows, forward each, and length-weight-average the 4 logits back to one score vector per doc.

## Known data quirks / gotchas (read before editing)

1. **Partial LFS download.** As of this session:
   - `tokens/`, `scores_dclm-fasttext/`, `scores_fineweb-edu__rounded/`, `documents/`, `domains_*__logits/`, `domains_clusters-k24/` — **100/100 of sampled shards**.
   - `scores_fineweb-edu/` (raw continuous) — **13/100**. `fwedu_rounded` has the rounded-to-int version and is complete.
   - `domains_topics/<stem>__choice.npy` — 54/100. `domains_formats/<stem>__choice.npy` — 42/100.
   - **`01_load_annotations.py` derives topic/format choice from logits via argmax when the `__choice` .npy is missing.** Verified: `logits.argmax(-1) == choice` exactly on a shard where both existed.

2. **fasttext × numpy 2 incompatibility.** The installed `fasttext` 0.9.3 calls `np.array(probs, copy=False)` inside `predict()`, which raises `ValueError` under numpy 2.x. We monkey-patch `_FastText.predict` in `fasttext_filters.py` — that patch must be imported before `fasttext.load_model` is called. The patch re-implements the original correctly (including the `+= "\n"` line check).

3. **Lambada English target has no uploaded binary.** `perplexity-correlations/fasttext-lambada-target` repo on HF only has README.md. We substituted `perplexity-correlations/fasttext-lambada-es-target` (Spanish target) per user decision. Diversity still comes from the 4 benchmarks + 4 training methodologies, but note: the lambada_es filter was trained against a Spanish LAMBADA target, so its interpretation as an "English quality" signal is weaker. Flag this in the report.

4. **QuRater loaded via plain `transformers.AutoModelForSequenceClassification`.** The `princeton-nlp/QuRating` repo's `qurater_annotate.py` depends on `flash-attn` and custom `modeling_flash_llama.py`. QuRater-1.3B is actually a standard `LlamaForSequenceClassification` (confirmed from config.json: `architectures: ["LlamaForSequenceClassification"]`, 2048 hidden, 24 layers, 4 labels). Our adapter avoids the whole flash-attn dependency chain.

5. **`fasttext_filters.py` is imported for its side effect.** The numpy-2 patch registers at import time. If you ever `import fasttext` directly without going through `fasttext_filters` first, `predict()` will still be broken. `03_fasttext_scores.py` imports `fasttext_filters` before `fasttext` — keep that ordering.

6. **Empty-doc handling in heuristics.** `char_count==0` or `word_count==0` → all non-count heuristic features NaN. `ngram_rep_{n}` requires `word_count >= n`; otherwise NaN. Full-corpus run produced only 3 NaN cells total (ngram_rep_3 × 1, ngram_rep_4 × 2), all from ultra-short docs.

7. **URL parsing computes BOTH `netloc` and `registered_domain`** (via tldextract) per user decision; we choose during EDA.

8. **Per-worker model memory for fastText.** Each model is ~3.7 GB. The production script uses `Pool(n_workers, initializer=_init_worker)` so each process loads the model exactly once. With 4 workers that's ~15 GB peak — within budget. Don't raise n_workers past 8 without checking RAM.

## How to resume

### If fwedu_score raw finishes downloading
```bash
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer/milestone2
../.venv/bin/python3 00_subsample_shards.py    # prints fresh availability report
../.venv/bin/python3 01_load_annotations.py    # now 100/100 → data/annotations.parquet
```

### If QuRater smoke looks good
Submit the full array:
```bash
sbatch qurater_array.sbatch
# outputs: data/scores_qurater/CC_shard_NNNNNNNN_processed.npy (shape (n, 4) float32)
```
Monitor: `squeue --me` and `ls data/scores_qurater/ | wc -l`.

### Next scripts to write (not done yet)

- `03_join_features.py` — inner-join `annotations.parquet` + all per-shard features + fastText + QuRater into `data/master.parquet`. Join key: `(shard_idx, doc_idx)`.
- `06_within_topic_percentiles.py` — for each of the 10 quality scores, compute within-topic and within-format percentile ranks. Pure pandas `groupby(...).rank(pct=True)`.
- `nb01_sanity.ipynb`, `nb02_bivariate.ipynb`, `nb03_hierarchy.ipynb` — as specified in `CLAUDE.md` Part 2.
- Report PDF (Milestone 2 deliverable).

## Key numbers to sanity-check against

- 100 sampled shard indices starting `[431, 626, 667, 752, 841, ...]`, ending `..., 9093, 9279, 9531, 9545, 9561]`, seed 42.
- Total docs after Stage 2: **2,263,456**.
- Unique `url_registered_domain`: **798,128** (top: blogspot.com 127,798; wordpress.com 47,710; wikia.com 6,841).
- Stage 2 throughput: ~412 docs/sec per worker (8 workers total ~12 min wall).
- Stage 3 fastText throughput: ~3,600 docs/sec per worker; ~3 min/model with 4 workers.

## Open design decisions for next session

- Lambada substitution: keep `lambada_es`, drop it, or swap for another 4th filter (e.g. NVIDIA quality classifier)?
- Report the 3 NaN rows — drop, impute, or keep as NaN going into the Bayesian model.
- Master parquet format — single file vs partitioned.
