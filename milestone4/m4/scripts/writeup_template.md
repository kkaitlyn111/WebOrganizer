# Milestone 4 — Criticism and Revision

## Summary

We built an interpretable model that predicts pretraining data quality from human-interpretable
document features, then validated it as a data filter against DCLM and FineWeb-Edu.

## 1. Setup

- **Corpus**: WebOrganizer/Corpus-200B 100-shard subsample (~2.26M docs, 2.0B tokens).
- **Train/val split (LM eval)**: 10 shards held out as fixed LM val set.
- **Features (~50 per doc)**:
  - 16 m2 heuristics (length, repetition, line stats, ...)
  - 18 synthetic-data-inspired heuristics (discourse markers, density, readability, ...)
  - 3 entity features (spaCy)
  - 4 QuRater axes, 4 perplexity-corr fastText scores, textbook quality, AI-gen P(Fake)
  - DCLM-fasttext, FineWeb-Edu rounded (used only as baselines)
  - topic (24-way), format (24-way) categorical
- **Pretokenization**: pre-tokenized whole corpus into uint32 memmap (~7.3GB train pool, ~0.7GB val)
  + per-doc offset arrays, enabling diagnostic runs to load tokens in seconds.

## 2. Phase 1 — Diagnostic runs

We ran **N** 15M-parameter Llama-style LMs (192/576/12L/6H, ~15.4M params, GPT-NeoX tokenizer)
each on 500M tokens drawn from a different feature-based selection of the train pool. Selection
types span single-feature thresholds, multi-feature intersections, random linear combinations,
targeted choices for the paper bar chart, and topic-conditional selections.

We evaluate val loss on a fixed 2000-sequence subsample of the held-out val shards (4M tokens),
giving a consistent comparison across runs.

## 3. Phase 2 — Document-level target

Per-doc target y = mean_quality(runs that included) - mean_quality(runs that excluded), with
quality = min-max-normalized -val_loss. Sanity checks reported in `target_sanity.json`.

## 4. Phase 3 — Interpretable models

| Model | R^2 (test) | Notes |
| --- | --- | --- |
| Lasso (one-hot topic/format) | ? | linear baseline |
| EBM (all features)            | ? | with 20 detected pairwise interactions |
| EBM (no perplexity-corr)      | ? | tests purely interpretable feature set |

Top 20 features by importance: see `phase3_summary.json`.

## 5. Phase 4 — Validation runs

We trained 10 fresh 15M LMs on selections from a small held-out fresh-shard pool. Comparing val
loss (lower is better):

| Run | Selection | val_loss |
| --- | --- | --- |
| V0/V7/V8 | Random 15% | ? |
| V1 | Top 15% DCLM | ? |
| V2 | FWedu ≥ 3 | ? |
| V3 | Top 15% EBM | ? |
| V4 | Top 15% EBM (no PC) | ? |
| V5 | Top 15% Lasso | ? |
| V6 | Within-topic top 15% EBM | ? |
| V9 | Top 15% EBM (seed 43) | ? |

## 6. Findings

(TBD — populate after diagnostic+phase 3 done)

## 7. Limitations

- Per-run LM training (500M tokens, no eval-during-training) means val loss has variance.
  V7/V8 (random replicates) bound the noise floor.
- The "fresh" pool is small (~8 shards) so V0-V9 selection statistics have variance.
- EBM trained on targets derived from the same train pool — some leakage even with the held-out
  fresh pool.
