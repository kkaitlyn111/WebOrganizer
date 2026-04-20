"""
Join all per-shard data into a single master.parquet.

Sources merged on (shard_idx, doc_idx):
  - data/annotations.parquet          (topic, format, cluster, dclm, fwedu, tokens)
  - data/features/<stem>.parquet      (url fields + 16 heuristics)
  - data/scores_pc/<model>/<stem>.npy (perplexity-correlation fastText scores)
  - data/scores_qurater/<stem>.npy    (QuRater 4-axis logits, optional)

Output: data/master.parquet

Run after 01_load_annotations.py and 02_extract_jsonl_features.py are complete.
QuRater (.npy) is optional: missing shards get NaN columns so the file can
be rebuilt later by re-running with --overwrite once QuRater finishes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATA_ROOT, FEATURES_DIR, SAMPLED_SHARDS_JSON,
    ANNOTATIONS_PARQUET, MASTER_PARQUET, shard_stem,
)

PC_MODELS = ["arc_easy", "piqa", "sciq", "lambada_es"]
QURATER_AXES = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]

PC_SCORES_DIR = DATA_ROOT / "scores_pc"
QURATER_DIR = DATA_ROOT / "scores_qurater"


def load_shard(
    shard_idx: int,
    ann_by_shard: dict[int, pd.DataFrame],
) -> pd.DataFrame | None:
    stem = shard_stem(shard_idx)

    # --- annotations slice ---
    ann = ann_by_shard.get(shard_idx)
    if ann is None or len(ann) == 0:
        tqdm.write(f"  shard {shard_idx}: no annotations, skipping")
        return None

    # --- heuristic features ---
    feat_path = FEATURES_DIR / f"{stem}.parquet"
    if not feat_path.exists():
        tqdm.write(f"  shard {shard_idx}: features parquet missing, skipping")
        return None
    feat = pd.read_parquet(feat_path)

    # inner join: doc_idx must appear in both; handles rare ultra-short-doc NaN rows
    df = ann.merge(feat, on="doc_idx", how="inner")
    n_dropped = len(ann) - len(df)
    if n_dropped:
        tqdm.write(f"  shard {shard_idx}: dropped {n_dropped} docs on features join")

    # --- perplexity-correlation fastText scores ---
    # scores[i] = p(__label__include) for doc at position i in the shard
    for model_name in PC_MODELS:
        npy_path = PC_SCORES_DIR / model_name / f"{stem}.npy"
        if npy_path.exists():
            scores = np.load(npy_path)
            df[f"pc_{model_name}"] = scores[df["doc_idx"].values]
        else:
            df[f"pc_{model_name}"] = np.nan

    # --- QuRater 4-axis logits (optional) ---
    qr_path = QURATER_DIR / f"{stem}.npy"
    if qr_path.exists():
        qr = np.load(qr_path)  # shape (n_docs, 4)
        for i, axis in enumerate(QURATER_AXES):
            df[f"qr_{axis}"] = qr[df["doc_idx"].values, i]
    else:
        for axis in QURATER_AXES:
            df[f"qr_{axis}"] = np.nan

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Debug: only process first N sampled shards")
    ap.add_argument("--output", type=Path, default=MASTER_PARQUET)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"{args.output} already exists. Pass --overwrite to rebuild.")
        return

    shards: list[int] = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    if not ANNOTATIONS_PARQUET.exists():
        print(f"ERROR: {ANNOTATIONS_PARQUET} not found. Run 01_load_annotations.py first.")
        return

    print(f"Loading {ANNOTATIONS_PARQUET} ...")
    annot = pd.read_parquet(ANNOTATIONS_PARQUET)
    print(f"  {len(annot):,} rows, shards: {annot['shard_idx'].nunique()}")

    # group by shard for O(1) per-shard lookup
    ann_by_shard: dict[int, pd.DataFrame] = {
        s: g.reset_index(drop=True)
        for s, g in annot.groupby("shard_idx")
    }

    # count available score files for summary
    pc_counts = {m: sum((PC_SCORES_DIR / m / f"{shard_stem(s)}.npy").exists() for s in shards)
                 for m in PC_MODELS}
    qr_count = sum((QURATER_DIR / f"{shard_stem(s)}.npy").exists() for s in shards)
    print(f"\nAvailable scores over {len(shards)} shards:")
    for m, cnt in pc_counts.items():
        print(f"  pc_{m}: {cnt}/{len(shards)}")
    print(f"  qurater:  {qr_count}/{len(shards)}")

    dfs = []
    for s in tqdm(shards, desc="joining shards"):
        df = load_shard(s, ann_by_shard)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No shards assembled — check that annotations and features exist.")
        return

    master = pd.concat(dfs, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(args.output, index=False)

    print(f"\nWrote {args.output}")
    print(f"  rows: {len(master):,}")
    print(f"  columns ({len(master.columns)}): {list(master.columns)}")
    print(f"\nMissing-value counts (columns with any NaN):")
    nan_counts = master.isna().sum()
    for col, cnt in nan_counts[nan_counts > 0].items():
        print(f"  {col}: {cnt:,} NaN  ({100*cnt/len(master):.1f}%)")
    print(f"\nFirst 3 rows:\n{master.head(3).to_string()}")


if __name__ == "__main__":
    main()
