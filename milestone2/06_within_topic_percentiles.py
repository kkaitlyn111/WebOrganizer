"""
Add within-group percentile rank columns to master.parquet.

For each quality score, computes each document's percentile rank (0-1)
within its assigned topic and within its assigned format group:

  {score}_pct_topic   = rank within topic    (OLMo 3 insight: quality is topic-relative)
  {score}_pct_format  = rank within format

NaN scores are excluded from ranking and kept as NaN in the output.

Reads data/master.parquet, writes data/master.parquet in place
(or --output for a different path).

Prerequisite: 03_join_features.py must have run first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MASTER_PARQUET

# all quality scores present in master.parquet
# (some will be all-NaN if QuRater / lambada_es not yet complete — handled gracefully)
SCORE_COLS = [
    "dclm_score",
    "fwedu_score",
    "pc_arc_easy",
    "pc_piqa",
    "pc_sciq",
    "pc_lambada_es",
    "qr_writing_style",
    "qr_required_expertise",
    "qr_facts_and_trivia",
    "qr_educational_value",
]


def add_percentiles(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """
    For each score in score_cols, add two new columns:
      {score}_pct_topic  and  {score}_pct_format

    Uses pandas groupby().rank(pct=True, na_option='keep') which:
      - ranks non-NaN values within each group (0 = lowest, 1 = highest)
      - leaves NaN values as NaN in the output
    """
    for col in tqdm(score_cols, desc="computing percentiles"):
        if col not in df.columns:
            tqdm.write(f"  {col}: not in master, skipping")
            continue
        n_valid = df[col].notna().sum()
        if n_valid == 0:
            tqdm.write(f"  {col}: all NaN, skipping")
            df[f"{col}_pct_topic"] = np.nan
            df[f"{col}_pct_format"] = np.nan
            continue

        df[f"{col}_pct_topic"] = (
            df.groupby("topic")[col]
            .rank(pct=True, na_option="keep")
        )
        df[f"{col}_pct_format"] = (
            df.groupby("format")[col]
            .rank(pct=True, na_option="keep")
        )
        tqdm.write(
            f"  {col}: {n_valid:,} valid docs ranked "
            f"({df[col].isna().sum():,} NaN kept)"
        )

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=MASTER_PARQUET)
    ap.add_argument("--output", type=Path, default=None,
                    help="Default: overwrite input file in place")
    ap.add_argument("--scores", nargs="*", default=None,
                    help="Subset of score columns to process (default: all)")
    args = ap.parse_args()

    out_path = args.output or args.input

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run 03_join_features.py first.")
        return

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    score_cols = args.scores if args.scores else SCORE_COLS

    # only process scores that actually exist as columns
    present = [c for c in score_cols if c in df.columns]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        print(f"  Columns not found (will skip): {missing}")
    print(f"  Processing {len(present)} score columns: {present}\n")

    # sanity check grouping variables
    for grp in ("topic", "format"):
        n_groups = df[grp].nunique()
        print(f"  {grp}: {n_groups} unique values  "
              f"(min group size: {df.groupby(grp).size().min()}, "
              f"max: {df.groupby(grp).size().max()})")
    print()

    df = add_percentiles(df, present)

    # summary: show mean within-topic percentile by topic for dclm_score
    # (sanity check: should be ~0.5 for all topics if ranking is correct)
    if "dclm_score_pct_topic" in df.columns:
        print("\nSanity check — mean(dclm_score_pct_topic) by topic (should all be ~0.50):")
        means = df.groupby("topic")["dclm_score_pct_topic"].mean()
        print(means.to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    new_cols = [c for c in df.columns if c.endswith("_pct_topic") or c.endswith("_pct_format")]
    print(f"\nWrote {out_path}")
    print(f"  Total columns: {len(df.columns)}  ({len(new_cols)} new percentile columns)")
    print(f"  New columns: {new_cols}")


if __name__ == "__main__":
    main()
