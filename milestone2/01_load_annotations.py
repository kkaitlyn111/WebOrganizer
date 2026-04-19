
#load + concat existing .npy annotations for all shards
# outputs data/annotations.parquet w/ one row per doc

# handles partial downloads:
# - topic/format "choice" derived from logits (argmax) if choice file absent.
# - fwedu_score, tokens are left NaN / -1 if not yet downloaded.

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    N_SHARDS, DATA_ROOT, SAMPLED_SHARDS_JSON, ANNOTATIONS_PARQUET,
    annot_path,
)


def load_shard(shard_idx: int) -> pd.DataFrame | None:
    # build per-shard df, return None if essential files missing
    # logits are our "ground truth" length signal; cluster24 and dclm are required.
    topic_logits_p = annot_path(shard_idx, "topic_logits")
    format_logits_p = annot_path(shard_idx, "format_logits")
    cluster_p = annot_path(shard_idx, "cluster24")
    dclm_p = annot_path(shard_idx, "dclm_score")
    if not (topic_logits_p.exists() and format_logits_p.exists()
            and cluster_p.exists() and dclm_p.exists()):
        return None

    topic_logits = np.load(topic_logits_p)
    format_logits = np.load(format_logits_p)
    cluster24 = np.load(cluster_p)
    dclm = np.load(dclm_p)
    n = len(dclm)
    assert len(cluster24) == n, f"shard {shard_idx}: cluster len {len(cluster24)} != dclm {n}"
    assert topic_logits.shape[0] == n, f"shard {shard_idx}: topic_logits {topic_logits.shape} vs n={n}"
    assert format_logits.shape[0] == n, f"shard {shard_idx}: format_logits {format_logits.shape} vs n={n}"

    # topic/format: prefer saved choice file, ohterwise argmax(logits)
    topic_p = annot_path(shard_idx, "topic")
    format_p = annot_path(shard_idx, "format")
    topic = np.load(topic_p).astype(np.int32) if topic_p.exists() else topic_logits.argmax(-1).astype(np.int32)
    format_ = np.load(format_p).astype(np.int32) if format_p.exists() else format_logits.argmax(-1).astype(np.int32)
    assert len(topic) == n and len(format_) == n

    # fwedu_score, fwedu_rounded, tokens: optional (NaN / -1 if missing)
    fwedu_p = annot_path(shard_idx, "fwedu_score")
    fwedu = np.load(fwedu_p).astype(np.float32) if fwedu_p.exists() else np.full(n, np.nan, dtype=np.float32)
    fwedu_r_p = annot_path(shard_idx, "fwedu_rounded")
    fwedu_r = np.load(fwedu_r_p).astype(np.int32) if fwedu_r_p.exists() else np.full(n, -1, dtype=np.int32)
    tokens_p = annot_path(shard_idx, "tokens")
    tokens = np.load(tokens_p).astype(np.int32) if tokens_p.exists() else np.full(n, -1, dtype=np.int32)
    assert len(fwedu) == n and len(fwedu_r) == n and len(tokens) == n

    return pd.DataFrame({
        "shard_idx": np.full(n, shard_idx, dtype=np.int32),
        "doc_idx": np.arange(n, dtype=np.int32),
        "topic": topic,
        "format": format_,
        "cluster24": cluster24.astype(np.int32),
        "dclm_score": dclm.astype(np.float32),
        "fwedu_score": fwedu,
        "fwedu_rounded": fwedu_r,
        "token_count": tokens,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Debug: only load first N sampled shards")
    ap.add_argument("--output", type=Path, default=ANNOTATIONS_PARQUET)
    args = ap.parse_args()

    shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    dfs = []
    skipped = []
    for s in tqdm(shards, desc="shards"):
        df = load_shard(s)
        if df is None:
            skipped.append(s)
            continue
        dfs.append(df)

    if not dfs:
        print("No shards loaded. Required files missing.")
        return

    full = pd.concat(dfs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.output, index=False)

    print(f"\nLoaded {len(dfs)} shards, skipped {len(skipped)} (missing essential files)")
    if skipped[:5]:
        print(f"  first skipped: {skipped[:5]}")
    print(f"Total docs: {len(full):,}")
    print(f"Wrote {args.output}")
    print("\nFirst 5 rows:")
    print(full.head())
    print("\ndescribe():")
    print(full.describe())
    print("\ntopic value counts (top 5):")
    print(full["topic"].value_counts().head())
    print(f"\nfwedu_score NaNs: {full['fwedu_score'].isna().sum():,} / {len(full):,}")
    print(f"fwedu_rounded == -1: {(full['fwedu_rounded'] == -1).sum():,} / {len(full):,}")
    print(f"token_count == -1: {(full['token_count'] == -1).sum():,} / {len(full):,}")


if __name__ == "__main__":
    main()
