"""
Pass thru each doc + extract from JSONL:
- URL
- 16 rly simple heuristic NLP features per doc
Takes in .jsonl.zst shards, writes one parquet per shard, saved under data/features/
"""
from __future__ import annotations
import argparse
import io
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from config import (
    SAMPLED_SHARDS_JSON, FEATURES_DIR, shard_stem, doc_path,
)
from heuristics import extract_features, FEATURE_COLS, URL_COLS


def iter_docs(path: Path, max_docs: int | None = None):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for i, line in enumerate(text_stream):
            if max_docs is not None and i >= max_docs:
                return
            if not line.strip():
                continue
            obj = json.loads(line)
            yield i, obj


def process_shard(args: tuple[int, int | None, Path, bool]) -> tuple[int, int, float]:
    shard_idx, max_docs, out_dir, overwrite = args
    out_path = out_dir / f"{shard_stem(shard_idx)}.parquet"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0

    in_path = doc_path(shard_idx)
    t0 = time.time()
    rows = []
    for doc_idx, obj in iter_docs(in_path, max_docs=max_docs):
        text = obj.get("text", "") or ""
        url = obj.get("url", "") or ""
        feats = extract_features(text, url)
        feats["doc_idx"] = doc_idx
        rows.append(feats)
    elapsed = time.time() - t0
    cols = ["doc_idx"] + URL_COLS + FEATURE_COLS
    df = pd.DataFrame(rows, columns=cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return shard_idx, len(df), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Per-shard doc cap (debug only)")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--shards", type=int, nargs="*",
                    help="Specific shard indices (overrides sampled_shards.json)")
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    work = [(s, args.max_docs, args.output_dir, args.overwrite) for s in shards]
    print(f"Processing {len(work)} shards with {args.n_workers} workers "
          f"(max_docs={args.max_docs}, overwrite={args.overwrite})")

    total_docs = 0
    total_time = 0.0
    if args.n_workers == 1:
        results = (process_shard(w) for w in work)
        for r in tqdm(results, total=len(work), desc="shards"):
            sidx, n, elapsed = r
            if n >= 0:
                total_docs += n; total_time += elapsed
                tqdm.write(f"  shard {sidx}: {n} docs in {elapsed:.1f}s")
            else:
                tqdm.write(f"  shard {sidx}: SKIPPED (exists)")
    else:
        with Pool(args.n_workers) as pool:
            for sidx, n, elapsed in tqdm(
                pool.imap_unordered(process_shard, work),
                total=len(work), desc="shards"
            ):
                if n >= 0:
                    total_docs += n; total_time += elapsed
                else:
                    tqdm.write(f"  shard {sidx}: SKIPPED (exists)")

    print(f"\nTotal docs: {total_docs:,}")
    print(f"Summed worker time: {total_time:.1f}s "
          f"({total_docs / max(total_time, 1e-9):.0f} docs/sec/worker)")


if __name__ == "__main__":
    main()
