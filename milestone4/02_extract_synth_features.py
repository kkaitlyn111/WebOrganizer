"""
Single-pass JSONL extraction of milestone4 synth heuristics (B2-B5 + C).
Writes one parquet per shard under milestone2/data/features_synth/.
Mirrors milestone2/02_extract_jsonl_features.py.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "milestone2"))

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path  # type: ignore
from _shared_io import iter_docs  # type: ignore

from heuristics_synth import extract_synth_features, SYNTH_FEATURE_COLS

FEATURES_SYNTH_DIR = DATA_ROOT / "features_synth"


def process_shard(args) -> tuple[int, int, float]:
    shard_idx, max_docs, out_dir, overwrite = args
    out_path = out_dir / f"{shard_stem(shard_idx)}.parquet"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0
    in_path = doc_path(shard_idx)
    t0 = time.time()
    rows = []
    for doc_idx, obj in iter_docs(in_path, max_docs=max_docs):
        text = obj.get("text", "") or ""
        feats = extract_synth_features(text)
        feats["doc_idx"] = doc_idx
        rows.append(feats)
    elapsed = time.time() - t0
    cols = ["doc_idx"] + SYNTH_FEATURE_COLS
    df = pd.DataFrame(rows, columns=cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return shard_idx, len(df), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=FEATURES_SYNTH_DIR)
    ap.add_argument("--shards", type=int, nargs="*")
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
    print(f"Output: {args.output_dir}")

    total_docs = 0
    total_time = 0.0
    if args.n_workers == 1:
        for w in tqdm(work, desc="shards"):
            sidx, n, el = process_shard(w)
            if n >= 0:
                total_docs += n; total_time += el
                tqdm.write(f"  shard {sidx}: {n} docs in {el:.1f}s")
            else:
                tqdm.write(f"  shard {sidx}: SKIPPED")
    else:
        with Pool(args.n_workers) as pool:
            for sidx, n, el in tqdm(
                pool.imap_unordered(process_shard, work),
                total=len(work), desc="shards"
            ):
                if n >= 0:
                    total_docs += n; total_time += el

    print(f"\nTotal docs: {total_docs:,}")
    print(f"Summed worker time: {total_time:.1f}s "
          f"({total_docs / max(total_time, 1e-9):.0f} docs/sec/worker)")


if __name__ == "__main__":
    main()
