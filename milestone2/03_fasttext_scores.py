
#Run perplexity correlations fastText filters over shards!

#save one .npy per (model, shard) under data/scores_pc/{model}/CC_shard_XXXXXXXX.npy,
#containing the per-doc prob. of the `__label__include` class

#each worker loads the model once (~3.7 GB RAM per worker per model)
#processes one model at a time across shards to bound RAM

from __future__ import annotations
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# fasttext_filters applies the numpy-2 patch on import.
from fasttext_filters import PC_FASTTEXT_MODELS, ensure_model
import fasttext as _ft

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path
from _shared_io import iter_docs

SCORES_DIR = DATA_ROOT / "scores_pc"
POS_LABEL = "__label__include"

# set once per worker
_WORKER_MODEL = None
_WORKER_NAME = None


def _clean(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def _init_worker(model_name: str):
    global _WORKER_MODEL, _WORKER_NAME
    _WORKER_NAME = model_name
    _WORKER_MODEL = _ft.load_model(str(ensure_model(model_name)))


def _process_shard(args) -> tuple[int, int, float]:
    shard_idx, out_dir, overwrite = args
    out_path = out_dir / f"{shard_stem(shard_idx)}.npy"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0

    model = _WORKER_MODEL
    t0 = time.time()
    probs = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx)):
        text = _clean(obj.get("text", "") or "")
        labels, ps = model.predict(text, k=1)
        top_label = labels[0]
        top_p = float(ps[0])
        probs.append(top_p if top_label == POS_LABEL else 1.0 - top_p)
    arr = np.asarray(probs, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return shard_idx, len(arr), time.time() - t0


def run_model(model_name: str, shards: list[int], n_workers: int, overwrite: bool):
    out_dir = SCORES_DIR / model_name
    work = [(s, out_dir, overwrite) for s in shards]
    print(f"\n=== {model_name}: {len(work)} shards, {n_workers} workers ===")
    totals = [0, 0.0]
    if n_workers == 1:
        _init_worker(model_name)
        for w in tqdm(work, desc=model_name):
            _, n, el = _process_shard(w)
            if n >= 0:
                totals[0] += n; totals[1] += el
    else:
        with Pool(n_workers, initializer=_init_worker, initargs=(model_name,)) as pool:
            for _, n, el in tqdm(pool.imap_unordered(_process_shard, work),
                                 total=len(work), desc=model_name):
                if n >= 0:
                    totals[0] += n; totals[1] += el
    print(f"  {model_name}: {totals[0]:,} docs, "
          f"{totals[0] / max(totals[1], 1e-9):.0f} docs/sec/worker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PC_FASTTEXT_MODELS) + ["all"], default="all")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--n-workers", type=int, default=4,
                    help="Each worker loads the model (~3.7 GB RAM).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    models = list(PC_FASTTEXT_MODELS) if args.model == "all" else [args.model]
    for m in models:
        run_model(m, shards, args.n_workers, args.overwrite)


if __name__ == "__main__":
    main()
