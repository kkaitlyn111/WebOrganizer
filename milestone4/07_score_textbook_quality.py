"""
A3: textbook-quality fastText classifier
    kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2

Loads fastText binary, scores each doc, saves P(textbook-quality) as .npy per shard.

Output: milestone2/data/scores_textbook_quality/CC_shard_XXXXXXXX.npy
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "milestone2"))

# import patches fasttext for numpy 2
from fasttext_filters import MODELS_DIR  # type: ignore
import fasttext as _ft  # noqa: E402

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path  # type: ignore
from _shared_io import iter_docs  # type: ignore

from huggingface_hub import hf_hub_download

REPO_ID = "kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2"
FILENAME = "model.bin"
NAME = "textbook_quality"
OUT_DIR = DATA_ROOT / f"scores_{NAME}"

# This classifier outputs three labels (High/Mid/Low). We map to a continuous
# "textbook score" = P(High) + 0.5 * P(Mid).  See model card.
HIGH_LABELS = {"__label__High", "__label__high"}
MID_LABELS = {"__label__Mid", "__label__mid", "__label__Medium", "__label__medium"}

_WORKER_MODEL = None


def _ensure_model() -> Path:
    out = MODELS_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                                local_dir=str(out)))


def _init_worker():
    global _WORKER_MODEL
    _WORKER_MODEL = _ft.load_model(str(_ensure_model()))


def _clean(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def _score_text(model, text: str) -> float:
    text = _clean(text)
    if not text:
        return float("nan")
    labels, probs = model.predict(text, k=-1)
    p_high = p_mid = 0.0
    for lab, p in zip(labels, probs):
        if lab in HIGH_LABELS:
            p_high = float(p)
        elif lab in MID_LABELS:
            p_mid = float(p)
    return p_high + 0.5 * p_mid


def _process_shard(args) -> tuple[int, int, float]:
    shard_idx, out_dir, overwrite, max_docs = args
    out_path = out_dir / f"{shard_stem(shard_idx)}.npy"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0
    model = _WORKER_MODEL
    t0 = time.time()
    vals = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx), max_docs=max_docs):
        vals.append(_score_text(model, obj.get("text", "") or ""))
    arr = np.asarray(vals, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return shard_idx, len(arr), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    work = [(s, args.output_dir, args.overwrite, args.max_docs) for s in shards]
    print(f"=== textbook_quality: {len(work)} shards, {args.n_workers} workers ===")
    print(f"Output: {args.output_dir}")
    totals = [0, 0.0]
    if args.n_workers == 1:
        _init_worker()
        for w in tqdm(work, desc=NAME):
            _, n, el = _process_shard(w)
            if n >= 0:
                totals[0] += n; totals[1] += el
                tqdm.write(f"  shard {w[0]}: {n} docs in {el:.1f}s")
    else:
        with Pool(args.n_workers, initializer=_init_worker) as pool:
            for _, n, el in tqdm(pool.imap_unordered(_process_shard, work),
                                  total=len(work), desc=NAME):
                if n >= 0:
                    totals[0] += n; totals[1] += el
    print(f"\n{NAME}: {totals[0]:,} docs, "
          f"{totals[0] / max(totals[1], 1e-9):.0f} docs/sec/worker")


if __name__ == "__main__":
    main()
