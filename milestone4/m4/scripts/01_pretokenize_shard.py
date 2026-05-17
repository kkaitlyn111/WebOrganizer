"""Pretokenize a single shard with GPT-NeoX tokenizer.

Writes per-shard:
  pershard/{stem}_token_ids.npy  (uint32, total tokens)
  pershard/{stem}_doc_offsets.npy (int64, len = n_docs + 1)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import PERSHARD_TOKENS_DIR, TOKENIZER_NAME, REPO_ROOT  # noqa: E402

import importlib.util as _ilu
def _load(name, path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_m2cfg = _load("m2cfg", REPO_ROOT / "milestone2" / "config.py")
_m2io = _load("m2io", REPO_ROOT / "milestone2" / "_shared_io.py")
doc_path = _m2cfg.doc_path
shard_stem = _m2cfg.shard_stem
iter_docs = _m2io.iter_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stem = shard_stem(args.shard)
    out_tok = PERSHARD_TOKENS_DIR / f"{stem}_token_ids.npy"
    out_off = PERSHARD_TOKENS_DIR / f"{stem}_doc_offsets.npy"
    if out_tok.exists() and out_off.exists() and not args.overwrite:
        n = np.load(out_off, mmap_mode="r")
        print(f"shard {args.shard}: already done, {len(n)-1} docs, {n[-1]} toks")
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    t0 = time.time()
    # Single pass: tokenize each doc, accumulate.
    chunks = []
    offsets = [0]
    texts_batch = []
    BATCH = 256

    def flush():
        if not texts_batch:
            return
        enc = tok(texts_batch, add_special_tokens=False, return_attention_mask=False)
        for ids in enc["input_ids"]:
            a = np.asarray(ids, dtype=np.uint32)
            chunks.append(a)
            offsets.append(offsets[-1] + len(a))
        texts_batch.clear()

    n_docs = 0
    for _doc_idx, obj in iter_docs(doc_path(args.shard), max_docs=args.max_docs):
        texts_batch.append(obj.get("text", "") or "")
        n_docs += 1
        if len(texts_batch) >= BATCH:
            flush()
    flush()

    if chunks:
        token_ids = np.concatenate(chunks)
    else:
        token_ids = np.zeros(0, dtype=np.uint32)
    offsets_arr = np.asarray(offsets, dtype=np.int64)

    np.save(out_tok, token_ids)
    np.save(out_off, offsets_arr)
    dt = time.time() - t0
    print(f"shard {args.shard}: {n_docs} docs, {int(offsets_arr[-1])} toks in "
          f"{dt:.1f}s ({n_docs/max(dt,1e-9):.0f} docs/sec)")


if __name__ == "__main__":
    main()
