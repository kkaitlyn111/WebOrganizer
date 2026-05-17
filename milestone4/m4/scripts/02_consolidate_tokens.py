"""After all per-shard tokens exist, concatenate into big memmaps + pack val.

Train pool: 90 shards -> train_pool_token_ids.bin + train_pool_doc_offsets.npy
Val:        10 shards -> val_token_ids.bin       + val_doc_offsets.npy
                          + val_sequences.npy (pre-packed, shape (N, SEQ_LEN+1))

We also save train_pool_doc_index.npy: array of (shard_idx, doc_local_idx) per doc
so that diagnostic runs can map between master-parquet rows and pretokenized docs.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (PERSHARD_TOKENS_DIR, TOKENS_DIR, VAL_SHARDS_JSON,
                    TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, TRAIN_DOC_INDEX,
                    VAL_TOKEN_IDS, VAL_DOC_OFFSETS, VAL_SEQUENCES,
                    TOKENS_META, SEQ_LEN, REPO_ROOT)  # noqa

import importlib.util as _ilu
def _load(name, p):
    spec = _ilu.spec_from_file_location(name, str(p))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
_m2cfg = _load("m2cfg", REPO_ROOT / "milestone2" / "config.py")
shard_stem = _m2cfg.shard_stem


def load_shard(idx):
    s = shard_stem(idx)
    toks = np.load(PERSHARD_TOKENS_DIR / f"{s}_token_ids.npy", mmap_mode="r")
    off = np.load(PERSHARD_TOKENS_DIR / f"{s}_doc_offsets.npy")
    return toks, off


def consolidate(shards, out_tok_bin: Path, out_offsets_npy: Path,
                out_doc_index_npy: Path | None = None):
    # Two-pass: tally sizes, then memmap-write.
    lengths_per_shard = []
    total_tokens = 0
    total_docs = 0
    for s in shards:
        _, off = load_shard(s)
        n_docs = len(off) - 1
        total_docs += n_docs
        total_tokens += int(off[-1])
        lengths_per_shard.append((s, n_docs, int(off[-1]), off))
    print(f"  {len(shards)} shards, {total_docs:,} docs, {total_tokens:,} toks "
          f"({total_tokens * 4 / 1e9:.2f} GB)")

    mm = np.memmap(out_tok_bin, dtype=np.uint32, mode="w+", shape=(total_tokens,))
    offsets = np.zeros(total_docs + 1, dtype=np.int64)
    doc_index = np.zeros((total_docs, 2), dtype=np.int32) if out_doc_index_npy else None

    tok_cursor = 0
    doc_cursor = 0
    for s, n_docs, n_toks, off in lengths_per_shard:
        toks, _ = load_shard(s)
        mm[tok_cursor: tok_cursor + n_toks] = toks
        # Per-doc offsets within shard relative to tok_cursor:
        offsets[doc_cursor: doc_cursor + n_docs + 1] = off + tok_cursor
        # Note: the (n_docs+1)-th entry of the *next* shard will overwrite this end
        if doc_index is not None:
            doc_index[doc_cursor: doc_cursor + n_docs, 0] = s
            doc_index[doc_cursor: doc_cursor + n_docs, 1] = np.arange(n_docs)
        tok_cursor += n_toks
        doc_cursor += n_docs
    offsets[-1] = total_tokens
    mm.flush()
    np.save(out_offsets_npy, offsets)
    if doc_index is not None:
        np.save(out_doc_index_npy, doc_index)
    return total_docs, total_tokens


def pack_val(val_token_ids_bin: Path, val_offsets_npy: Path,
             out_sequences_npy: Path, seq_len: int):
    # Concatenate val token stream into (N, seq_len + 1) packed sequences.
    # Use seq_len+1 because LM training shifts inputs/targets by one.
    offsets = np.load(val_offsets_npy)
    total = int(offsets[-1])
    mm = np.memmap(val_token_ids_bin, dtype=np.uint32, mode="r", shape=(total,))
    n = total // (seq_len + 1)
    arr = np.asarray(mm[: n * (seq_len + 1)], dtype=np.uint32).reshape(n, seq_len + 1)
    np.save(out_sequences_npy, arr)
    print(f"  packed val: {n} sequences of length {seq_len + 1}")


def main():
    split = json.loads(VAL_SHARDS_JSON.read_text())
    train_shards = split["train"]
    val_shards = split["val"]

    # Verify all per-shard files exist
    missing = []
    for s in train_shards + val_shards:
        stem = shard_stem(s)
        if not (PERSHARD_TOKENS_DIR / f"{stem}_token_ids.npy").exists():
            missing.append(s)
    if missing:
        raise SystemExit(f"Missing per-shard tokens for {len(missing)} shards: {missing[:10]}...")

    print(f"=== Train pool ({len(train_shards)} shards) ===")
    n_tr_docs, n_tr_toks = consolidate(
        train_shards, TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, TRAIN_DOC_INDEX)
    print(f"=== Val ({len(val_shards)} shards) ===")
    n_va_docs, n_va_toks = consolidate(val_shards, VAL_TOKEN_IDS, VAL_DOC_OFFSETS)

    print(f"=== Packing val sequences (seq_len={SEQ_LEN}) ===")
    pack_val(VAL_TOKEN_IDS, VAL_DOC_OFFSETS, VAL_SEQUENCES, SEQ_LEN)

    meta = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "n_train_docs": n_tr_docs,
        "n_train_tokens": n_tr_toks,
        "n_val_docs": n_va_docs,
        "n_val_tokens": n_va_toks,
        "seq_len": SEQ_LEN,
    }
    TOKENS_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {TOKENS_META}")


if __name__ == "__main__":
    main()
