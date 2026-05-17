"""Repack val_token_ids.bin into (N, 1025) sequences for seq_len=1024 training.

Writes val_sequences_1024.npy and val_sequences_eval_1024.npy (subsample
of 2000 sequences for fast in-loop eval).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (TOKENS_DIR, VAL_TOKEN_IDS, VAL_SEQUENCES,
                     VAL_SEQUENCES_EVAL, SEED)  # noqa

SEQ_LEN = 1024


def main():
    print(f"loading {VAL_TOKEN_IDS}...")
    stream = np.memmap(VAL_TOKEN_IDS, dtype=np.uint32, mode="r")
    n = len(stream)
    print(f"  total val tokens: {n:,}")
    chunk = SEQ_LEN + 1
    n_seqs = n // chunk
    print(f"  packing {n_seqs:,} sequences of {chunk} tokens each")
    arr = np.array(stream[: n_seqs * chunk]).reshape(n_seqs, chunk)
    print(f"  array dtype={arr.dtype} shape={arr.shape}")

    np.save(VAL_SEQUENCES, arr)
    print(f"  wrote {VAL_SEQUENCES}")

    rng = np.random.default_rng(SEED)
    n_eval = min(2000, n_seqs)
    idx = rng.choice(n_seqs, size=n_eval, replace=False)
    np.save(VAL_SEQUENCES_EVAL, arr[idx])
    print(f"  wrote {VAL_SEQUENCES_EVAL} (n={n_eval})")


if __name__ == "__main__":
    main()
