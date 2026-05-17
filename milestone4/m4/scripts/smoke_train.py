"""Smoke-test train_lm.py on a few pretokenized shards.

No master.parquet needed. Just verifies that:
  - prepare_token_stream works
  - LM training runs to completion
  - val eval returns a sane loss
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
from config import PERSHARD_TOKENS_DIR, SEQ_LEN
from train_lm import TrainConfig, train_and_eval, prepare_token_stream

# Pick the first 4 shards
toks_files = sorted(PERSHARD_TOKENS_DIR.glob("*_token_ids.npy"))[:4]
print(f"Using {len(toks_files)} shards: {[f.name for f in toks_files]}")
streams = [np.load(f, mmap_mode="r") for f in toks_files]
all_toks = np.concatenate([np.asarray(s) for s in streams])
print(f"total tokens: {len(all_toks):,}")

# Use 2 shards as "train", 2 as "val"
n_train = sum(len(s) for s in streams[:2])
train_stream = all_toks[:n_train]
val_stream = all_toks[n_train:]

# Pack
train_packed = prepare_token_stream(train_stream, total_tokens=4_000_000,
                                     seq_len=SEQ_LEN, rng=np.random.default_rng(0))
val_packed = prepare_token_stream(val_stream, total_tokens=400_000,
                                   seq_len=SEQ_LEN, rng=np.random.default_rng(1))
print(f"train_packed: {train_packed.shape}, val_packed: {val_packed.shape}")

# Train for just 200 steps
cfg = TrainConfig(total_tokens=100 * 64 * SEQ_LEN, log_every=20, seed=0,
                   bf16=True, micro_batch_size=4, eval_micro_batch=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
t0 = time.time()
result = train_and_eval(train_packed, val_packed, cfg, device=device)
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"val_loss = {result['val_loss']:.4f}")
