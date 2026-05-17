"""Run a single diagnostic experiment.

Inputs:
  --run-id INT  : key into diagnostic_run_configs.json
  --configs PATH (optional, defaults to DIAGNOSTIC_CONFIGS_JSON)
  --max-steps INT (optional, for smoke testing)

Pipeline:
  1. Load run config (features, thresholds, directions)
  2. Apply selection to master parquet -> boolean mask over train-pool docs
  3. Save mask to selection_masks/run_{rid}.npy
  4. Gather tokens for selected docs from memmap
  5. Pack into (N, SEQ_LEN+1) sequences (1B tokens, up to 4x repetition)
  6. Train 15M LM, eval on pre-packed val_sequences.npy
  7. Save val_loss + metadata to diagnostic_results/run_{rid}.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import (DIAGNOSTIC_CONFIGS_JSON, MASTER_PARQUET, MASKS_DIR,
                    RESULTS_DIR, TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS,
                    VAL_SEQUENCES_EVAL, SEQ_LEN)  # noqa
from train_lm import TrainConfig, train_and_eval, prepare_token_stream


# --------------------- selection helpers ---------------------

def apply_selection(df, cfg):
    """Apply a run config to a feature dataframe; return boolean mask."""
    n = len(df)
    typ = cfg["type"]
    if typ == "single_threshold":
        f = cfg["feature"]
        p = cfg["percentile"]
        direction = cfg["direction"]
        col = df[f].to_numpy()
        thresh = np.nanpercentile(col, p)
        mask = (col >= thresh) if direction == "above" else (col <= thresh)
    elif typ == "multi_intersection":
        mask = np.ones(n, dtype=bool)
        for sel in cfg["selectors"]:
            col = df[sel["feature"]].to_numpy()
            thresh = np.nanpercentile(col, sel["percentile"])
            m = (col >= thresh) if sel["direction"] == "above" else (col <= thresh)
            mask &= m
    elif typ == "linear_top":
        feats = cfg["features"]
        weights = np.asarray(cfg["weights"], dtype=np.float64)
        means = np.asarray(cfg["means"], dtype=np.float64)
        stds = np.asarray(cfg["stds"], dtype=np.float64)
        X = df[feats].to_numpy().astype(np.float64)
        X = (X - means) / np.where(stds > 0, stds, 1.0)
        scores = X @ weights
        top_pct = cfg["top_pct"]
        thresh = np.nanpercentile(scores, 100 - top_pct)
        mask = scores >= thresh
    elif typ == "linear_top_simple":
        feats = cfg["features"]
        top_pct = cfg["top_pct"]
        X = df[feats].to_numpy().astype(np.float64)
        # z-score then sum
        m = np.nanmean(X, axis=0); s = np.nanstd(X, axis=0)
        Z = (X - m) / np.where(s > 0, s, 1.0)
        scores = np.nansum(Z, axis=1)
        thresh = np.nanpercentile(scores, 100 - top_pct)
        mask = scores >= thresh
    elif typ == "targeted_top":
        f = cfg["feature"]
        top_pct = cfg["top_pct"]
        direction = cfg.get("direction", "above")
        col = df[f].to_numpy()
        if direction == "above":
            thresh = np.nanpercentile(col, 100 - top_pct)
            mask = col >= thresh
        else:
            thresh = np.nanpercentile(col, top_pct)
            mask = col <= thresh
    elif typ == "topic_conditional":
        topic = cfg["topic"]
        f = cfg["feature"]
        top_pct = cfg["top_pct"]
        col = df[f].to_numpy()
        topics = df["topic"].to_numpy()
        mask = np.zeros(n, dtype=bool)
        topic_mask = topics == topic
        if topic_mask.any():
            vals = col[topic_mask]
            thresh = np.nanpercentile(vals, 100 - top_pct)
            mask[topic_mask] = vals >= thresh
    elif typ == "within_topic_top":
        f = cfg["feature"]
        top_pct = cfg["top_pct"]
        col = df[f].to_numpy()
        topics = df["topic"].to_numpy()
        mask = np.zeros(n, dtype=bool)
        for t in np.unique(topics):
            tmask = topics == t
            vals = col[tmask]
            if len(vals) == 0:
                continue
            thresh = np.nanpercentile(vals, 100 - top_pct)
            mask[tmask] = vals >= thresh
    elif typ == "fwedu_geq":
        thr = cfg.get("threshold", 3)
        mask = df["fwedu_rounded"].to_numpy() >= thr
    elif typ == "random":
        rng = np.random.default_rng(cfg["seed"])
        frac = cfg["frac"]
        mask = rng.random(n) < frac
    elif typ == "dclm_top_then_restrict":
        dclm_top = cfg["dclm_top_pct"]
        f = cfg["restrict_feature"]
        restrict_top = cfg["restrict_top_pct"]
        d_col = df["dclm_score"].to_numpy()
        d_thr = np.nanpercentile(d_col, 100 - dclm_top)
        m1 = d_col >= d_thr
        f_col = df[f].to_numpy()
        f_thr = np.nanpercentile(f_col[m1], 100 - restrict_top)
        mask = m1 & (f_col >= f_thr)
    else:
        raise ValueError(f"Unknown selection type: {typ}")
    return mask


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--configs", type=Path, default=DIAGNOSTIC_CONFIGS_JSON)
    ap.add_argument("--master", type=Path, default=MASTER_PARQUET)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override total steps for smoke test")
    ap.add_argument("--total-tokens", type=int, default=300_000_000)
    ap.add_argument("--unique-tokens", type=int, default=150_000_000,
                    help="Subsample selection to exactly this many UNIQUE tokens. "
                          "Drop the run if selection has fewer. Set 0 to disable.")
    ap.add_argument("--micro-batch-size", type=int, default=None,
                    help="Auto-pick based on GPU memory if not set")
    ap.add_argument("--wandb-project", default="datamixes")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configs = json.loads(args.configs.read_text())
    cfg = next((c for c in configs if c["run_id"] == args.run_id), None)
    if cfg is None:
        raise SystemExit(f"run_id {args.run_id} not in {args.configs}")

    print(f"=== Run {args.run_id} ({cfg['type']}) ===")
    t0 = time.time()

    import pandas as pd
    df = pd.read_parquet(args.master)
    # Restrict to train-pool docs
    df = df[df["__in_train_pool"]].reset_index(drop=True)
    print(f"  train-pool docs: {len(df):,}")

    mask = apply_selection(df, cfg)
    n_sel = int(mask.sum())
    print(f"  selected: {n_sel:,} ({100*n_sel/len(mask):.2f}%)")
    if n_sel < 200:
        print(f"  WARNING: too few docs selected. Aborting run.")
        out = {"run_id": args.run_id, "val_loss": None, "n_selected": n_sel,
                "error": "too_few_docs"}
        (RESULTS_DIR / f"run_{args.run_id:04d}.json").write_text(json.dumps(out, indent=2))
        return

    # Save mask (boolean, against train-pool order)
    np.save(MASKS_DIR / f"run_{args.run_id:04d}.npy", mask)

    # Load token streams
    token_ids = np.memmap(TRAIN_TOKEN_IDS, dtype=np.uint32, mode="r")
    offsets = np.load(TRAIN_DOC_OFFSETS)
    # NOTE: pretokenized docs are ordered by train-shard order, and
    # the master parquet is ordered the same way (we enforce this in 03_build_master.py).
    sel_idx = np.where(mask)[0]

    # Compute per-doc token counts cheaply via offset diffs
    doc_tok_counts = (offsets[sel_idx + 1] - offsets[sel_idx]).astype(np.int64)
    total_unique_tokens = int(doc_tok_counts.sum())
    print(f"  selection has {total_unique_tokens:,} unique tokens")

    # Enforce equal unique-token budget U
    rng = np.random.default_rng(args.seed + args.run_id)
    if args.unique_tokens > 0:
        if total_unique_tokens < args.unique_tokens:
            print(f"  DROP: selection has {total_unique_tokens:,} < "
                  f"{args.unique_tokens:,} (U). Aborting run.")
            out = {"run_id": args.run_id, "val_loss": None,
                    "n_selected": n_sel,
                    "n_unique_tokens": total_unique_tokens,
                    "error": "selection_below_unique_token_budget"}
            (RESULTS_DIR / f"run_{args.run_id:04d}.json").write_text(json.dumps(out, indent=2))
            return
        # Shuffle docs, take docs until token-sum crosses U.
        perm = rng.permutation(len(sel_idx))
        cumsum = np.cumsum(doc_tok_counts[perm])
        k = int(np.searchsorted(cumsum, args.unique_tokens) + 1)
        keep = perm[:k]
        sel_idx_kept = sel_idx[keep]
        print(f"  subsampled to {k:,} docs ({int(cumsum[k-1]):,} unique tokens)")
        # Save the subsampled mask (overwrite earlier full-selection mask)
        sub_mask = np.zeros_like(mask)
        sub_mask[sel_idx_kept] = True
        np.save(MASKS_DIR / f"run_{args.run_id:04d}.npy", sub_mask)
        sel_idx = sel_idx_kept

    # Build the token stream by slicing memmap (lazy).
    print(f"  gathering tokens for {len(sel_idx):,} docs from memmap...")
    t1 = time.time()
    chunks = [np.asarray(token_ids[offsets[i]: offsets[i+1]]) for i in sel_idx]
    stream = np.concatenate(chunks)
    print(f"  total selected tokens: {len(stream):,} (in {time.time()-t1:.1f}s)")

    # Pack (rng was already constructed above for sub-sampling)
    packed = prepare_token_stream(stream, args.total_tokens, SEQ_LEN, rng=rng)
    print(f"  packed train sequences: {packed.shape}")

    # Val sequences (subsampled for fast eval)
    val_seq = np.load(VAL_SEQUENCES_EVAL, mmap_mode="r")
    print(f"  val sequences: {val_seq.shape}")

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro = args.micro_batch_size
    if micro is None and torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Heuristic: 40GB A100 -> 16; 24GB 3090 -> 4
        # Tier: 24GB->4, 40GB->16, 49GB->32, 80GB->64, 141GB->96
        if gpu_mem_gb >= 100:        micro = 96
        elif gpu_mem_gb >= 70:       micro = 64
        elif gpu_mem_gb >= 45:       micro = 32
        elif gpu_mem_gb >= 35:       micro = 16
        else:                         micro = 4
        print(f"  auto micro_batch_size={micro} (gpu_mem={gpu_mem_gb:.1f}GB)")
    tcfg = TrainConfig(total_tokens=args.total_tokens,
                        seed=args.seed + args.run_id,
                        micro_batch_size=micro or 16,
                        eval_micro_batch=micro or 16)
    if args.max_steps is not None:
        tcfg.total_tokens = args.max_steps * tcfg.batch_size * tcfg.seq_len

    # Optional wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = f"diag-{args.run_id:04d}-{cfg['type']}"
            wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                                    config={**{k: v for k, v in cfg.items()
                                                if k not in ("weights", "means", "stds")},
                                            "n_selected": n_sel,
                                            "n_selected_tokens": int(len(stream)),
                                            "total_tokens": args.total_tokens,
                                            "micro_batch_size": tcfg.micro_batch_size,
                                            "seed": tcfg.seed},
                                    tags=[cfg["type"]], reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}; continuing without wandb")
            wandb_run = None

    result = train_and_eval(packed, val_seq, tcfg, device=device,
                             wandb_run=wandb_run)
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    out = {
        "run_id": args.run_id,
        "cfg": cfg,
        "n_selected": n_sel,
        "n_selected_tokens": int(len(stream)),
        "wall_time_s": time.time() - t0,
        **result,
    }
    out_path = RESULTS_DIR / f"run_{args.run_id:04d}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"=== run {args.run_id} val_loss={result['val_loss']:.4f}  wall={out['wall_time_s']:.1f}s  -> {out_path}")


if __name__ == "__main__":
    main()
