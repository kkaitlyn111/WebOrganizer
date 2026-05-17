"""Phase 4: validation runs V0-V9.

Selection pool: fresh shards (downloaded separately).
LM eval: original 10 val shards (same as diagnostic).

For each validation run:
  1. Apply selection to fresh-pool feature dataframe
  2. Gather tokens from fresh-pool pretokenized memmap
  3. Train 15M LM (same hyperparams as diagnostic)
  4. Eval on val_sequences_eval.npy
  5. Save val_loss

Validation runs:
  V0: random 15% (seed 42)
  V1: top 15% by DCLM-fasttext
  V2: top 15% by FineWeb-Edu (rounded >= 3)
  V3: top 15% by EBM predicted score
  V4: top 15% by EBM-no-PC predicted score
  V5: top 15% by Lasso predicted score
  V6: within-topic top 15% by EBM score
  V7: random 15% (seed 43)
  V8: random 15% (seed 44)
  V9: top 15% by EBM (seed 43)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import (DATA_DIR, MASKS_DIR, RESULTS_DIR, TOKENS_DIR,
                    VAL_SEQUENCES_EVAL, SEQ_LEN)  # noqa
from train_lm import TrainConfig, train_and_eval, prepare_token_stream


def build_runs(fresh_df, ebm_scores, ebm_int_scores, lasso_scores):
    n = len(fresh_df)
    runs = []

    def add(rid, name, mask):
        n_sel = int(mask.sum())
        runs.append({"rid": rid, "name": name, "mask": mask, "n_sel": n_sel})

    def random_mask(seed, frac=0.15):
        rng = np.random.default_rng(seed)
        return rng.random(n) < frac

    def top_pct_mask(scores, pct=15):
        thr = np.nanpercentile(scores, 100 - pct)
        return scores >= thr

    add("V0", "random_seed42", random_mask(42))
    add("V1", "dclm_top15", top_pct_mask(fresh_df["dclm_score"].to_numpy()))
    add("V2", "fwedu_ge3", fresh_df["fwedu_rounded"].to_numpy() >= 3)
    add("V3", "ebm_top15", top_pct_mask(ebm_scores))
    add("V4", "ebm_interp_top15", top_pct_mask(ebm_int_scores))
    add("V5", "lasso_top15", top_pct_mask(lasso_scores))
    # Within-topic top 15% by EBM
    topics = fresh_df["topic"].to_numpy()
    mask = np.zeros(n, dtype=bool)
    for t in np.unique(topics):
        tmask = topics == t
        if tmask.sum() < 50:
            continue
        thr = np.nanpercentile(ebm_scores[tmask], 85)
        mask[tmask] = ebm_scores[tmask] >= thr
    add("V6", "ebm_within_topic_top15", mask)
    add("V7", "random_seed43", random_mask(43))
    add("V8", "random_seed44", random_mask(44))
    # V9: EBM top 15% with a different seed (for the LM training)
    add("V9", "ebm_top15_seed43", top_pct_mask(ebm_scores))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-master", type=Path, required=True,
                    help="Parquet of fresh shards with all features + topic")
    ap.add_argument("--fresh-tokens", type=Path, required=True,
                    help="Memmap *_token_ids.bin of fresh-pool tokens")
    ap.add_argument("--fresh-offsets", type=Path, required=True)
    ap.add_argument("--ebm-scores", type=Path,
                    default=DATA_DIR / "fresh_ebm_scores.npy")
    ap.add_argument("--ebm-interp-scores", type=Path,
                    default=DATA_DIR / "fresh_ebm_interp_scores.npy")
    ap.add_argument("--lasso-scores", type=Path,
                    default=DATA_DIR / "fresh_lasso_scores.npy")
    ap.add_argument("--run", type=str, default=None,
                    help="If set, only run this V0-V9")
    ap.add_argument("--total-tokens", type=int, default=600_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="m4-validation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    fresh_df = pd.read_parquet(args.fresh_master)
    print(f"fresh-pool docs: {len(fresh_df):,}")

    ebm = np.load(args.ebm_scores)
    ebm_int = np.load(args.ebm_interp_scores)
    lasso = np.load(args.lasso_scores)
    runs = build_runs(fresh_df, ebm, ebm_int, lasso)

    if args.run:
        runs = [r for r in runs if r["rid"] == args.run]
    if not runs:
        raise SystemExit("No matching runs.")

    token_ids = np.memmap(args.fresh_tokens, dtype=np.uint32, mode="r")
    offsets = np.load(args.fresh_offsets)
    val_seq = np.load(VAL_SEQUENCES_EVAL, mmap_mode="r")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = DATA_DIR / "validation_results"
    out_dir.mkdir(exist_ok=True)
    mask_dir = DATA_DIR / "validation_masks"
    mask_dir.mkdir(exist_ok=True)

    for r in runs:
        rid = r["rid"]
        print(f"\n=== {rid} {r['name']}  selected={r['n_sel']:,} ===")
        t0 = time.time()
        np.save(mask_dir / f"{rid}.npy", r["mask"])
        sel_idx = np.where(r["mask"])[0]
        chunks = [np.asarray(token_ids[offsets[i]: offsets[i+1]]) for i in sel_idx]
        stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)
        print(f"  tokens: {len(stream):,}")
        # Use a separate seed for V9 to differ from V3
        run_seed = args.seed + ord(rid[1]) if rid in ("V9",) else args.seed
        rng = np.random.default_rng(run_seed + 1000)
        packed = prepare_token_stream(stream, args.total_tokens, SEQ_LEN, rng=rng)
        tcfg = TrainConfig(total_tokens=args.total_tokens, seed=run_seed)
        wandb_run = None
        if not args.no_wandb:
            try:
                import wandb
                wandb_run = wandb.init(project=args.wandb_project,
                                        name=f"{rid}-{r['name']}",
                                        config={"rid": rid, "name": r["name"],
                                                 "n_sel": r["n_sel"],
                                                 "total_tokens": args.total_tokens,
                                                 "seed": run_seed},
                                        tags=[rid], reinit=True)
            except Exception as e:
                print(f"  wandb init failed: {e}")
        result = train_and_eval(packed, val_seq, tcfg, device=device,
                                 wandb_run=wandb_run)
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        out = {"rid": rid, "name": r["name"], "n_sel": r["n_sel"],
                "wall_time_s": time.time() - t0, **result}
        (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> val_loss={result['val_loss']:.4f}")


if __name__ == "__main__":
    main()
