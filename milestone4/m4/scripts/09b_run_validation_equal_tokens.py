"""Phase 4 validation runs — corrected protocol.

Each strategy selects EXACTLY U=150M unique tokens (matching Phase 1's
diagnostic-mix protocol). All runs train on 1 epoch of those 150M tokens.

Selection by strategy (replaces 09_run_validation.py top-15% logic):
  V0/V7/V8 random_seedX     : shuffle docs by (seed), keep until cumtokens >= U
  V1 dclm                   : sort by dclm desc, keep until cumtokens >= U
  V2 fwedu                  : sort by fwedu desc (random tie-break), keep until U
  V3 ebm / V9 ebm_seed43    : sort by ebm score desc, keep until U
  V4 ebm_interp             : sort by ebm_interp desc, keep until U
  V5 lasso                  : sort by lasso desc, keep until U
  V6 ebm_within_topic       : within each topic, allocate U * (topic share) tokens
                              by selecting top-EBM docs in that topic until quota.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import DATA_DIR, MODELS_DIR  # noqa
from train_lm import TrainConfig, train_and_eval, prepare_token_stream  # noqa
from config import VAL_SEQUENCES_EVAL, SEQ_LEN  # noqa

U = 150_000_000


def select_by_score_until_tokens(scores: np.ndarray, tokens: np.ndarray,
                                 budget: int, seed: int = 0) -> np.ndarray:
    """Sort docs by `scores` descending (random tie-break), keep until cumulative
    `tokens` >= budget. Returns a boolean mask aligned with scores."""
    n = len(scores)
    rng = np.random.default_rng(seed)
    # NaN-safe sort: NaNs sink to the bottom (treated as worst)
    s = np.where(np.isnan(scores), -np.inf, scores)
    # break ties uniformly so we don't always pick the lexicographically first doc
    jitter = rng.uniform(0, 1e-9, size=n)
    order = np.lexsort((jitter, -s))   # primary: -s desc; secondary: jitter
    csum = np.cumsum(tokens[order])
    k = int(np.searchsorted(csum, budget, side="left") + 1)
    k = min(k, n)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def select_random_until_tokens(tokens: np.ndarray, budget: int, seed: int):
    return select_by_score_until_tokens(np.zeros_like(tokens, dtype=float),
                                        tokens, budget, seed=seed)


def select_within_topic_until_tokens(scores, tokens, topics, budget, seed=0):
    """Rank docs by within-topic percentile of `scores`, then apply a single
    cumulative-tokens cutoff. Guarantees the selection hits `budget` unique
    tokens (matching Phase-1 equal-budget protocol) while preferring
    top-of-topic docs in every topic uniformly.
    """
    n = len(scores)
    rank = np.full(n, np.nan, dtype=float)
    for t in np.unique(topics[~pd.isna(topics)]):
        tmask = topics == t
        if tmask.sum() < 50:
            continue
        sub = scores[tmask]
        # percentile rank within topic (NaN-safe). Larger = better.
        # rankdata-style without scipy: argsort of argsort, mapped to [0,1].
        s = np.where(np.isnan(sub), -np.inf, sub)
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(order)) / max(1, len(order) - 1)
        rank[tmask] = ranks
    return select_by_score_until_tokens(rank, tokens, budget, seed=seed)


def gather_tokens(mask: np.ndarray, token_ids_mm: np.memmap,
                  doc_offsets: np.ndarray) -> np.ndarray:
    chunks = []
    sel_idx = np.where(mask)[0]
    for i in sel_idx:
        s, e = int(doc_offsets[i]), int(doc_offsets[i + 1])
        chunks.append(np.asarray(token_ids_mm[s:e], dtype=np.uint32))
    if not chunks:
        return np.zeros(0, dtype=np.uint32)
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="V0..V9")
    ap.add_argument("--budget", type=int, default=U)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="m4-validation-eq")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--model-preset", default="15m", choices=["15m", "60m", "120m"])
    ap.add_argument("--run-bench", action="store_true")
    ap.add_argument("--bench-subset", type=int, default=None)
    ap.add_argument("--out-suffix", default="",
                    help="appended to validation_results_eq dir + .json filename")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="save bf16 state_dict to validation_results_eq{suffix}/{rid}_ckpt.pt")
    ap.add_argument("--epochs", type=int, default=1,
                    help="number of times to repeat the selected tokens (total training tokens = budget * epochs)")
    args = ap.parse_args()

    fresh_df = pd.read_parquet(DATA_DIR / "fresh_master.parquet")
    ebm = np.load(DATA_DIR / "fresh_ebm_scores.npy")
    ebm_int = np.load(DATA_DIR / "fresh_ebm_interp_scores.npy")
    lasso = np.load(DATA_DIR / "fresh_lasso_scores.npy")
    tokens = fresh_df["token_count"].to_numpy().astype(np.int64)
    topics = fresh_df["topic"].to_numpy()

    rid = args.run
    if rid in ("V0",):  rseed, name = 42, "random_eq150M_seed42"
    elif rid == "V7":   rseed, name = 43, "random_eq150M_seed43"
    elif rid == "V8":   rseed, name = 44, "random_eq150M_seed44"
    elif rid == "V1":   name = "dclm_eq150M";        mask = select_by_score_until_tokens(fresh_df["dclm_score"].to_numpy(), tokens, args.budget, seed=0)
    elif rid == "V2":   name = "fwedu_eq150M";       mask = select_by_score_until_tokens(fresh_df["fwedu_rounded"].to_numpy(), tokens, args.budget, seed=0)
    elif rid == "V3":   name = "ebm_eq150M";          mask = select_by_score_until_tokens(ebm, tokens, args.budget, seed=0)
    elif rid == "V4":   name = "ebm_interp_eq150M";   mask = select_by_score_until_tokens(ebm_int, tokens, args.budget, seed=0)
    elif rid == "V5":   name = "lasso_eq150M";        mask = select_by_score_until_tokens(lasso, tokens, args.budget, seed=0)
    elif rid == "V6":   name = "ebm_within_topic_eq150M"; mask = select_within_topic_until_tokens(ebm, tokens, topics, args.budget)
    elif rid == "V9":   name = "ebm_eq150M_seed43";   mask = select_by_score_until_tokens(ebm, tokens, args.budget, seed=43)
    else: raise SystemExit(f"unknown rid {rid}")

    if rid in ("V0", "V7", "V8"):
        mask = select_random_until_tokens(tokens, args.budget, rseed)

    n_sel = int(mask.sum())
    unique_tokens = int(tokens[mask].sum())
    print(f"=== {rid} {name}  n_sel={n_sel:,}  unique_tokens={unique_tokens:,} ===")

    # Save mask for later re-analysis
    suffix = args.out_suffix
    out_dir = DATA_DIR / f"validation_results_eq{suffix}"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"{rid}_mask.npy", mask)

    # Gather tokens
    tok_bin = DATA_DIR / "tokens" / "fresh_token_ids.bin"
    off_npy = DATA_DIR / "tokens" / "fresh_doc_offsets.npy"
    tok_mm = np.memmap(tok_bin, dtype=np.uint32, mode="r")
    offsets = np.load(off_npy)
    print(f"gathering {n_sel:,} docs from memmap...")
    t0 = time.time()
    stream = gather_tokens(mask, tok_mm, offsets)
    print(f"  total tokens gathered: {len(stream):,}  in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed + ord(rid[1]))
    total_train_tokens = args.budget * args.epochs
    packed = prepare_token_stream(stream, total_train_tokens, SEQ_LEN, rng=rng)
    print(f"  packed shape: {packed.shape}  (epochs={args.epochs}, total_train_tokens={total_train_tokens:,})")

    val_seq = np.load(VAL_SEQUENCES_EVAL)
    ckpt_path = None
    if args.save_checkpoint:
        ckpt_path = str(out_dir / f"{rid}_ckpt.pt")
    tcfg = TrainConfig(total_tokens=total_train_tokens, seed=args.seed + ord(rid[1]),
                       model_preset=args.model_preset,
                       run_bench=args.run_bench,
                       bench_subset=args.bench_subset,
                       save_checkpoint=ckpt_path)

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"{rid}_{name}",
                   config={"rid": rid, "name": name, "n_sel": n_sel,
                           "unique_tokens": unique_tokens},
                   tags=[rid], reinit=True)

    result = train_and_eval(packed, val_seq, tcfg, device="cuda")

    out = {"rid": rid, "name": name, "n_sel": n_sel,
           "unique_tokens": unique_tokens,
           "val_loss": float(result["val_loss"]),
           "model_preset": args.model_preset,
           "total_tokens": args.budget,
           "n_params": int(result.get("n_params", 0)),
           "wall_time_s": float(result.get("wall_time_s", 0)),
           "tokens_per_sec": float(result.get("tokens_per_sec", 0))}
    # add bench results if present
    for k in ("hellaswag_acc","hellaswag_loss","piqa_acc","piqa_loss",
              "arc_easy_acc","arc_easy_loss","lambada_acc","lambada_loss"):
        if k in result:
            out[k] = float(result[k])
    (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_dir/f'{rid}.json'}")


if __name__ == "__main__":
    main()
