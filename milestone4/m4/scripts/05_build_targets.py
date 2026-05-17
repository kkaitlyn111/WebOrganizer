"""Phase 2: build document-level quality targets from diagnostic run results.

Reads:
  - diagnostic_run_configs.json
  - selection_masks/run_NNNN.npy  (boolean over train-pool docs, in train-pool order)
  - diagnostic_results/run_NNNN.json  (val_loss)
  - master.parquet (for sanity checks vs DCLM, topic, etc.)

Writes:
  - data/document_quality_target.npy  (float32 of length n_train_pool)
  - data/target_sanity.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (DIAGNOSTIC_CONFIGS_JSON, MASTER_PARQUET, MASKS_DIR,
                    RESULTS_DIR, DATA_DIR)  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DATA_DIR / "document_quality_target.npy")
    ap.add_argument("--require-min", type=int, default=20,
                    help="Min #runs that must include a doc for it to receive a target")
    args = ap.parse_args()

    configs = json.loads(DIAGNOSTIC_CONFIGS_JSON.read_text())
    n_runs_planned = len(configs)
    # Collect valid runs (have both mask and result with val_loss)
    val_losses = []
    masks = []
    run_ids = []
    for cfg in configs:
        rid = cfg["run_id"]
        mp = MASKS_DIR / f"run_{rid:04d}.npy"
        rp = RESULTS_DIR / f"run_{rid:04d}.json"
        if not (mp.exists() and rp.exists()):
            continue
        result = json.loads(rp.read_text())
        vl = result.get("val_loss")
        if vl is None:
            continue
        val_losses.append(float(vl))
        masks.append(np.load(mp))
        run_ids.append(rid)
    print(f"Loaded {len(run_ids)}/{n_runs_planned} valid runs")

    if len(run_ids) < 10:
        raise SystemExit("Too few valid runs — at least 10 needed.")

    n_docs = len(masks[0])
    assert all(len(m) == n_docs for m in masks), "Masks have different lengths"
    val_losses = np.asarray(val_losses, dtype=np.float64)

    # Stack as (n_runs, n_docs)
    selected = np.stack(masks)
    print(f"Mask shape: {selected.shape}")

    # Convert to quality (higher = better)
    quality = -val_losses
    quality = (quality - quality.min()) / (quality.max() - quality.min() + 1e-12)

    sel_f = selected.astype(np.float32)
    n_included = sel_f.sum(axis=0)
    n_excluded = sel_f.shape[0] - n_included
    sum_included = (sel_f * quality[:, None]).sum(axis=0)
    sum_excluded = quality.sum() - sum_included
    mean_included = np.where(n_included > 0, sum_included / np.maximum(n_included, 1), 0)
    mean_excluded = np.where(n_excluded > 0, sum_excluded / np.maximum(n_excluded, 1), 0)
    y = (mean_included - mean_excluded).astype(np.float32)

    # Mask docs without enough coverage
    sparse = n_included < args.require_min
    y[sparse] = 0.0

    np.save(args.output, y)
    print(f"Wrote {args.output}: shape={y.shape}, dtype={y.dtype}")

    # ---- Sanity checks ----
    print("\n=== Sanity checks ===")
    print(f"Coverage (n_included per doc): mean={n_included.mean():.1f}, "
          f"min={n_included.min():.0f}, max={n_included.max():.0f}, "
          f"sparse={sparse.sum()} ({100*sparse.mean():.2f}%)")
    print(f"y stats: mean={y.mean():.5f}, std={y.std():.5f}, "
          f"min={y.min():.5f}, max={y.max():.5f}")

    sanity = {
        "n_runs": len(run_ids),
        "n_docs": int(n_docs),
        "val_loss_min": float(val_losses.min()),
        "val_loss_max": float(val_losses.max()),
        "val_loss_mean": float(val_losses.mean()),
        "val_loss_std": float(val_losses.std()),
        "y_mean": float(y.mean()), "y_std": float(y.std()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "coverage_mean": float(n_included.mean()),
        "coverage_min": int(n_included.min()),
        "sparse_count": int(sparse.sum()),
    }

    # Correlate with DCLM, topic-R^2
    df = pd.read_parquet(MASTER_PARQUET)
    df = df[df["__in_train_pool"]].reset_index(drop=True)
    assert len(df) == n_docs, f"master has {len(df)} but masks expect {n_docs}"

    from scipy.stats import spearmanr
    dclm = df["dclm_score"].to_numpy()
    rho, p = spearmanr(y[~sparse], dclm[~sparse])
    print(f"Spearman(y, DCLM) = {rho:.4f}  p={p:.2e}")
    sanity["spearman_y_dclm"] = float(rho)

    fwedu = df["fwedu_rounded"].to_numpy()
    rho_f, _ = spearmanr(y[~sparse], fwedu[~sparse])
    print(f"Spearman(y, FWedu) = {rho_f:.4f}")
    sanity["spearman_y_fwedu"] = float(rho_f)

    # Topic R^2 (one-way ANOVA)
    topics = df["topic"].to_numpy()
    y_full = y
    grand = y_full.mean()
    ss_total = float(((y_full - grand) ** 2).sum())
    ss_between = 0.0
    for t in np.unique(topics):
        m = topics == t
        if m.sum() > 0:
            ss_between += m.sum() * (y_full[m].mean() - grand) ** 2
    r2_topic = ss_between / max(ss_total, 1e-12)
    print(f"R^2 by topic alone = {r2_topic:.4f}")
    sanity["r2_topic"] = float(r2_topic)

    # Top/bottom 50 features summary
    order = np.argsort(y_full)
    top_idx = order[-50:]
    bot_idx = order[:50]
    for label, idx in [("BOTTOM 50", bot_idx), ("TOP 50", top_idx)]:
        sub = df.iloc[idx]
        print(f"\n{label} (mean of y={y_full[idx].mean():.4f})")
        for col in ("topic", "dclm_score", "fwedu_rounded", "qr_educational_value",
                     "textbook_quality", "entity_density", "discourse_causal",
                     "char_count", "ngram_rep_2"):
            if col in sub.columns:
                v = sub[col].mean()
                print(f"  {col}: {v:.3f}")

    (DATA_DIR / "target_sanity.json").write_text(json.dumps(sanity, indent=2))
    print(f"\nWrote sanity: {DATA_DIR / 'target_sanity.json'}")


if __name__ == "__main__":
    main()
