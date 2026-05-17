"""Phase 6: make 8 figures for the milestone 4 report.

  Fig 1: bar chart of val_loss for Type-4 targeted diagnostic runs
  Fig 2: EBM feature importance bar chart
  Fig 3: EBM shape functions for top 6 features
  Fig 4: EBM top 2-3 interaction heatmaps
  Fig 5: Lasso vs EBM vs EBM-no-PC R^2
  Fig 6: validation val_loss bar chart (V0-V9)
  Fig 7: synthetic vs web KDE plots (skipped per user)
  Fig 8: synthetic vs web EBM score distributions (skipped per user)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (DATA_DIR, MODELS_DIR, MASKS_DIR, RESULTS_DIR,
                    DIAGNOSTIC_CONFIGS_JSON)  # noqa

FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_diag_results():
    cfgs = json.loads(DIAGNOSTIC_CONFIGS_JSON.read_text())
    rows = []
    for c in cfgs:
        rp = RESULTS_DIR / f"run_{c['run_id']:04d}.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        if r.get("val_loss") is None:
            continue
        rows.append({**c, **r})
    return pd.DataFrame(rows)


def fig1_targeted(df):
    sub = df[df["type"].isin(("targeted_top", "fwedu_geq", "random"))].copy()

    def label(row):
        t = row["type"]
        if t == "fwedu_geq":
            return "fwedu>=3"
        if t == "random":
            return f"random_seed{row.get('seed', '?')}"
        return f"{row['feature']}_{row.get('direction','top')}_p{row['top_pct']}"
    sub["label"] = sub.apply(label, axis=1)
    sub = sub.sort_values("val_loss")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(sub))))
    ax.barh(sub["label"], sub["val_loss"], color="steelblue")
    ax.set_xlabel("val loss")
    ax.set_title("Fig 1: Targeted diagnostic runs (Type 4)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_targeted.pdf"); plt.close()
    print("wrote fig1_targeted.pdf")


def fig2_ebm_importance():
    s = json.loads((DATA_DIR / "phase3_summary.json").read_text())
    top = s.get("ebm_top20", [])
    if not top:
        return
    names = [t["feature"] for t in top][::-1]
    imps = [t["importance"] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, imps, color="indigo")
    ax.set_xlabel("EBM importance (mean abs contribution)")
    ax.set_title("Fig 2: EBM feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_ebm_importance.pdf"); plt.close()
    print("wrote fig2_ebm_importance.pdf")


def fig3_shape_functions():
    ebm = joblib.load(MODELS_DIR / "ebm.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    names = gd["names"]
    scores = gd["scores"]
    order = np.argsort(-np.asarray(scores))
    top6 = [i for i in order if "&" not in names[i]][:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, idx in zip(axes.flat, top6):
        sd = glob.data(idx)
        if sd.get("type") == "univariate":
            xs = sd["names"]
            ys = sd["scores"]
            # EBM gives bin edges in xs (len = len(ys)+1 for continuous features)
            try:
                if len(xs) == len(ys) + 1:
                    centers = [(float(xs[i]) + float(xs[i+1])) / 2 for i in range(len(ys))]
                    ax.plot(centers, ys, color="indigo")
                elif len(xs) == len(ys):
                    try:
                        ax.plot([float(x) for x in xs], ys, color="indigo")
                    except (ValueError, TypeError):
                        # Categorical
                        ax.bar(range(len(xs)), ys, color="indigo")
                        ax.set_xticks(range(len(xs)))
                        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f"plot err: {e}", ha="center", va="center", fontsize=7)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(names[idx], fontsize=10)
        else:
            ax.text(0.5, 0.5, names[idx], ha="center", va="center")
    plt.suptitle("Fig 3: EBM shape functions for top 6 features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_shapes.pdf"); plt.close()
    print("wrote fig3_shapes.pdf")


def fig4_interactions():
    ebm = joblib.load(MODELS_DIR / "ebm.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    names = gd["names"]
    scores = gd["scores"]
    inter_idxs = [i for i, n in enumerate(names) if "&" in n]
    inter_idxs = sorted(inter_idxs, key=lambda i: -scores[i])[:3]
    if not inter_idxs:
        print("no interactions to plot")
        return
    fig, axes = plt.subplots(1, len(inter_idxs), figsize=(5 * len(inter_idxs), 4))
    if len(inter_idxs) == 1:
        axes = [axes]
    for ax, idx in zip(axes, inter_idxs):
        sd = glob.data(idx)
        if "scores" in sd and isinstance(sd["scores"], list) and isinstance(sd["scores"][0], list):
            im = np.asarray(sd["scores"])
            ax.imshow(im, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(im).max(), vmax=np.abs(im).max())
            ax.set_title(names[idx], fontsize=9)
    plt.suptitle("Fig 4: EBM top pairwise interactions")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_interactions.pdf"); plt.close()
    print("wrote fig4_interactions.pdf")


def fig5_r2():
    s = json.loads((DATA_DIR / "phase3_summary.json").read_text())
    labels = ["Lasso", "EBM (all)", "EBM (no PC)"]
    vals = [s["lasso_r2"], s["ebm_r2"], s.get("ebm_interp_r2", 0)]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=["gray", "indigo", "purple"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom")
    ax.set_ylabel("R^2 on held-out 20%")
    ax.set_title("Fig 5: Model R^2 comparison")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_r2.pdf"); plt.close()
    print("wrote fig5_r2.pdf")


def fig6_validation():
    out_dir = DATA_DIR / "validation_results"
    if not out_dir.exists():
        print("no validation results yet")
        return
    rows = []
    for f in sorted(out_dir.glob("V*.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("rid")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df["rid"] + ": " + df["name"], df["val_loss"], color="teal")
    for b, v in zip(bars, df["val_loss"]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("val loss")
    ax.set_title("Fig 6: Validation runs V0-V9")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_validation.pdf"); plt.close()
    print("wrote fig6_validation.pdf")


def main():
    print("=== Loading diagnostic results ===")
    df = load_diag_results()
    print(f"{len(df)} diagnostic runs with val_loss")

    if len(df):
        fig1_targeted(df)
    if (DATA_DIR / "phase3_summary.json").exists():
        fig2_ebm_importance()
        fig5_r2()
    if (MODELS_DIR / "ebm.joblib").exists():
        fig3_shape_functions()
        fig4_interactions()
    fig6_validation()


if __name__ == "__main__":
    main()
