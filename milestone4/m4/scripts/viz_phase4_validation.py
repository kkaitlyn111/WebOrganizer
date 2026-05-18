"""Analyze the 10 validation runs (V0-V9) and check whether our learned
per-doc score (Lasso / EBM / EBM-no-PC) actually predicts LM val_loss
out-of-distribution.

Outputs:
  figures/phase4/val_loss_leaderboard.png
  figures/phase4/pred_vs_val_loss.png
  figures/phase4/calibration_table.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
RES = DATA / "validation_results"
OUT = HERE / "figures" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)


def random_mask(n, seed, frac=0.15):
    rng = np.random.default_rng(seed)
    k = int(frac * n)
    idx = rng.choice(n, size=k, replace=False)
    m = np.zeros(n, dtype=bool); m[idx] = True
    return m


def top_pct_mask(scores, pct=15):
    thr = np.nanpercentile(scores, 100 - pct)
    return (scores >= thr) & np.isfinite(scores)


def build_masks(fresh_df, ebm_scores, ebm_int_scores, lasso_scores):
    n = len(fresh_df)
    runs = {}
    runs["V0"] = ("random_seed42",        random_mask(n, 42))
    runs["V1"] = ("dclm_top15",           top_pct_mask(fresh_df["dclm_score"].to_numpy()))
    runs["V2"] = ("fwedu_ge3",            fresh_df["fwedu_rounded"].to_numpy() >= 3)
    runs["V3"] = ("ebm_top15",            top_pct_mask(ebm_scores))
    runs["V4"] = ("ebm_interp_top15",     top_pct_mask(ebm_int_scores))
    runs["V5"] = ("lasso_top15",          top_pct_mask(lasso_scores))
    # V6: within-topic top 15% by EBM
    topics = fresh_df["topic"].to_numpy()
    mask = np.zeros(n, dtype=bool)
    for t in np.unique(topics[~pd.isna(topics)]):
        tmask = topics == t
        if tmask.sum() < 50:
            continue
        thr = np.nanpercentile(ebm_scores[tmask], 85)
        mask[tmask] = ebm_scores[tmask] >= thr
    runs["V6"] = ("ebm_within_topic_top15", mask)
    runs["V7"] = ("random_seed43",        random_mask(n, 43))
    runs["V8"] = ("random_seed44",        random_mask(n, 44))
    runs["V9"] = ("ebm_top15_seed43",     top_pct_mask(ebm_scores))
    return runs


def main():
    fresh_df = pd.read_parquet(DATA / "fresh_master.parquet")
    ebm = np.load(DATA / "fresh_ebm_scores.npy")
    ebm_int = np.load(DATA / "fresh_ebm_interp_scores.npy")
    lasso = np.load(DATA / "fresh_lasso_scores.npy")
    print(f"fresh docs: {len(fresh_df):,}")

    # Prefer per-run masks saved by 09b_run_validation_equal_tokens.py;
    # fall back to reconstructed top-15% masks for the old protocol.
    runs = build_masks(fresh_df, ebm, ebm_int, lasso)
    for rid in list(runs):
        saved = RES / f"{rid}_mask.npy"
        if saved.exists():
            runs[rid] = (runs[rid][0], np.load(saved))

    # Load actual val_loss results
    rows = []
    for rid, (name, mask) in runs.items():
        rp = RES / f"{rid}.json"
        if not rp.exists():
            print(f"  WARN: {rp} missing — run may still be in flight")
            continue
        r = json.loads(rp.read_text())
        row = {
            "rid": rid,
            "name": name,
            "n_sel": int(mask.sum()),
            "val_loss": r.get("val_loss"),
            "frac_selected": mask.mean(),
            "mean_pred_ebm": float(np.nanmean(ebm[mask])),
            "mean_pred_ebm_interp": float(np.nanmean(ebm_int[mask])),
            "mean_pred_lasso": float(np.nanmean(lasso[mask])),
            "mean_dclm_sel": float(np.nanmean(fresh_df.loc[mask, "dclm_score"])),
            "mean_fwedu_sel": float(np.nanmean(fresh_df.loc[mask, "fwedu_rounded"])),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "calibration_table.csv", index=False)
    print(f"wrote {OUT/'calibration_table.csv'}")

    if df["val_loss"].isna().all() or len(df) < 3:
        print("Not enough results yet to plot.")
        return

    # ============================================================
    # Fig 1: val_loss leaderboard
    # ============================================================
    df_sorted = df.sort_values("val_loss")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = []
    for n in df_sorted["name"]:
        if n.startswith("random"):       palette.append("#888")
        elif "ebm" in n and "interp" in n: palette.append("#085")
        elif "ebm" in n:                  palette.append("#3a8")
        elif "lasso" in n:                palette.append("#6cc")
        elif "dclm" in n:                 palette.append("#c83")
        elif "fwedu" in n:                palette.append("#a33")
        else:                              palette.append("#999")
    labels = [f"{r}: {n}" for r, n in zip(df_sorted["rid"], df_sorted["name"])]
    bars = ax.bar(range(len(df_sorted)), df_sorted["val_loss"], color=palette)
    for b, v in zip(bars, df_sorted["val_loss"]):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("val_loss (lower = better)")
    ax.set_title("Validation runs (V0-V9) on fresh shards — 150M tokens each")
    ax.margins(x=0.02)
    plt.tight_layout()
    plt.savefig(OUT / "val_loss_leaderboard.pdf"); plt.savefig(OUT / "val_loss_leaderboard.png", dpi=120)
    print(f"wrote {OUT/'val_loss_leaderboard.png'}")

    # ============================================================
    # Fig 2: mean predicted-y vs actual val_loss
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    from scipy.stats import pearsonr, spearmanr
    for ax, col, label in zip(axes,
                              ["mean_pred_ebm", "mean_pred_ebm_interp", "mean_pred_lasso"],
                              ["EBM (full)", "EBM (no PC)", "Lasso"]):
        valid = df[col].notna() & df["val_loss"].notna()
        x = df.loc[valid, col].to_numpy()
        y = df.loc[valid, "val_loss"].to_numpy()
        rids = df.loc[valid, "rid"].to_list()
        ax.scatter(x, y, s=80, color="#085", edgecolor="k")
        for xi, yi, r in zip(x, y, rids):
            ax.annotate(r, (xi, yi), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        if len(x) >= 3:
            # linear fit
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b0 + b1*xs, color="red", lw=1.0, ls="--",
                    label=f"slope={b1:+.3f}")
            r_p, _ = pearsonr(x, y)
            r_s, _ = spearmanr(x, y)
            ax.set_title(f"{label}\nPearson r={r_p:+.3f}  Spearman ρ={r_s:+.3f}",
                          fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title(label)
        ax.set_xlabel(f"mean({col.replace('mean_pred_','predicted_y_')}) on selection")
        ax.set_ylabel("actual val_loss")
    plt.suptitle("Out-of-distribution calibration: do per-doc predicted scores predict LM val_loss?",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT / "pred_vs_val_loss.pdf"); plt.savefig(OUT / "pred_vs_val_loss.png", dpi=120)
    print(f"wrote {OUT/'pred_vs_val_loss.png'}")

    # ============================================================
    # Markdown summary
    # ============================================================
    lines = ["# Phase 4 — validation analysis\n\n"]
    lines.append("## Calibration table\n\n")
    lines.append("| rid | strategy | n_sel | val_loss | mean(EBM) | mean(EBM-noPC) | mean(Lasso) | mean(DCLM) | mean(FWedu) |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df.iterrows():
        lines.append(f"| {r['rid']} | {r['name']} | {r['n_sel']:,} | "
                     f"{r['val_loss']:.4f} | {r['mean_pred_ebm']:+.4f} | "
                     f"{r['mean_pred_ebm_interp']:+.4f} | {r['mean_pred_lasso']:+.4f} | "
                     f"{r['mean_dclm_sel']:+.4f} | {r['mean_fwedu_sel']:.2f} |\n")
    # winner / loser
    best = df.loc[df["val_loss"].idxmin()]
    worst = df.loc[df["val_loss"].idxmax()]
    lines.append(f"\n**Best run:** {best['rid']} ({best['name']}) at val_loss = {best['val_loss']:.4f}\n")
    lines.append(f"\n**Worst run:** {worst['rid']} ({worst['name']}) at val_loss = {worst['val_loss']:.4f}\n")

    # noise floor estimate from V0/V7/V8 randoms
    rand = df[df["rid"].isin(["V0", "V7", "V8"])]
    if len(rand) > 1:
        lines.append(f"\nRandom-baseline noise: 3 runs span {rand['val_loss'].min():.4f} — {rand['val_loss'].max():.4f} "
                      f"(spread {rand['val_loss'].max()-rand['val_loss'].min():.4f} nats)\n")
    (OUT / "calibration_table.md").write_text("".join(lines))
    print(f"wrote {OUT/'calibration_table.md'}")


if __name__ == "__main__":
    main()
