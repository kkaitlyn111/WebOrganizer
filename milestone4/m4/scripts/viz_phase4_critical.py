"""Better Phase-4 graphics + critical comparison of 15M vs 60M scale.

Produces:
  figures/phase4_critical/leaderboard_15m_vs_60m.png   side-by-side leaderboards
  figures/phase4_critical/scaling_lines.png            per-strategy 15M->60M scaling
  figures/phase4_critical/gap_from_random.png          gap to mean-random at each scale
  figures/phase4_critical/critical_summary.md          text writeup
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE / "figures" / "phase4_critical"
OUT.mkdir(parents=True, exist_ok=True)


def load_dir(p: Path):
    rows = []
    for f in sorted(p.glob("V*.json")):
        r = json.loads(f.read_text())
        rows.append({"rid": r["rid"], "name": r["name"],
                     "val_loss": float(r["val_loss"]),
                     "n_params_M": float(r.get("n_params", 0)) / 1e6})
    return pd.DataFrame(rows)


df_15 = load_dir(DATA / "validation_results_eq")
df_60 = load_dir(DATA / "validation_results_eq_60m_2ep")
print(df_15)
print(df_60)

# Family color map
def family(rid):
    if rid in ("V0","V7","V8"): return "Random"
    if rid in ("V3","V4","V9"): return "EBM (global top-k)"
    if rid == "V5":             return "Lasso"
    if rid == "V1":             return "DCLM"
    if rid == "V2":             return "FineWeb-Edu"
    if rid == "V6":             return "EBM within-topic"
    return "Other"

COLORS = {"Random": "#888",
          "EBM within-topic": "#085",
          "EBM (global top-k)": "#3a8",
          "Lasso": "#6cc",
          "DCLM": "#c83",
          "FineWeb-Edu": "#a33"}

for df in (df_15, df_60):
    df["family"] = df["rid"].map(family)

# Pretty short label
def short(rid, name):
    n = name.replace("_eq150M", "").replace("_seed42","(s42)").replace("_seed43","(s43)").replace("_seed44","(s44)")
    return f"{rid}: {n}"

# ============================================================
# Fig 1: side-by-side leaderboards
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
for ax, df, title in [(axes[0], df_15, "15M params  ·  150M unique tok  ·  1 epoch"),
                       (axes[1], df_60, "60M params  ·  150M unique tok  ·  2 epochs (300M total)")]:
    d = df.sort_values("val_loss").reset_index(drop=True)
    cols = [COLORS[f] for f in d["family"]]
    bars = ax.bar(range(len(d)), d["val_loss"], color=cols)
    for b, v in zip(bars, d["val_loss"]):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    labels = [short(r, n) for r, n in zip(d["rid"], d["name"])]
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("val_loss (lower = better)")
    ax.set_title(title)
    ax.set_ylim(d["val_loss"].min() - 0.02, d["val_loss"].max() + 0.025)
    # mark mean of random
    randmean = df[df["family"]=="Random"]["val_loss"].mean()
    ax.axhline(randmean, color="#888", ls=":", lw=1,
                label=f"random mean = {randmean:.3f}")
    ax.legend(loc="upper left", fontsize=8)

from matplotlib.patches import Patch
fams = ["Random","EBM within-topic","EBM (global top-k)","Lasso","FineWeb-Edu","DCLM"]
fig.legend(handles=[Patch(facecolor=COLORS[f], label=f) for f in fams],
           loc="lower center", ncol=6, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "leaderboard_15m_vs_60m.pdf"); plt.savefig(OUT / "leaderboard_15m_vs_60m.png", dpi=120)
print(f"wrote {OUT/'leaderboard_15m_vs_60m.png'}")

# ============================================================
# Fig 2: scaling lines (15M -> 60M per strategy)
# ============================================================
merged = df_15.merge(df_60, on=["rid","family"], suffixes=("_15","_60"))
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for _, r in merged.iterrows():
    c = COLORS[r["family"]]
    ax.plot([15, 60], [r["val_loss_15"], r["val_loss_60"]],
            "-o", color=c, alpha=0.85, lw=2, ms=6)
    ax.annotate(r["rid"], (60, r["val_loss_60"]), fontsize=8,
                 xytext=(6, -2), textcoords="offset points")
ax.set_xlabel("model parameters (M)")
ax.set_ylabel("val_loss")
ax.set_xscale("log")
ax.set_xticks([15, 60])
ax.set_xticklabels(["15M", "60M"])
ax.set_title("Scaling: does the gap between filters and random close with model size?")
fams = ["Random","EBM within-topic","EBM (global top-k)","Lasso","FineWeb-Edu","DCLM"]
ax.legend(handles=[Patch(facecolor=COLORS[f], label=f) for f in fams],
          loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "scaling_lines.pdf"); plt.savefig(OUT / "scaling_lines.png", dpi=120)
print(f"wrote {OUT/'scaling_lines.png'}")

# ============================================================
# Fig 3: gap from random mean (signed, both scales)
# ============================================================
rand_mean_15 = df_15[df_15["family"]=="Random"]["val_loss"].mean()
rand_mean_60 = df_60[df_60["family"]=="Random"]["val_loss"].mean()
non_rand = merged[merged["family"]!="Random"].copy()
non_rand["gap_15"] = non_rand["val_loss_15"] - rand_mean_15
non_rand["gap_60"] = non_rand["val_loss_60"] - rand_mean_60
# Sort by 60M gap
non_rand = non_rand.sort_values("gap_60")
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(non_rand))
w = 0.4
b1 = ax.bar(x - w/2, non_rand["gap_15"], width=w, color="#fc8",
            edgecolor="k", lw=0.4, label="15M params")
b2 = ax.bar(x + w/2, non_rand["gap_60"], width=w, color="#8cc",
            edgecolor="k", lw=0.4, label="60M params (2 ep)")
ax.axhline(0, color="k", lw=0.8, ls="--")
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
labels = [short(r, n) for r, n in zip(non_rand["rid"], non_rand["name_60"])]
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("val_loss  -  mean(random)")
ax.set_title("How much each filter beats / loses to the mean random run, at each scale\n"
             "(0 = ties random; positive = worse than random; negative = beats random)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "gap_from_random.pdf"); plt.savefig(OUT / "gap_from_random.png", dpi=120)
print(f"wrote {OUT/'gap_from_random.png'}")

# ============================================================
# Text summary
# ============================================================
lines = [
    "# Phase 4 critical comparison: 15M vs 60M\n",
    "\n## Raw leaderboards\n",
    "\n### 15M params, 1 epoch, 150M unique tokens (CC val)\n",
    df_15.sort_values("val_loss")[["rid","name","val_loss"]]
        .to_string(index=False, float_format="%.4f"),
    "\n\n### 60M params, 2 epochs, 150M unique tokens (CC val)\n",
    df_60.sort_values("val_loss")[["rid","name","val_loss"]]
        .to_string(index=False, float_format="%.4f"),
    f"\n\nMean random:  15M = {rand_mean_15:.4f}   60M = {rand_mean_60:.4f}\n",
]
# Numeric summary block
fam_summary = []
for fam in fams:
    sub_15 = df_15[df_15["family"]==fam]
    sub_60 = df_60[df_60["family"]==fam]
    if not len(sub_15): continue
    g15 = sub_15["val_loss"].mean() - rand_mean_15
    g60 = sub_60["val_loss"].mean() - rand_mean_60
    fam_summary.append((fam, sub_15["val_loss"].mean(), sub_60["val_loss"].mean(),
                         g15, g60, g60 - g15))
lines.append("\n## Family means (averaged across seeds where applicable)\n")
lines.append("| family | mean val_loss 15M | mean val_loss 60M | gap-vs-random 15M | gap-vs-random 60M | change with scale |\n")
lines.append("|---|---:|---:|---:|---:|---:|\n")
for fam, m15, m60, g15, g60, dg in fam_summary:
    arrow = "✅ narrows" if dg < -0.005 else ("❌ widens" if dg > 0.005 else "≈ unchanged")
    lines.append(f"| {fam} | {m15:.4f} | {m60:.4f} | {g15:+.4f} | {g60:+.4f} | {dg:+.4f} {arrow} |\n")

(OUT / "critical_summary.md").write_text("".join(lines))
print(f"wrote {OUT/'critical_summary.md'}")
