"""More phase-3 EBM (no perplexity-corr) figures:
  1. Calibration scatter: EBM pred vs true y on test set
  2. Top pairwise interactions (heatmaps)
  3. Topic shape function (categorical bar)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
MODELS = DATA / "models"
OUT = HERE / "figures" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

# WebOrganizer topic-id -> short name (from their paper / config)
TOPIC_NAMES = {
    0: "Adult", 1: "Art & Design", 2: "Software Development",
    3: "Crime & Law", 4: "Education & Jobs", 5: "Hardware",
    6: "Entertainment", 7: "Social Life", 8: "Fashion & Beauty",
    9: "Finance & Business", 10: "Food & Dining", 11: "Games",
    12: "Health", 13: "History", 14: "Home & Hobbies",
    15: "Industrial", 16: "Literature", 17: "Politics",
    18: "Religion", 19: "Science & Technology", 20: "Software",
    21: "Sports & Fitness", 22: "Transportation", 23: "Travel",
}


# ===== Setup =====
ebm = joblib.load(MODELS / "ebm_interp.joblib")
glob = ebm.explain_global()
gd = glob.data()
feature_names = list(ebm.feature_names_in_)

m = pd.read_parquet(DATA / "master.parquet")
y_full = np.load(DATA / "document_quality_target.npy")
val_shards = set(json.load(open(DATA / "val_shard_indices.json"))["val"])
train_mask = (~m["__shard_idx"].isin(val_shards)).to_numpy()
m_tr = m.loc[train_mask].reset_index(drop=True)
ebm_pred = np.load(DATA / "ebm_interp_predicted_scores.npy")[train_mask]

# Use the same 80/20 stratified split as 06_train_ebm.py (seed=42, by topic)
from sklearn.model_selection import train_test_split
idx = np.arange(len(m_tr))
strat = m_tr["topic"].to_numpy()
_, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)
y_te = y_full[te_idx]
pred_te = ebm_pred[te_idx]
print(f"test set: {len(te_idx):,} rows")

# ============================================================
# 1. Calibration scatter (hexbin) — EBM pred vs true y
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 5.5))
hb = ax.hexbin(pred_te, y_te, gridsize=70, cmap="viridis",
               mincnt=3, extent=(-0.15, 0.15, -0.20, 0.15))
lo = min(pred_te.min(), y_te.min())
hi = max(pred_te.max(), y_te.max())
ax.plot([-0.2, 0.2], [-0.2, 0.2], color="red", lw=1.2, ls="--",
        label="y = pred")
# Binned mean line: average y inside bins of pred
bins = np.linspace(-0.12, 0.10, 23)
which = np.digitize(pred_te, bins) - 1
mids = 0.5 * (bins[:-1] + bins[1:])
means = np.array([y_te[which == i].mean() if (which == i).sum() > 50 else np.nan
                   for i in range(len(mids))])
ax.plot(mids, means, color="white", lw=2.4, marker="o", ms=4,
        label="binned mean(y | pred)")
from scipy.stats import pearsonr, spearmanr
r, _ = pearsonr(pred_te, y_te)
rho, _ = spearmanr(pred_te, y_te)
ax.set_xlabel("EBM (no PC) predicted score")
ax.set_ylabel("Causal target  y")
ax.set_title(f"Calibration on held-out test ({len(te_idx):,} docs)\n"
             f"Pearson r = {r:.3f}    Spearman ρ = {rho:.3f}    R² = 0.657")
ax.legend(loc="upper left", fontsize=8)
plt.colorbar(hb, ax=ax, label="docs / bin")
plt.tight_layout()
plt.savefig(OUT / "calibration.pdf"); plt.savefig(OUT / "calibration.png", dpi=120)
print(f"wrote {OUT/'calibration.png'}")


# ============================================================
# 2. Top pairwise interactions
# ============================================================
# Find pair terms in the global explanation.
pair_terms = []
for i, (name, imp) in enumerate(zip(gd["names"], gd["scores"])):
    if " & " in name:
        pair_terms.append((i, name, imp))
pair_terms.sort(key=lambda kv: -kv[2])
print(f"# pairwise terms: {len(pair_terms)}")
top_pairs = pair_terms[:4]
print("top pairs:", [(n, round(s, 5)) for _, n, s in top_pairs])

fig, axes = plt.subplots(1, len(top_pairs), figsize=(5 * len(top_pairs), 4.5))
if len(top_pairs) == 1:
    axes = [axes]
for ax, (idx_p, name, imp) in zip(axes, top_pairs):
    sd = glob.data(int(idx_p))
    # sd["scores"] is a 2-D matrix of shape (n_bins_x, n_bins_y)
    # sd["names"] is a list/tuple [edges_x, edges_y]
    edges_x, edges_y = sd["left_names"], sd["right_names"]
    Z = np.asarray(sd["scores"])
    # bin midpoints
    ex = np.asarray(edges_x, dtype=float) if not isinstance(edges_x[0], str) else None
    ey = np.asarray(edges_y, dtype=float) if not isinstance(edges_y[0], str) else None

    # Clip extreme outliers in colormap for legibility
    vmax = np.nanpercentile(np.abs(Z), 99) or 1e-6
    im = ax.imshow(
        Z.T, origin="lower", aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[
            float(ex[0]) if ex is not None else 0,
            float(ex[-1]) if ex is not None else Z.shape[0],
            float(ey[0]) if ey is not None else 0,
            float(ey[-1]) if ey is not None else Z.shape[1],
        ],
    )
    fx, fy = name.split(" & ")
    ax.set_xlabel(fx, fontsize=9)
    ax.set_ylabel(fy, fontsize=9)
    ax.set_title(f"{fx}  ×  {fy}\n(imp={imp:.4f})", fontsize=9)
    plt.colorbar(im, ax=ax, label="Δ y from interaction")

plt.suptitle("EBM (no perplexity-corr) — top pairwise interactions",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "ebm_interactions.pdf"); plt.savefig(OUT / "ebm_interactions.png", dpi=120)
print(f"wrote {OUT/'ebm_interactions.png'}")


# ============================================================
# 3. Topic shape function
# ============================================================
topic_idx = feature_names.index("topic")
sd = glob.data(topic_idx)
scores = np.asarray(sd["scores"])
upper = np.asarray(sd.get("upper_bounds")) if sd.get("upper_bounds") is not None else None
lower = np.asarray(sd.get("lower_bounds")) if sd.get("lower_bounds") is not None else None
# EBM unfortunately treated topic as continuous (training step did
# X[c] = X[c].astype(str) but did not pass feature_types=['nominal',...]).
# The 24 bins it created align 1:1 with the 24 integer topic ids.
assert len(scores) == 24, f"expected 24 topic bins, got {len(scores)}"
topic_ids = list(range(24))

# Order by score (most-positive first)
order = np.argsort(scores)[::-1]
levels_s = [TOPIC_NAMES.get(topic_ids[i], str(topic_ids[i])) for i in order]
scores_s = scores[order]
yerr = None
if upper is not None and lower is not None:
    yerr = np.vstack([scores_s - lower[order], upper[order] - scores_s])

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = ["#085" if v > 0 else "#a33" for v in scores_s]
ax.bar(range(len(levels_s)), scores_s, color=colors,
       yerr=yerr if yerr is not None else None,
       error_kw=dict(ecolor="k", lw=0.7, capsize=2))
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(levels_s)))
ax.set_xticklabels(levels_s, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Δ y  (EBM contribution from topic)")
imp_topic = float(gd["scores"][gd["names"].index("topic")])
ax.set_title(f"Topic shape function — green = boosts y, red = hurts y "
             f"(global importance {imp_topic:.4f})")
plt.tight_layout()
plt.savefig(OUT / "ebm_topic_shape.pdf"); plt.savefig(OUT / "ebm_topic_shape.png", dpi=120)
print(f"wrote {OUT/'ebm_topic_shape.png'}")
