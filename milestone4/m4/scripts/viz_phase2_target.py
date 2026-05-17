"""Quick visualizations of the Phase-2 doc-level quality target."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE / "figures" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

y = np.load(DATA / "document_quality_target.npy")
m = pd.read_parquet(DATA / "master.parquet")
val = json.load(open(DATA / "val_shard_indices.json"))
val_shards = set(val["val"])
train_mask = ~m["__shard_idx"].isin(val_shards)
m_train = m.loc[train_mask].reset_index(drop=True)
assert len(m_train) == len(y), (len(m_train), len(y))

# Per-doc coverage = how many runs included each doc. Reconstruct from masks dir.
masks_dir = DATA / "selection_masks"
sample_files = sorted(masks_dir.glob("run_*.npy"))[:200]
cov = np.zeros(len(y), dtype=np.int32)
for f in sample_files:
    cov += np.load(f).astype(np.int32)
# scale to all
cov = cov * (593 / len(sample_files))
covered = cov >= 14

# Diagnostic run val_losses
res_dir = DATA / "diagnostic_results"
losses = []
for f in sorted(res_dir.glob("run_*.json")):
    r = json.loads(f.read_text())
    if r.get("val_loss") is not None:
        losses.append(r["val_loss"])
losses = np.array(losses)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Histogram of y
ax = axes[0, 0]
ax.hist(y[covered], bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("y  (doc-level quality target)")
ax.set_ylabel("# docs")
ax.set_title(f"Phase-2 target distribution (n={covered.sum():,})")

# 2. Per-doc coverage histogram
ax = axes[0, 1]
ax.hist(cov, bins=50, color="darkorange", edgecolor="white", linewidth=0.3)
ax.set_xlabel("# diagnostic runs including this doc")
ax.set_ylabel("# docs")
ax.set_title(f"Coverage: mean={cov.mean():.1f}  median={np.median(cov):.0f}")

# 3. Per-run val_loss distribution
ax = axes[0, 2]
ax.hist(losses, bins=40, color="seagreen", edgecolor="white", linewidth=0.3)
ax.set_xlabel("val_loss (per-run mean over 2000 val seqs)")
ax.set_ylabel("# runs")
ax.set_title(f"Per-run val_loss  (n={len(losses)} runs)"
             f"\nmin={losses.min():.4f}  max={losses.max():.4f}  spread={losses.max()-losses.min():.4f}")

# 4. y vs DCLM (hexbin)
ax = axes[1, 0]
mask = covered & m_train["dclm_score"].notna().values
sub = m_train.loc[mask].sample(min(80000, mask.sum()), random_state=0)
ys = y[mask][sub.index - m_train.loc[mask].index[0]] if False else None
# safer: random sample of indices
idx = np.where(mask)[0]
rng = np.random.default_rng(0)
take = rng.choice(idx, size=min(80000, idx.size), replace=False)
ax.hexbin(m_train["dclm_score"].values[take], y[take],
          gridsize=60, cmap="viridis", mincnt=2)
from scipy.stats import spearmanr
rho_d, _ = spearmanr(m_train["dclm_score"].values[take], y[take])
ax.set_xlabel("DCLM-fastText score")
ax.set_ylabel("y (target)")
ax.set_title(f"y vs DCLM   Spearman ρ = {rho_d:.3f}")

# 5. y vs FineWeb-Edu (boxplot since rounded 0-5)
ax = axes[1, 1]
mask = covered & m_train["fwedu_rounded"].notna().values
ymask = y[mask]
fwe = m_train["fwedu_rounded"].values[mask].astype(int)
groups = [ymask[fwe == k] for k in range(6)]
ax.boxplot(groups, positions=range(6), widths=0.6, showfliers=False,
           patch_artist=True,
           boxprops=dict(facecolor="lightcoral", alpha=0.6))
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("FineWeb-Edu (rounded 0-5)")
ax.set_ylabel("y")
ax.set_title("y vs FWedu")

# 6. y vs QuRater-educational_value (hexbin)
ax = axes[1, 2]
mask = covered & m_train["qr_educational_value"].notna().values
idx = np.where(mask)[0]
take = rng.choice(idx, size=min(80000, idx.size), replace=False)
ax.hexbin(m_train["qr_educational_value"].values[take], y[take],
          gridsize=60, cmap="magma", mincnt=2)
rho_q, _ = spearmanr(m_train["qr_educational_value"].values[take], y[take])
ax.set_xlabel("QuRater educational_value")
ax.set_ylabel("y")
ax.set_title(f"y vs QuRater-edu   Spearman ρ = {rho_q:.3f}")

plt.tight_layout()
plt.savefig(OUT / "target_overview.pdf")
plt.savefig(OUT / "target_overview.png", dpi=120)
print(f"wrote {OUT/'target_overview.pdf'}  +  .png")
print(f"DCLM ρ={rho_d:.3f}   QuRater-edu ρ={rho_q:.3f}")

# 7. y by topic (boxplot)
fig, ax = plt.subplots(figsize=(14, 5))
topics = sorted(m_train["topic"].dropna().unique())
groups = []
labels = []
for t in topics:
    mask = covered & (m_train["topic"].values == t)
    if mask.sum() < 100:
        continue
    groups.append(y[mask])
    labels.append(f"{int(t)}\n(n={mask.sum()//1000}k)")
ax.boxplot(groups, positions=range(len(groups)), widths=0.6, showfliers=False,
           patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.5))
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("WebOrganizer topic id")
ax.set_ylabel("y (target)")
ax.set_title("y by topic  -- topic alone explains R^2 = 0.076")
plt.tight_layout()
plt.savefig(OUT / "target_by_topic.pdf")
plt.savefig(OUT / "target_by_topic.png", dpi=120)
print(f"wrote {OUT/'target_by_topic.pdf'}  +  .png")
