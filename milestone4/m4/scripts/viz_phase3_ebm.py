"""Phase 3 visualizations: model R2 + EBM-no-PC importances + shape functions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
MODELS = DATA / "models"
OUT = HERE / "figures" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

s = json.loads((DATA / "phase3_summary.json").read_text())

# === Fig A: model R^2 bar ===
fig, ax = plt.subplots(figsize=(6, 3.2))
names = ["Topic-only\n(ANOVA)", "Lasso\n(linear)", "EBM\n(no perplex-corr)", "EBM\n(all features)"]
r2s = [0.076, s["lasso_r2"], s["ebm_interp_r2"], s["ebm_r2"]]
colors = ["#aaa", "#6cc", "#3a8", "#085"]
bars = ax.bar(names, r2s, color=colors)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Test R²  on  doc-level y")
ax.set_ylim(0, 0.75)
ax.set_title("Predicting causal doc-level quality target from features")
plt.tight_layout()
plt.savefig(OUT / "model_r2.pdf"); plt.savefig(OUT / "model_r2.png", dpi=120)
print(f"wrote {OUT/'model_r2.png'}")

# Switch to the no-PC EBM for both importance + shape plots
ebm = joblib.load(MODELS / "ebm_interp.joblib")
glob = ebm.explain_global()
gd = glob.data()                       # overall: names + scores
all_names = gd["names"]
all_imps = gd["scores"]


def fam(n: str) -> str:
    if n.startswith("qr_"): return "#c44"
    if n in ("textbook_quality", "ai_generated"): return "#fc8"
    if n in ("entity_density", "entity_diversity", "entity_per_sentence",
             "proper_noun_density", "numeric_density"): return "#4a8"
    if n in ("topic", "format"): return "#999"
    if n.startswith(("discourse_", "explanation_", "question_",
                     "pronoun_", "first_person_", "hedge_",
                     "hapax_", "header_", "code_", "list_", "avg_paragraph",
                     "flesch")): return "#48c"
    return "#888"


# === Fig B: EBM-no-PC top-15 importances ===
order = np.argsort(all_imps)[::-1][:15]
top_names = [all_names[i] for i in order][::-1]   # bottom-up for barh
top_imps = [all_imps[i] for i in order][::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(top_names)), top_imps,
        color=[fam(n) for n in top_names])
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names, fontsize=9)
ax.set_xlabel("EBM global importance  (mean |contribution to y|)")
ax.set_title("EBM (no perplexity-corr) — top 15 features by importance")
from matplotlib.patches import Patch
leg = [Patch(facecolor="#c44", label="QuRater"),
       Patch(facecolor="#fc8", label="Textbook / AI-gen"),
       Patch(facecolor="#4a8", label="Entity / NER-like"),
       Patch(facecolor="#48c", label="Structural / readability"),
       Patch(facecolor="#999", label="Topic / format"),
       Patch(facecolor="#888", label="Length / charset")]
ax.legend(handles=leg, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "ebm_importance.pdf"); plt.savefig(OUT / "ebm_importance.png", dpi=120)
print(f"wrote {OUT/'ebm_importance.png'}")

# === Fig C: shape functions for top-6 *univariate* features ===
# Pick top-6 features whose explain_global per-feature data is univariate
# (skip pair interactions). Iterate by importance and inspect each feature's type.
top_univariate = []
order_all = np.argsort(all_imps)[::-1]
for i in order_all:
    sd = glob.data(int(i))
    if sd.get("type") == "univariate" and all_names[i] not in ("topic", "format"):
        top_univariate.append((i, all_names[i], all_imps[i], sd))
    if len(top_univariate) == 6:
        break

print("plotting shape for:", [n for _, n, _, _ in top_univariate])

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (fi, name, imp, sd) in zip(axes.flat, top_univariate):
    edges = np.asarray(sd["names"], dtype=float)   # length B+1
    scores = np.asarray(sd["scores"], dtype=float)  # length B
    upper = np.asarray(sd.get("upper_bounds"), dtype=float) if sd.get("upper_bounds") is not None else None
    lower = np.asarray(sd.get("lower_bounds"), dtype=float) if sd.get("lower_bounds") is not None else None
    mids = 0.5 * (edges[:-1] + edges[1:])
    # Clip extreme tails for cleaner reading (1st-99th percentile of bin edges)
    lo, hi = np.quantile(edges, [0.01, 0.99])
    keep = (mids >= lo) & (mids <= hi)
    mx = mids[keep]; sx = scores[keep]
    ax.plot(mx, sx, color="steelblue", lw=2)
    if upper is not None and lower is not None:
        ax.fill_between(mx, lower[keep], upper[keep],
                        color="steelblue", alpha=0.18)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(name)
    ax.set_ylabel("Δ y")
    ax.set_title(f"{name}   (imp={imp:.4f})", fontsize=10)

plt.suptitle("EBM (no perplexity-corr) — shape functions of top 6 univariate features",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "ebm_shape_top6.pdf"); plt.savefig(OUT / "ebm_shape_top6.png", dpi=120)
print(f"wrote {OUT/'ebm_shape_top6.png'}")
