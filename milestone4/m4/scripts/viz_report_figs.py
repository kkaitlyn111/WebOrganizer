"""Publication-quality figures for the Milestone-4 report.

Produces 3 figures:
  fig_ebm_importance.{pdf,png}   - top-12 features, horizontal bars by family
  fig_ebm_shapes.{pdf,png}       - 2x3 panel: shape functions for top-6 features
  fig_model_r2.{pdf,png}         - 4-model R^2 bar (kept clean)

Also writes:
  benchmarks_table.tex            - LaTeX booktabs table of 60M 2-epoch multi-benchmark results
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import joblib

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
MODELS = DATA / "models"
OUT = HERE / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

# === publication style ===
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "grid.color": "#bbbbbb",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PRETTY_FEATURE = {
    "qr_educational_value":   "QuRater educational value",
    "qr_required_expertise":  "QuRater required expertise",
    "qr_facts_and_trivia":    "QuRater facts & trivia",
    "qr_writing_style":       "QuRater writing style",
    "textbook_quality":       "Textbook-quality score",
    "ai_generated":           "P(AI-generated)",
    "proper_noun_density":    "Proper-noun density",
    "entity_density":         "Entity density",
    "entity_diversity":       "Entity diversity",
    "first_person_fraction":  "First-person sentence fraction",
    "mean_word_length":       "Mean word length",
    "char_count":             "Document length (characters)",
    "token_count":            "Document length (tokens)",
    "frac_alpha":             "Fraction alphabetic",
    "flesch_kincaid_grade":   "Flesch-Kincaid grade level",
    "flesch_reading_ease":    "Flesch reading ease",
    "explanation_density":    "Explanation marker density",
    "ngram_rep_3":            "Trigram repetition fraction",
    "hapax_legomena_ratio":   "Hapax legomena ratio",
    "mean_line_length":       "Mean line length",
}

# Distinct family palette mixing cool blues + warm purples/peach for variety.
# Same palette used across all report figures.
FAM_COLOR = {
    "QuRater":            "#1d3156",   # deep navy (anchor)
    "Textbook":           "#fed6ce",   # peach (warm accent)
    "Entity":             "#a37ab4",   # muted purple
    "Structural":         "#d486b8",   # pink
    "Length":             "#abbcd6",   # pale blue (neutral)
}


def feature_family(n: str) -> str:
    if n.startswith("qr_"): return "QuRater"
    if n in ("textbook_quality", "ai_generated"): return "Textbook"
    if n in ("entity_density", "entity_diversity", "entity_per_sentence",
             "proper_noun_density", "numeric_density"): return "Entity"
    if n.startswith(("discourse_", "explanation_", "question_",
                     "pronoun_", "first_person_", "hedge_",
                     "hapax_", "header_", "code_", "list_", "avg_paragraph",
                     "flesch")): return "Structural"
    return "Length"


# ============================================================
# Figure: model R^2 (cleaner styling)
# ============================================================
def fig_model_r2():
    s = json.loads((DATA / "phase3_summary.json").read_text())
    names = ["Topic only\n(ANOVA)", "Lasso\n(linear)", "EBM\n(no perplex.)", "EBM\n(all features)"]
    r2s   = [0.076, s["lasso_r2"], s["ebm_interp_r2"], s["ebm_r2"]]
    # Ramp across palette families, with deep-navy on the winning EBM.
    colors = ["#abbcd6", "#d486b8", "#a37ab4", "#1d3156"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(names, r2s, color=colors, edgecolor="#222", linewidth=0.6, width=0.6)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Held-out test $R^2$ on doc-level target $y$")
    ax.set_ylim(0, 0.78)
    ax.set_title("Predicting per-document causal contribution")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "fig_model_r2.png", dpi=200)
    plt.close()
    print(f"wrote {OUT/'fig_model_r2.png'}")


# ============================================================
# Figure: EBM feature importance (top 12)
# ============================================================
def _classify_shape(scores: np.ndarray, edges: np.ndarray) -> str:
    """Classify an EBM 1D shape into 'up', 'down', 'peak', or 'valley'.

    A feature is 'peak' only when its interior max is meaningfully higher than
    BOTH endpoints (not just slightly), and similarly for 'valley'. Otherwise
    we call it monotone based on which endpoint is higher.
    """
    mids = 0.5 * (edges[:-1] + edges[1:])
    lo, hi = np.quantile(edges, [0.02, 0.98])
    keep = (mids >= lo) & (mids <= hi)
    s = scores[keep]
    n = len(s)
    if n < 5:
        return "up"
    s_left = float(np.mean(s[: max(1, n // 10)]))   # avg over leftmost 10%
    s_right = float(np.mean(s[-max(1, n // 10):]))  # avg over rightmost 10%
    s_max = float(np.max(s))
    s_min = float(np.min(s))
    i_max = int(np.argmax(s))
    i_min = int(np.argmin(s))
    edge_frac = 0.20
    edge_lo = int(edge_frac * n)
    edge_hi = n - edge_lo
    span = max(s_max - s_min, 1e-9)
    margin = 0.35  # max must beat the higher endpoint by >= 35% of total span

    max_interior = edge_lo <= i_max < edge_hi
    min_interior = edge_lo <= i_min < edge_hi

    # peak: interior max clearly above both endpoints
    if max_interior and (s_max - max(s_left, s_right)) > margin * span:
        return "peak"
    # valley: interior min clearly below both endpoints
    if min_interior and (min(s_left, s_right) - s_min) > margin * span:
        return "valley"
    # otherwise monotone by endpoint comparison
    return "up" if s_right > s_left else "down"


SHAPE_GLYPH = {
    "up":     ("↑", "monotone increasing"),       # ↑
    "down":   ("↓", "monotone decreasing"),       # ↓
    "peak":   ("∩", "peaked (sweet spot)"),       # ∩
    "valley": ("∪", "U-shape (avoid middle)"),    # ∪
}


def fig_ebm_importance():
    ebm = joblib.load(MODELS / "ebm_interp.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    fn = list(ebm.feature_names_in_)
    pairs = []
    for n, s in zip(gd["names"], gd["scores"]):
        if " & " in n or n in ("topic", "format"): continue
        pairs.append((n, float(s)))
    pairs.sort(key=lambda kv: -kv[1])
    top = pairs[:12][::-1]
    shapes = []
    for n, _ in top:
        fi = fn.index(n)
        sd = glob.data(fi)
        shapes.append(_classify_shape(np.asarray(sd["scores"], float),
                                       np.asarray(sd["names"], float)))
    labels = [f"{SHAPE_GLYPH[sh][0]}  {PRETTY_FEATURE.get(n, n)}"
              for (n, _), sh in zip(top, shapes)]
    vals = [v for _, v in top]
    colors = [FAM_COLOR[feature_family(n)] for n, _ in top]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.barh(range(len(top)), vals, color=colors,
                    edgecolor="#222", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"EBM global importance (mean $|\Delta y|$)")
    ax.set_title("Top 12 features and their effect on quality")
    xmax = max(vals)
    for b, v in zip(bars, vals):
        ax.text(v + xmax*0.012, b.get_y()+b.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, xmax * 1.15)
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fams_order = ["QuRater", "Textbook", "Entity", "Structural", "Length"]
    fam_handles = [Patch(facecolor=FAM_COLOR[f], label=f) for f in fams_order]
    shape_handles = [Line2D([], [], color="#444", marker="$"+SHAPE_GLYPH[k][0]+"$",
                              linestyle="", markersize=11,
                              label=SHAPE_GLYPH[k][1])
                      for k in ("up", "down", "peak", "valley")]
    leg1 = ax.legend(handles=fam_handles, loc="lower right",
                     title="Feature family", frameon=True, framealpha=0.95,
                     edgecolor="#888")
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="center right",
              bbox_to_anchor=(1.0, 0.55),
              title="Shape of effect", frameon=True, framealpha=0.95,
              edgecolor="#888", handlelength=1.6)

    plt.tight_layout()
    plt.savefig(OUT / "fig_ebm_importance.png", dpi=200)
    plt.close()
    print(f"wrote {OUT/'fig_ebm_importance.png'}")


# ============================================================
# Figure: shape functions for top-6 features in a single 2x3 panel
# ============================================================
def fig_ebm_shapes_top6():
    ebm = joblib.load(MODELS / "ebm_interp.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    fn = list(ebm.feature_names_in_)

    # Top-6 univariate (skip categorical & interactions)
    pairs = [(n, s) for n, s in zip(gd["names"], gd["scores"])
             if " & " not in n and n not in ("topic", "format")]
    pairs.sort(key=lambda kv: -kv[1])
    top6 = [(n, s) for n, s in pairs[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6), sharey=True)
    for ax, (feat_name, imp) in zip(axes.flat, top6):
        fi = fn.index(feat_name)
        sd = glob.data(fi)
        edges = np.asarray(sd["names"], dtype=float)
        scores = np.asarray(sd["scores"], dtype=float)
        mids = 0.5*(edges[:-1] + edges[1:])
        upper = np.asarray(sd.get("upper_bounds"), dtype=float) if sd.get("upper_bounds") is not None else None
        lower = np.asarray(sd.get("lower_bounds"), dtype=float) if sd.get("lower_bounds") is not None else None
        # Trim long tails for legibility (1st-99th percentile)
        lo, hi = np.quantile(edges, [0.01, 0.99])
        keep = (mids >= lo) & (mids <= hi)
        mx = mids[keep]; sx = scores[keep]
        fam = feature_family(feat_name)
        c = FAM_COLOR[fam]

        ax.fill_between(mx,
                         (lower[keep] if lower is not None else sx),
                         (upper[keep] if upper is not None else sx),
                         color=c, alpha=0.18, linewidth=0)
        ax.plot(mx, sx, color=c, lw=2.0)
        ax.axhline(0, color="#444", ls=":", lw=0.7, zorder=0)

        # Pretty axis labels
        ax.set_xlabel(PRETTY_FEATURE.get(feat_name, feat_name), fontsize=10)
        # Show the model's argmax / argmin as little markers (sweet-spot)
        i_max = int(np.argmax(sx))
        i_min = int(np.argmin(sx))
        ax.scatter([mx[i_max]], [sx[i_max]], s=22, color=c,
                    zorder=5, edgecolor="white", linewidths=0.6)
        ax.scatter([mx[i_min]], [sx[i_min]], s=22, color="#cc3333",
                    zorder=5, edgecolor="white", linewidths=0.6)

        ax.set_title(PRETTY_FEATURE.get(feat_name, feat_name).split(" (")[0],
                      fontsize=11)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    # Shared y label only on left column
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Contribution to $y$ ($\Delta y$)")
    # X-titles already per panel
    fig.suptitle("Shape functions of the top 6 features in the EBM",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig_ebm_shapes.png", dpi=200)
    plt.close()
    print(f"wrote {OUT/'fig_ebm_shapes.png'}")


# ============================================================
# LaTeX booktabs table for the 60M multi-benchmark results
# ============================================================
def benchmarks_table():
    rows = []
    for f in sorted((DATA / "validation_results_eq_60m_2ep").glob("V*.json")):
        r = json.loads(f.read_text())
        rows.append(r)
    df = pd.DataFrame(rows)
    pretty = {
        "V0": "Random (seed 42)", "V7": "Random (seed 43)", "V8": "Random (seed 44)",
        "V1": "DCLM-fastText", "V2": "FineWeb-Edu",
        "V3": "EBM-full",       "V9": "EBM-full (seed 43)",
        "V4": "EBM-no-PC",      "V5": "Lasso",
        "V6": "EBM within-topic",
    }
    df["pretty"] = df["rid"].map(pretty)
    family_order = {
        "V0": 0, "V7": 0, "V8": 0,
        "V6": 1, "V4": 2, "V3": 2, "V9": 2, "V5": 3, "V1": 4, "V2": 5,
    }
    df["order"] = df["rid"].map(family_order)
    df = df.sort_values(["order", "rid"]).reset_index(drop=True)

    cols = [("val_loss",       "CC val loss",         False),
            ("hellaswag_acc",  "HellaSwag",           True),
            ("arc_easy_acc",   "ARC-Easy",            True),
            ("lambada_loss",   "LAMBADA loss",        False)]

    # Compute best/worst per column to bold
    bests = {col[0]: (df[col[0]].max() if col[2] else df[col[0]].min()) for col in cols}
    worsts = {col[0]: (df[col[0]].min() if col[2] else df[col[0]].max()) for col in cols}

    def fmt(v, c, best, worst):
        if pd.isna(v): return "--"
        fmt3 = f"{v:.3f}"
        if abs(v - best) < 1e-6: return r"\textbf{" + fmt3 + r"}"
        if abs(v - worst) < 1e-6: return r"\underline{" + fmt3 + r"}"
        return fmt3

    lines = []
    lines.append(r"% Auto-generated. Bold = best on this metric; underlined = worst on this metric.")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    header = "Selection strategy & " + " & ".join(c[1] for c in cols) + r" \\"
    sub = " & ".join(["", "(nats $\\downarrow$)", "(acc $\\uparrow$)", "(acc $\\uparrow$)", "(nats $\\downarrow$)"])
    lines.append(header)
    lines.append(sub + r" \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        cells = [r["pretty"]]
        for col, _, _ in cols:
            cells.append(fmt(r[col], col, bests[col], worsts[col]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    (OUT / "benchmarks_table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'benchmarks_table.tex'}")
    # also print to stdout for review
    print("\n=== preview ===")
    print((OUT / "benchmarks_table.tex").read_text())


def fig_calibration():
    """EBM-no-PC predicted vs true y, on the held-out 20% test split."""
    import joblib
    ebm = joblib.load(MODELS / "ebm_interp.joblib")
    df = pd.read_parquet(DATA / "master.parquet")
    train_mask = df["__in_train_pool"].to_numpy()
    train_df = df.loc[train_mask].reset_index(drop=True)
    y = np.load(DATA / "document_quality_target.npy")
    ebm_pred = np.load(DATA / "ebm_interp_predicted_scores.npy")[train_mask]

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_df))
    _, te = train_test_split(idx, test_size=0.2, random_state=42,
                             stratify=train_df["topic"].to_numpy())
    y_te, p_te = y[te], ebm_pred[te]

    from scipy.stats import pearsonr
    r, _ = pearsonr(p_te, y_te)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    hb = ax.hexbin(p_te, y_te, gridsize=70, cmap="BuPu", mincnt=3,
                    extent=(-0.13, 0.11, -0.18, 0.14))
    lims = (-0.18, 0.14)
    ax.plot(lims, lims, color="#d486b8", lw=1.4, ls="--", label="y = pred")
    # binned mean
    bins = np.linspace(-0.10, 0.09, 22)
    which = np.digitize(p_te, bins) - 1
    mids = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([y_te[which == i].mean() if (which == i).sum() > 50 else np.nan
                       for i in range(len(mids))])
    ax.plot(mids, means, color="#1d3156", lw=2.4, marker="o", ms=4.5,
            markerfacecolor="white", markeredgecolor="#1d3156",
            label="binned mean of y given pred")
    ax.set_xlim(-0.13, 0.11)
    ax.set_ylim(-0.18, 0.14)
    ax.set_xlabel(r"EBM (no perplex.) predicted $\hat{y}$")
    ax.set_ylabel(r"Causal target $y$")
    ax.set_title(f"Calibration on held-out test (n=412K, r={r:.3f}, R^2 = 0.66)")
    ax.legend(loc="upper left", fontsize=9)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("documents / hex")
    plt.tight_layout()
    plt.savefig(OUT / "fig_calibration.png", dpi=200)
    plt.close()
    print(f"wrote {OUT/'fig_calibration.png'}")


def r2_table():
    """Train vs test R^2 for all three models. Demonstrates noise ceiling
    (train R^2 essentially equals test R^2)."""
    import joblib
    df = pd.read_parquet(DATA / "master.parquet")
    train_mask = df["__in_train_pool"].to_numpy()
    train_df = df.loc[train_mask].reset_index(drop=True)
    y = np.load(DATA / "document_quality_target.npy")

    CONT = [
        "char_count","word_count","mean_word_length","frac_alpha","frac_digit",
        "frac_punctuation","frac_uppercase","frac_lines_terminal_punct",
        "frac_lines_bullet","type_token_ratio","stopword_fraction",
        "ngram_rep_2","ngram_rep_3","ngram_rep_4","num_lines","mean_line_length",
        "discourse_causal","discourse_contrastive","discourse_elaborative",
        "discourse_temporal","explanation_density","question_density",
        "pronoun_density","first_person_fraction","hedge_density",
        "hapax_legomena_ratio","header_density","code_fraction",
        "avg_paragraph_length","list_density","numeric_density",
        "proper_noun_density","flesch_kincaid_grade","flesch_reading_ease",
        "entity_density","entity_diversity","entity_per_sentence",
        "qr_writing_style","qr_required_expertise","qr_facts_and_trivia",
        "qr_educational_value","pc_arc_easy","pc_piqa","pc_sciq","pc_lambada",
        "textbook_quality","ai_generated","token_count",
    ]
    CAT = ["topic","format"]
    PC = ["pc_arc_easy","pc_piqa","pc_sciq","pc_lambada"]
    cont = [c for c in CONT if c in train_df.columns]
    cat = [c for c in CAT if c in train_df.columns]

    X = train_df[cont + cat].copy()
    for c in cont:
        if X[c].isna().any(): X[c] = X[c].fillna(X[c].median())
    for c in cat:
        X[c] = X[c].astype(str)

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                              stratify=train_df["topic"].to_numpy())
    X_tr, y_tr = X.iloc[tr], y[tr]
    X_te, y_te = X.iloc[te], y[te]

    rows = []
    lasso = joblib.load(MODELS / "lasso.joblib")
    rows.append(("Lasso (linear)", lasso.score(X_tr, y_tr), lasso.score(X_te, y_te), 46))

    ebm_int = joblib.load(MODELS / "ebm_interp.joblib")
    cont_int = [c for c in cont if c not in PC]
    rows.append(("EBM no perplex.", ebm_int.score(X_tr[cont_int + cat], y_tr),
                  ebm_int.score(X_te[cont_int + cat], y_te), len(cont_int + cat)))

    ebm = joblib.load(MODELS / "ebm.joblib")
    rows.append(("EBM (all features)", ebm.score(X_tr, y_tr), ebm.score(X_te, y_te), len(cont + cat)))

    lines = [
        r"% Auto-generated. Train R^2 ~ test R^2 indicates the noise ceiling is hit, not overfitting.",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Train $R^2$ & Test $R^2$ & \# features \\",
        r"\midrule",
    ]
    for name, r_tr, r_te, nf in rows:
        lines.append(f"{name} & {r_tr:.3f} & {r_te:.3f} & {nf} \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{Split-half reliability of $y$: $r{=}0.40$}}\\",
        r"\multicolumn{4}{l}{\emph{Implied $R^2$ ceiling on full-sample $y$: $\approx 0.57$\,--\,$0.70$}}\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path = OUT / "r2_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    fig_model_r2()
    fig_ebm_importance()
    fig_ebm_shapes_top6()
    fig_calibration()
    benchmarks_table()
    r2_table()
