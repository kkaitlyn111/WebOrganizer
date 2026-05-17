"""Basic feature EDA after master.parquet is built.

Produces:
  data/figures_eda/feature_histograms.pdf           — 1 panel per continuous feature
  data/figures_eda/feature_correlation.pdf          — heatmap of all features
  data/figures_eda/topic_format_counts.pdf          — bar charts
  data/figures_eda/url_zipf.pdf                     — log-log domain frequency
  data/figures_eda/quality_score_pairs.pdf          — pairplot of 10 quality scores
  data/figures_eda/quality_by_topic.pdf             — boxplots: 4 main quality scores by 24 topics
  data/feature_stats.csv                            — per-feature summary stats
  data/feature_stats_by_topic.csv                   — per-feature means split by topic
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import MASTER_PARQUET, DATA_DIR  # noqa

OUT_DIR = DATA_DIR / "figures_eda"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CONT = [
    # heuristics
    "char_count", "word_count", "mean_word_length", "frac_alpha", "frac_digit",
    "frac_punctuation", "frac_uppercase", "frac_lines_terminal_punct",
    "frac_lines_bullet", "type_token_ratio", "stopword_fraction",
    "ngram_rep_2", "ngram_rep_3", "ngram_rep_4", "num_lines", "mean_line_length",
    # synth heuristics
    "discourse_causal", "discourse_contrastive", "discourse_elaborative",
    "discourse_temporal", "explanation_density", "question_density",
    "pronoun_density", "first_person_fraction", "hedge_density",
    "hapax_legomena_ratio", "header_density", "code_fraction",
    "avg_paragraph_length", "list_density", "numeric_density",
    "proper_noun_density", "flesch_kincaid_grade", "flesch_reading_ease",
    # entities
    "entity_density", "entity_diversity", "entity_per_sentence",
    # quality scores
    "dclm_score", "fwedu_rounded",
    "qr_writing_style", "qr_required_expertise",
    "qr_facts_and_trivia", "qr_educational_value",
    "pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada",
    "textbook_quality", "ai_generated",
    "token_count",
]
QUALITY = ["dclm_score", "fwedu_rounded", "textbook_quality", "ai_generated",
            "qr_educational_value", "qr_writing_style",
            "qr_required_expertise", "qr_facts_and_trivia",
            "pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada"]


def fig_histograms(df, feats):
    n = len(feats)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    for ax, f in zip(axes.flat, feats):
        vals = df[f].dropna().to_numpy()
        # heavy-tail-aware: log scale if range >> 10x
        if len(vals) and (vals > 0).any():
            v = vals[vals > 0]
            ax.hist(np.log10(v + 1e-9), bins=60, color="steelblue") if (v.max() / max(v.min(), 1e-9)) > 100 else ax.hist(vals, bins=60, color="steelblue")
        else:
            ax.hist(vals, bins=60, color="steelblue")
        ax.set_title(f, fontsize=8)
        ax.set_yticks([])
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.suptitle("Feature distributions (log10 x where heavy-tailed)", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_histograms.pdf"); plt.close()
    print("wrote feature_histograms.pdf")


def fig_correlation(df, feats):
    sub = df[feats].dropna(how="all", axis=1)
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                 square=True, xticklabels=True, yticklabels=True,
                 cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_correlation.pdf"); plt.close()
    print("wrote feature_correlation.pdf")


def fig_topic_format(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ("topic", "format")):
        if col not in df.columns:
            continue
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(int), counts.values, color="darkgreen")
        ax.set_title(f"{col} distribution (24 categories)")
        ax.set_xlabel(col); ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topic_format_counts.pdf"); plt.close()
    print("wrote topic_format_counts.pdf")


def fig_url_zipf(df):
    if "url_registered_domain" not in df.columns:
        print("skip url_zipf (no domain col)")
        return
    dom_counts = df["url_registered_domain"].value_counts()
    n_dom = len(dom_counts)
    n100 = int((dom_counts >= 100).sum())
    n1000 = int((dom_counts >= 1000).sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(np.arange(1, n_dom + 1), dom_counts.values, color="firebrick")
    ax.set_xlabel("domain rank"); ax.set_ylabel("count")
    ax.set_title(f"URL domain Zipf (n_unique={n_dom:,}, "
                  f">=100 docs: {n100:,}, >=1000: {n1000:,})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "url_zipf.pdf"); plt.close()
    print("wrote url_zipf.pdf")


def fig_quality_pairs(df):
    cols = [c for c in QUALITY if c in df.columns]
    if len(cols) < 2:
        return
    # subsample to 5000 for plot speed
    sub = df[cols].dropna().sample(min(5000, len(df)), random_state=0)
    g = sns.PairGrid(sub, diag_sharey=False, height=1.2, corner=True)
    g.map_lower(sns.scatterplot, s=3, alpha=0.2, color="indigo")
    g.map_diag(sns.histplot, bins=30, color="indigo")
    plt.suptitle("Pairwise scatter of quality scores", y=1.01)
    g.savefig(OUT_DIR / "quality_score_pairs.pdf"); plt.close()
    print("wrote quality_score_pairs.pdf")


def fig_quality_by_topic(df):
    cols = [c for c in ("dclm_score", "fwedu_rounded", "textbook_quality",
                          "qr_educational_value") if c in df.columns]
    if not cols or "topic" not in df.columns:
        return
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3 * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        sub = df[[c, "topic"]].dropna()
        sns.boxplot(data=sub, x="topic", y=c, ax=ax,
                     color="lightsteelblue", showfliers=False)
        ax.set_title(c)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quality_by_topic.pdf"); plt.close()
    print("wrote quality_by_topic.pdf")


def feature_stats(df, feats):
    rows = []
    for f in feats:
        if f not in df.columns:
            continue
        s = df[f].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "feature": f, "n": len(s),
            "n_nan": df[f].isna().sum(),
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "p1": float(s.quantile(0.01)),
            "p25": float(s.quantile(0.25)), "median": float(s.median()),
            "p75": float(s.quantile(0.75)), "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "skew": float(s.skew()), "kurt": float(s.kurt()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "feature_stats.csv", index=False)
    print(f"wrote feature_stats.csv ({len(out)} features)")
    return out


def feature_stats_by_topic(df, feats):
    if "topic" not in df.columns:
        return
    qcols = [c for c in QUALITY if c in df.columns]
    g = df.groupby("topic")[qcols].agg(["mean", "std", "count"])
    g.to_csv(DATA_DIR / "feature_stats_by_topic.csv")
    print("wrote feature_stats_by_topic.csv")


def main():
    print(f"loading {MASTER_PARQUET} ...")
    df = pd.read_parquet(MASTER_PARQUET)
    print(f"  shape: {df.shape}")
    print(f"  in_train_pool: {df['__in_train_pool'].sum():,}")

    feats = [f for f in CONT if f in df.columns]
    print(f"  {len(feats)}/{len(CONT)} continuous features available")

    fig_histograms(df, feats)
    fig_correlation(df, feats)
    fig_topic_format(df)
    fig_url_zipf(df)
    fig_quality_pairs(df)
    fig_quality_by_topic(df)
    feature_stats(df, feats)
    feature_stats_by_topic(df, feats)
    print(f"\nAll EDA outputs in: {OUT_DIR}")
    print(f"Stats CSVs in:      {DATA_DIR}")


if __name__ == "__main__":
    main()
