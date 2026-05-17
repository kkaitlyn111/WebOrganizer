"""Generate diagnostic run configs (800 total).

Run types per the plan:
  Type 1: 200 single-feature threshold
  Type 2: 300 multi-feature intersection (k=2..4)
  Type 3: 200 linear combination
  Type 4: 50  targeted single-feature (deterministic)
  Type 5: 50  topic-conditional random selections

Configs depend on which features actually exist in master.parquet. For Type 3
we need feature means/stds, so we compute them on the train pool.

Usage:
    python 04_generate_configs.py --master master.parquet \
        --types 1,2,3,4,5 --n-pilot 100 --seed 42
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import MASTER_PARQUET, DIAGNOSTIC_CONFIGS_JSON  # noqa

# All continuous features available in master.parquet (~50)
HEURISTICS_M2 = [
    "char_count", "word_count", "mean_word_length",
    "frac_alpha", "frac_digit", "frac_punctuation", "frac_uppercase",
    "frac_lines_terminal_punct", "frac_lines_bullet",
    "type_token_ratio", "stopword_fraction",
    "ngram_rep_2", "ngram_rep_3", "ngram_rep_4",
    "num_lines", "mean_line_length",
]
HEURISTICS_SYNTH = [
    "discourse_causal", "discourse_contrastive", "discourse_elaborative", "discourse_temporal",
    "explanation_density", "question_density",
    "pronoun_density", "first_person_fraction", "hedge_density",
    "hapax_legomena_ratio", "header_density", "code_fraction",
    "avg_paragraph_length", "list_density",
    "numeric_density", "proper_noun_density",
    "flesch_kincaid_grade", "flesch_reading_ease",
]
ENTITY = ["entity_density", "entity_diversity", "entity_per_sentence"]
QURATER = ["qr_writing_style", "qr_required_expertise",
           "qr_facts_and_trivia", "qr_educational_value"]
PC_SCORES = ["pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada"]
EXTRA_SCORES = ["textbook_quality", "ai_generated"]
TOKEN_FEATS = ["token_count"]

ALL_CONTINUOUS = (HEURISTICS_M2 + HEURISTICS_SYNTH + ENTITY +
                   QURATER + PC_SCORES + EXTRA_SCORES + TOKEN_FEATS)


def _filter_existing(cols, available):
    return [c for c in cols if c in available]


def gen_type1(rng, n, features, df, run_id_start):
    """Single-feature threshold selections."""
    out = []
    rid = run_id_start
    attempts = 0
    min_n = max(1000, int(0.025 * len(df)))
    while len(out) < n and attempts < n * 5:
        attempts += 1
        f = features[rng.integers(0, len(features))]
        p = float(rng.uniform(10, 90))
        direction = "above" if rng.random() < 0.7 else "below"
        col = df[f].to_numpy()
        thresh = np.nanpercentile(col, p)
        if direction == "above":
            n_sel = int((col >= thresh).sum())
        else:
            n_sel = int((col <= thresh).sum())
        if n_sel < min_n:
            continue
        out.append({
            "run_id": rid, "type": "single_threshold",
            "feature": f, "percentile": p, "direction": direction,
            "expected_n_sel": n_sel,
        })
        rid += 1
    return out


def gen_type2(rng, n, features, df, run_id_start):
    out = []
    rid = run_id_start
    attempts = 0
    min_n = max(500, int(0.015 * len(df)))
    while len(out) < n and attempts < n * 10:
        attempts += 1
        k = int(rng.integers(2, 5))  # 2,3,4
        picks = rng.choice(len(features), size=k, replace=False)
        selectors = []
        for i in picks:
            p = float(rng.uniform(20, 80))
            direction = "above" if rng.random() < 0.7 else "below"
            selectors.append({"feature": features[i], "percentile": p, "direction": direction})
        # Try with progressive relaxation
        for relax in (0, 10, 20, 30):
            mask = np.ones(len(df), dtype=bool)
            for sel in selectors:
                col = df[sel["feature"]].to_numpy()
                # Relax: pull thresholds toward 50%
                p = sel["percentile"]
                if sel["direction"] == "above":
                    p_eff = max(20, p - relax)
                else:
                    p_eff = min(80, p + relax)
                thresh = np.nanpercentile(col, p_eff)
                if sel["direction"] == "above":
                    mask &= (col >= thresh)
                else:
                    mask &= (col <= thresh)
            if mask.sum() >= min_n:
                # Apply relaxation back to stored percentiles
                final_selectors = []
                for sel in selectors:
                    p = sel["percentile"]
                    if sel["direction"] == "above":
                        p_eff = max(20, p - relax)
                    else:
                        p_eff = min(80, p + relax)
                    final_selectors.append({**sel, "percentile": p_eff})
                out.append({
                    "run_id": rid, "type": "multi_intersection",
                    "selectors": final_selectors,
                    "expected_n_sel": int(mask.sum()),
                })
                rid += 1
                break
    return out


def gen_type3(rng, n, features, df, run_id_start):
    """Random linear combination, top-p%."""
    out = []
    rid = run_id_start
    # Precompute means/stds
    means = df[features].mean(skipna=True).to_numpy()
    stds = df[features].std(skipna=True).replace(0, 1).to_numpy()
    for _ in range(n):
        w = rng.normal(0, 1, len(features))
        top_pct = float(rng.uniform(10, 35))
        out.append({
            "run_id": rid, "type": "linear_top",
            "features": features, "weights": w.tolist(),
            "means": means.tolist(), "stds": stds.tolist(),
            "top_pct": top_pct,
        })
        rid += 1
    return out


def gen_type4(run_id_start, available):
    """Deterministic targeted runs (~50 total)."""
    out = []
    rid = run_id_start

    def add(cfg):
        nonlocal rid
        cfg["run_id"] = rid; rid += 1
        out.append(cfg)

    # Top 15% by each of 14 features (above)
    top14 = [
        "discourse_causal", "entity_density", "flesch_kincaid_grade",
        "textbook_quality", "explanation_density", "hapax_legomena_ratio",
        "qr_educational_value", "qr_writing_style", "qr_required_expertise",
        "qr_facts_and_trivia",
        # ai_generated: select LOWEST P(Fake) => direction=below at 15th percentile
        "numeric_density", "proper_noun_density", "question_density",
    ]
    for f in top14:
        if f in available:
            add({"type": "targeted_top", "feature": f, "top_pct": 15, "direction": "above"})
    if "ai_generated" in available:
        add({"type": "targeted_top", "feature": "ai_generated", "top_pct": 15, "direction": "below"})

    # Bottom 15% by 5 key features
    bottom5 = [
        ("discourse_causal", "below"),
        ("entity_density", "below"),
        ("flesch_kincaid_grade", "below"),
        ("textbook_quality", "below"),
        ("qr_educational_value", "below"),
    ]
    for f, d in bottom5:
        if f in available:
            add({"type": "targeted_top", "feature": f, "top_pct": 15, "direction": d})

    # Black-box baselines
    add({"type": "targeted_top", "feature": "dclm_score", "top_pct": 15, "direction": "above"})
    add({"type": "fwedu_geq", "threshold": 3})
    for pc in PC_SCORES:
        if pc in available:
            add({"type": "targeted_top", "feature": pc, "top_pct": 15, "direction": "above"})

    # Random baselines
    for seed in (100, 200, 300):
        add({"type": "random", "seed": seed, "frac": 0.15})

    # Combinations
    if all(f in available for f in ("entity_density", "discourse_causal")):
        add({"type": "multi_intersection",
             "selectors": [{"feature": "entity_density", "percentile": 70, "direction": "above"},
                            {"feature": "discourse_causal", "percentile": 70, "direction": "above"}]})
    if all(f in available for f in ("qr_required_expertise", "entity_density")):
        add({"type": "multi_intersection",
             "selectors": [{"feature": "qr_required_expertise", "percentile": 70, "direction": "above"},
                            {"feature": "entity_density", "percentile": 70, "direction": "above"}]})
    # combined z-score (discourse_causal + entity_density + flesch_kincaid_grade)
    combo_a = [f for f in ("discourse_causal", "entity_density", "flesch_kincaid_grade") if f in available]
    if len(combo_a) >= 2:
        add({"type": "linear_top_simple", "features": combo_a, "top_pct": 15})
    combo_b = [f for f in QURATER if f in available]
    if len(combo_b) >= 2:
        add({"type": "linear_top_simple", "features": combo_b, "top_pct": 15})
    if "dclm_score" in available and "entity_density" in available:
        add({"type": "dclm_top_then_restrict", "dclm_top_pct": 15,
             "restrict_feature": "entity_density", "restrict_top_pct": 50})
    if "dclm_score" in available and "discourse_causal" in available:
        add({"type": "dclm_top_then_restrict", "dclm_top_pct": 15,
             "restrict_feature": "discourse_causal", "restrict_top_pct": 50})
    combo_c = [f for f in ("entity_density", "explanation_density", "proper_noun_density") if f in available]
    if len(combo_c) >= 2:
        add({"type": "linear_top_simple", "features": combo_c, "top_pct": 15})
    if all(f in available for f in ("textbook_quality", "qr_educational_value")):
        add({"type": "multi_intersection",
             "selectors": [{"feature": "textbook_quality", "percentile": 70, "direction": "above"},
                            {"feature": "qr_educational_value", "percentile": 70, "direction": "above"}]})

    # Within-topic selections
    for f in ("entity_density", "discourse_causal", "flesch_kincaid_grade", "textbook_quality"):
        if f in available:
            add({"type": "within_topic_top", "feature": f, "top_pct": 15})
    if "discourse_causal" in available and "entity_density" in available:
        add({"type": "within_topic_top", "feature": "__combo_de", "top_pct": 15})

    return out


def gen_type5(rng, n, features, df, run_id_start):
    out = []
    rid = run_id_start
    topics = df["topic"].to_numpy()
    unique_topics, topic_counts = np.unique(topics, return_counts=True)
    # min count proportional to df size; floor 100 so tests pass
    min_count = max(100, int(0.005 * len(df)))
    eligible = unique_topics[topic_counts >= min_count]
    if len(eligible) == 0:
        return []
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        t = int(eligible[rng.integers(0, len(eligible))])
        f = features[rng.integers(0, len(features))]
        col = df[f].to_numpy()
        tmask = topics == t
        if tmask.sum() < min_count:
            continue
        thresh = np.nanpercentile(col[tmask], 70)
        n_sel = int(((col >= thresh) & tmask).sum())
        if n_sel < max(50, int(0.001 * len(df))):
            continue
        out.append({
            "run_id": rid, "type": "topic_conditional",
            "topic": t, "feature": f, "top_pct": 30,
            "expected_n_sel": n_sel,
        })
        rid += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=MASTER_PARQUET)
    ap.add_argument("--output", type=Path, default=DIAGNOSTIC_CONFIGS_JSON)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-type1", type=int, default=200)
    ap.add_argument("--n-type2", type=int, default=300)
    ap.add_argument("--n-type3", type=int, default=200)
    ap.add_argument("--n-type5", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_parquet(args.master)
    df = df[df["__in_train_pool"]].reset_index(drop=True)
    print(f"train-pool docs: {len(df):,}")

    available = set(df.columns)
    feats = _filter_existing(ALL_CONTINUOUS, available)
    print(f"available continuous features ({len(feats)}): {feats}")

    rng = np.random.default_rng(args.seed)
    rid = 0
    out = []
    t1 = gen_type1(rng, args.n_type1, feats, df, rid); rid += len(t1); out += t1
    print(f"type 1: {len(t1)} runs")
    t2 = gen_type2(rng, args.n_type2, feats, df, rid); rid += len(t2); out += t2
    print(f"type 2: {len(t2)} runs")
    t3 = gen_type3(rng, args.n_type3, feats, df, rid); rid += len(t3); out += t3
    print(f"type 3: {len(t3)} runs")
    t4 = gen_type4(rid, available); rid += len(t4); out += t4
    print(f"type 4: {len(t4)} runs")
    t5 = gen_type5(rng, args.n_type5, feats, df, rid); rid += len(t5); out += t5
    print(f"type 5: {len(t5)} runs")
    print(f"TOTAL: {len(out)} runs")

    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
