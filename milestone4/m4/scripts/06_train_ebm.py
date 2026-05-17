"""Phase 3: train Lasso, EBM, and EBM-without-perplexity-correlations.

Inputs:
  - master.parquet (train-pool rows aligned with target)
  - data/document_quality_target.npy

Outputs:
  - models/lasso.joblib, models/ebm.joblib, models/ebm_interp.joblib
  - data/phase3_summary.json (R^2 + top features)
  - data/lasso_predicted_scores.npy, ebm_predicted_scores.npy, ebm_interp_predicted_scores.npy
    (predictions on ALL rows of master.parquet; train+val pool)
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import MASTER_PARQUET, DATA_DIR, MODELS_DIR  # noqa

warnings.filterwarnings("ignore")

CONT_FEATURES = [
    # heuristics m2
    "char_count", "word_count", "mean_word_length",
    "frac_alpha", "frac_digit", "frac_punctuation", "frac_uppercase",
    "frac_lines_terminal_punct", "frac_lines_bullet",
    "type_token_ratio", "stopword_fraction",
    "ngram_rep_2", "ngram_rep_3", "ngram_rep_4",
    "num_lines", "mean_line_length",
    # synth heuristics
    "discourse_causal", "discourse_contrastive", "discourse_elaborative", "discourse_temporal",
    "explanation_density", "question_density",
    "pronoun_density", "first_person_fraction", "hedge_density",
    "hapax_legomena_ratio", "header_density", "code_fraction",
    "avg_paragraph_length", "list_density",
    "numeric_density", "proper_noun_density",
    "flesch_kincaid_grade", "flesch_reading_ease",
    # entities
    "entity_density", "entity_diversity", "entity_per_sentence",
    # qurater
    "qr_writing_style", "qr_required_expertise",
    "qr_facts_and_trivia", "qr_educational_value",
    # perplexity-corr
    "pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada",
    # extras
    "textbook_quality", "ai_generated",
    "token_count",
]
CAT_FEATURES = ["topic", "format"]
PC_DROP = ["pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=DATA_DIR / "document_quality_target.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-docs", type=int, default=None,
                    help="Optional cap for EBM training (speeds up testing)")
    args = ap.parse_args()

    df = pd.read_parquet(MASTER_PARQUET)
    train_mask = df["__in_train_pool"].to_numpy()
    train_df = df[train_mask].reset_index(drop=True)
    y = np.load(args.target)
    assert len(y) == len(train_df), f"target {len(y)} != train_df {len(train_df)}"

    # Features available
    cont = [c for c in CONT_FEATURES if c in train_df.columns]
    cat = [c for c in CAT_FEATURES if c in train_df.columns]
    print(f"cont features ({len(cont)}): {cont}")
    print(f"cat features ({len(cat)}): {cat}")

    X = train_df[cont + cat].copy()
    # impute simple median for any NaN continuous
    for c in cont:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    # stratified split by topic
    from sklearn.model_selection import train_test_split
    strat = train_df["topic"].to_numpy() if "topic" in train_df.columns else None
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed,
                                       stratify=strat)
    if args.max_train_docs:
        rng = np.random.default_rng(args.seed)
        tr_idx = rng.choice(tr_idx, size=min(args.max_train_docs, len(tr_idx)), replace=False)

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]
    print(f"train: {len(X_tr):,}, test: {len(X_te):,}")

    summary = {"cont_features": cont, "cat_features": cat,
                "n_train": int(len(X_tr)), "n_test": int(len(X_te))}

    # ========== Lasso ==========
    print("\n=== Lasso ===")
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    pre = ColumnTransformer([
        ("num", StandardScaler(), cont),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat),
    ])
    lasso = Pipeline([
        ("pre", pre),
        ("model", LassoCV(cv=5, alphas=np.logspace(-5, 0, 30), max_iter=20000, n_jobs=-1)),
    ])
    lasso.fit(X_tr, y_tr)
    r2_lasso = float(lasso.score(X_te, y_te))
    print(f"  Lasso R^2: {r2_lasso:.4f}")
    coefs = lasso.named_steps["model"].coef_
    fnames = lasso.named_steps["pre"].get_feature_names_out()
    imp = pd.DataFrame({"feature": fnames, "coef": coefs})
    imp["abscoef"] = imp["coef"].abs()
    imp = imp.sort_values("abscoef", ascending=False)
    top_lasso = imp.head(20).to_dict("records")
    summary["lasso_r2"] = r2_lasso
    summary["lasso_top20"] = top_lasso
    summary["lasso_n_nonzero"] = int((coefs != 0).sum())
    joblib.dump(lasso, MODELS_DIR / "lasso.joblib")

    # ========== EBM (full) ==========
    print("\n=== EBM (all features) ===")
    from interpret.glassbox import ExplainableBoostingRegressor
    X_tr_ebm = X_tr.copy()
    X_te_ebm = X_te.copy()
    for c in cat:
        X_tr_ebm[c] = X_tr_ebm[c].astype(str)
        X_te_ebm[c] = X_te_ebm[c].astype(str)

    ebm = ExplainableBoostingRegressor(
        max_bins=256, interactions=20, learning_rate=0.01,
        max_rounds=8000, min_samples_leaf=50, outer_bags=4, inner_bags=2,
        random_state=args.seed, n_jobs=-1,
    )
    ebm.fit(X_tr_ebm, y_tr)
    r2_ebm = float(ebm.score(X_te_ebm, y_te))
    print(f"  EBM R^2: {r2_ebm:.4f}")
    summary["ebm_r2"] = r2_ebm
    glob = ebm.explain_global()
    gdata = glob.data()
    ebm_imp = sorted(zip(gdata["names"], gdata["scores"]), key=lambda kv: -kv[1])
    summary["ebm_top20"] = [{"feature": n, "importance": float(s)} for n, s in ebm_imp[:20]]
    joblib.dump(ebm, MODELS_DIR / "ebm.joblib")

    # ========== EBM (no perplexity-corr) ==========
    print("\n=== EBM (no PC) ===")
    cont_interp = [c for c in cont if c not in PC_DROP]
    X_tr_int = X_tr_ebm[cont_interp + cat]
    X_te_int = X_te_ebm[cont_interp + cat]
    ebm_int = ExplainableBoostingRegressor(
        max_bins=256, interactions=20, learning_rate=0.01,
        max_rounds=8000, min_samples_leaf=50, outer_bags=4, inner_bags=2,
        random_state=args.seed, n_jobs=-1,
    )
    ebm_int.fit(X_tr_int, y_tr)
    r2_ebm_int = float(ebm_int.score(X_te_int, y_te))
    print(f"  EBM (no PC) R^2: {r2_ebm_int:.4f}")
    summary["ebm_interp_r2"] = r2_ebm_int
    glob_i = ebm_int.explain_global()
    gdata_i = glob_i.data()
    int_imp = sorted(zip(gdata_i["names"], gdata_i["scores"]), key=lambda kv: -kv[1])
    summary["ebm_interp_top20"] = [{"feature": n, "importance": float(s)} for n, s in int_imp[:20]]
    joblib.dump(ebm_int, MODELS_DIR / "ebm_interp.joblib")

    (DATA_DIR / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {DATA_DIR / 'phase3_summary.json'}")
    print(f"Lasso R^2={r2_lasso:.4f}  EBM={r2_ebm:.4f}  EBM-noPC={r2_ebm_int:.4f}")

    # ====== Predictions on ALL rows of master ======
    X_full = df[cont + cat].copy()
    for c in cont:
        if X_full[c].isna().any():
            X_full[c] = X_full[c].fillna(X_full[c].median())
    np.save(DATA_DIR / "lasso_predicted_scores.npy",
            lasso.predict(X_full).astype(np.float32))
    X_full_ebm = X_full.copy()
    for c in cat:
        X_full_ebm[c] = X_full_ebm[c].astype(str)
    np.save(DATA_DIR / "ebm_predicted_scores.npy",
            ebm.predict(X_full_ebm).astype(np.float32))
    X_full_int = X_full_ebm[cont_interp + cat]
    np.save(DATA_DIR / "ebm_interp_predicted_scores.npy",
            ebm_int.predict(X_full_int).astype(np.float32))
    print("Saved predicted scores for all master rows.")


if __name__ == "__main__":
    main()
