"""Retrain EBM-no-PC with topic/format as proper nominal (categorical),
overwriting models/ebm_interp.joblib and ebm_interp_predicted_scores.npy.
Same hyperparams, same split (seed=42 stratified by topic) so R^2 is comparable.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import MASTER_PARQUET, DATA_DIR, MODELS_DIR  # noqa

import warnings
warnings.filterwarnings("ignore")

CONT = [
    "char_count", "word_count", "mean_word_length",
    "frac_alpha", "frac_digit", "frac_punctuation", "frac_uppercase",
    "frac_lines_terminal_punct", "frac_lines_bullet",
    "type_token_ratio", "stopword_fraction",
    "ngram_rep_2", "ngram_rep_3", "ngram_rep_4",
    "num_lines", "mean_line_length",
    "discourse_causal", "discourse_contrastive", "discourse_elaborative", "discourse_temporal",
    "explanation_density", "question_density",
    "pronoun_density", "first_person_fraction", "hedge_density",
    "hapax_legomena_ratio", "header_density", "code_fraction",
    "avg_paragraph_length", "list_density",
    "numeric_density", "proper_noun_density",
    "flesch_kincaid_grade", "flesch_reading_ease",
    "entity_density", "entity_diversity", "entity_per_sentence",
    "qr_writing_style", "qr_required_expertise",
    "qr_facts_and_trivia", "qr_educational_value",
    "textbook_quality", "ai_generated",
    "token_count",
]
CAT = ["topic", "format"]

print("Loading master + target...")
df = pd.read_parquet(MASTER_PARQUET)
train_mask = df["__in_train_pool"].to_numpy()
train_df = df[train_mask].reset_index(drop=True)
y = np.load(DATA_DIR / "document_quality_target.npy")
assert len(y) == len(train_df)

cont = [c for c in CONT if c in train_df.columns]
cat = [c for c in CAT if c in train_df.columns]
print(f"cont={len(cont)}  cat={len(cat)}")

X = train_df[cont + cat].copy()
for c in cont:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())
# treat cat as pandas categorical strings
for c in cat:
    X[c] = X[c].astype(str)

from sklearn.model_selection import train_test_split
idx = np.arange(len(X))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=train_df["topic"].to_numpy())
X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
X_te, y_te = X.iloc[te_idx], y[te_idx]

feature_types = ["continuous"] * len(cont) + ["nominal"] * len(cat)
print("feature_types tail:", feature_types[-4:])

from interpret.glassbox import ExplainableBoostingRegressor
ebm = ExplainableBoostingRegressor(
    feature_types=feature_types,
    feature_names=list(X.columns),
    max_bins=256, interactions=20, learning_rate=0.01,
    max_rounds=8000, min_samples_leaf=50, outer_bags=4, inner_bags=2,
    random_state=42, n_jobs=-1,
)
print("fitting EBM-no-PC with proper nominal topic/format...")
ebm.fit(X_tr, y_tr)
r2 = float(ebm.score(X_te, y_te))
print(f"R^2: {r2:.4f}")

joblib.dump(ebm, MODELS_DIR / "ebm_interp.joblib")
print(f"saved models/ebm_interp.joblib")

# rewrite predictions on ALL master rows
X_full = df[cont + cat].copy()
for c in cont:
    if X_full[c].isna().any():
        X_full[c] = X_full[c].fillna(X_full[c].median())
for c in cat:
    X_full[c] = X_full[c].astype(str)
preds = ebm.predict(X_full).astype(np.float32)
np.save(DATA_DIR / "ebm_interp_predicted_scores.npy", preds)
print(f"saved ebm_interp_predicted_scores.npy  (n={len(preds):,})")

# update phase3_summary.json's ebm_interp_r2 + top-20
s = json.loads((DATA_DIR / "phase3_summary.json").read_text())
s["ebm_interp_r2"] = r2
glob = ebm.explain_global()
gd = glob.data()
imps = sorted(zip(gd["names"], gd["scores"]), key=lambda kv: -kv[1])
s["ebm_interp_top20"] = [{"feature": n, "importance": float(v)} for n, v in imps[:20]]
s["ebm_interp_nominal"] = True
(DATA_DIR / "phase3_summary.json").write_text(json.dumps(s, indent=2))
print("updated phase3_summary.json")
