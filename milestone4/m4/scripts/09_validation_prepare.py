"""Phase 4 prep: build fresh_master.parquet, score with EBM, consolidate tokens.

Inputs:
  data/fresh_shards.json
  per-shard fresh features (heuristics + synth + entities + textbook + ai-gen + pc)
  trained EBM models in data/models/

Outputs:
  data/fresh_master.parquet
  data/tokens/fresh_token_ids.bin + fresh_doc_offsets.npy
  data/fresh_ebm_scores.npy, fresh_ebm_interp_scores.npy, fresh_lasso_scores.npy
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (DATA_DIR, MODELS_DIR, TOKENS_DIR, PERSHARD_TOKENS_DIR,
                    REPO_ROOT, M2_DATA)  # noqa

import importlib.util as _ilu
def _load(name, p):
    spec = _ilu.spec_from_file_location(name, str(p))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
_m2cfg = _load("m2cfg", REPO_ROOT / "milestone2" / "config.py")
shard_stem = _m2cfg.shard_stem


def load_fresh_shard(s):
    """Build per-shard dataframe from extracted features (best-effort, NaN for missing)."""
    stem = shard_stem(s)
    feat_p = M2_DATA / "features" / f"{stem}.parquet"
    if not feat_p.exists():
        raise SystemExit(f"missing features for shard {s}")
    df = pd.read_parquet(feat_p).rename(columns={"doc_idx": "__doc_idx"})
    n = len(df)
    df["__shard_idx"] = s

    # corpus-provided annotations
    corp = REPO_ROOT / "Corpus-200B"
    def opt(path):
        return np.load(path) if path.exists() else np.full(n, np.nan)
    df["dclm_score"] = opt(corp / "scores_dclm-fasttext" / f"{stem}.npy")
    df["fwedu_rounded"] = opt(corp / "scores_fineweb-edu__rounded" / f"{stem}__rounded.npy")
    df["topic"] = opt(corp / "domains_topics" / f"{stem}__choice.npy")
    df["format"] = opt(corp / "domains_formats" / f"{stem}__choice.npy")
    df["cluster24"] = opt(corp / "domains_clusters-k24" / f"{stem}.npy")
    df["token_count"] = opt(corp / "tokens" / f"{stem}.npy")

    # synth heuristics
    synth_p = M2_DATA / "features_synth" / f"{stem}.parquet"
    if synth_p.exists():
        s_df = pd.read_parquet(synth_p).rename(columns={"doc_idx": "__doc_idx"})
        df = df.merge(s_df, on="__doc_idx", how="left")

    # entities
    ent_p = M2_DATA / "entities" / f"{stem}.npz"
    if ent_p.exists():
        z = np.load(ent_p)
        for c in ("entity_density", "entity_diversity", "entity_per_sentence"):
            df[c] = z[c] if z[c].shape[0] == n else np.nan

    # qurater 4
    qr_p = M2_DATA / "scores_qurater" / f"{stem}.npy"
    qr_axes = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]
    if qr_p.exists():
        qr = np.load(qr_p)
        if qr.shape[0] == n:
            for i, a in enumerate(qr_axes):
                df[f"qr_{a}"] = qr[:, i]

    # pc
    for m in ("arc_easy", "piqa", "sciq", "lambada"):
        p = M2_DATA / "scores_pc" / m / f"{stem}.npy"
        if p.exists():
            arr = np.load(p)
            df[f"pc_{m}"] = arr if len(arr) == n else np.nan

    # textbook + ai-gen
    tb = M2_DATA / "scores_textbook_quality" / f"{stem}.npy"
    if tb.exists():
        arr = np.load(tb)
        df["textbook_quality"] = arr if len(arr) == n else np.nan
    ai = M2_DATA / "scores_ai_generated" / f"{stem}.npy"
    if ai.exists():
        arr = np.load(ai)
        df["ai_generated"] = arr if len(arr) == n else np.nan

    return df


def consolidate_fresh_tokens(shards):
    """Concat per-shard tokens into one memmap."""
    total = 0
    pieces = []
    for s in shards:
        stem = shard_stem(s)
        off = np.load(PERSHARD_TOKENS_DIR / f"{stem}_doc_offsets.npy")
        toks = np.load(PERSHARD_TOKENS_DIR / f"{stem}_token_ids.npy", mmap_mode="r")
        pieces.append((s, off, toks))
        total += int(off[-1])
    print(f"  total fresh tokens: {total:,}")
    out_bin = TOKENS_DIR / "fresh_token_ids.bin"
    out_off = TOKENS_DIR / "fresh_doc_offsets.npy"
    mm = np.memmap(out_bin, dtype=np.uint32, mode="w+", shape=(total,))
    n_docs = sum(len(off) - 1 for _, off, _ in pieces)
    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    tok_cursor = 0
    doc_cursor = 0
    for s, off, toks in pieces:
        n = int(off[-1])
        mm[tok_cursor: tok_cursor + n] = toks
        nd = len(off) - 1
        offsets[doc_cursor: doc_cursor + nd + 1] = off + tok_cursor
        tok_cursor += n
        doc_cursor += nd
    offsets[-1] = total
    mm.flush()
    np.save(out_off, offsets)
    print(f"  wrote {out_bin}, {out_off}")


def main():
    fresh_shards = json.loads((DATA_DIR / "fresh_shards.json").read_text())["shards"]
    print(f"fresh shards: {fresh_shards}")

    # Build dataframe
    parts = []
    for s in tqdm(fresh_shards, desc="loading"):
        parts.append(load_fresh_shard(s))
    fresh = pd.concat(parts, ignore_index=True)
    print(f"fresh docs: {len(fresh):,}, columns: {len(fresh.columns)}")

    # Consolidate tokens
    print("consolidating tokens ...")
    consolidate_fresh_tokens(fresh_shards)

    # Score with EBM / Lasso
    lasso = joblib.load(MODELS_DIR / "lasso.joblib")
    ebm = joblib.load(MODELS_DIR / "ebm.joblib")
    ebm_int = joblib.load(MODELS_DIR / "ebm_interp.joblib")
    # Use the EXACT feature list the EBM was trained on; we read it from phase3 summary
    summary = json.loads((DATA_DIR / "phase3_summary.json").read_text())
    cont = summary["cont_features"]
    cat = summary["cat_features"]
    # Ensure all needed columns exist (fill NaN)
    for c in cont + cat:
        if c not in fresh.columns:
            fresh[c] = np.nan
    X = fresh[cont + cat].copy()
    for c in cont:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat:
        if X[c].isna().any():
            X[c] = X[c].fillna(-1)
    X_ebm = X.copy()
    for c in cat:
        X_ebm[c] = X_ebm[c].astype(str)
    X_int = X_ebm[[c for c in cont if c not in ("pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada")] + cat]

    np.save(DATA_DIR / "fresh_lasso_scores.npy", lasso.predict(X).astype(np.float32))
    np.save(DATA_DIR / "fresh_ebm_scores.npy", ebm.predict(X_ebm).astype(np.float32))
    np.save(DATA_DIR / "fresh_ebm_interp_scores.npy", ebm_int.predict(X_int).astype(np.float32))
    print("  saved fresh_*_scores.npy")

    fresh.to_parquet(DATA_DIR / "fresh_master.parquet", index=False)
    print(f"wrote fresh_master.parquet")


if __name__ == "__main__":
    main()
