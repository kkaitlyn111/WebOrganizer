"""Build master.parquet for milestone 4.

Joins per-shard:
  - annotations (topic, format, cluster24, dclm_score, fwedu_rounded, token_count)
  - 16 heuristics from milestone2/data/features/<stem>.parquet  (plus url fields)
  - 18 synth heuristics from milestone2/data/features_synth/<stem>.parquet
  - 3 entity features from milestone2/data/entities/<stem>.npz
  - 4 QuRater axes from milestone2/data/scores_qurater/<stem>.npy
  - 4 perplexity-corr scores (arc_easy, piqa, sciq, lambada)
  - textbook quality from milestone2/data/scores_textbook_quality/<stem>.npy
  - AI generated from milestone2/data/scores_ai_generated/<stem>.npy

The output is ordered: train_pool shards (in order from val_shard_indices.json)
THEN val shards. This MUST match the pretokenized memmap ordering.

Each row has `__shard_idx`, `__doc_idx`, `__in_train_pool` columns.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import (M2_DATA, MASTER_PARQUET, VAL_SHARDS_JSON, REPO_ROOT)  # noqa

import importlib.util as _ilu
def _load(name, p):
    spec = _ilu.spec_from_file_location(name, str(p))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
_m2cfg = _load("m2cfg", REPO_ROOT / "milestone2" / "config.py")
shard_stem = _m2cfg.shard_stem

PC_MODELS = ["arc_easy", "piqa", "sciq", "lambada"]
QURATER_AXES = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]


def load_shard(s, ann_by_shard):
    stem = shard_stem(s)
    # heuristics (also gives doc_idx, url, etc.)
    feat = pd.read_parquet(M2_DATA / "features" / f"{stem}.parquet")
    n = len(feat)
    feat["__shard_idx"] = s
    # rename doc_idx -> __doc_idx
    feat = feat.rename(columns={"doc_idx": "__doc_idx"})

    # annotations from annotations.parquet (already loaded)
    ann = ann_by_shard.get(s)
    if ann is None:
        raise ValueError(f"no annotations for shard {s}")
    # ann has columns: shard_idx, doc_idx, topic, format, cluster24, dclm_score, fwedu_score, fwedu_rounded, token_count
    ann_slim = ann[["doc_idx", "topic", "format", "cluster24", "dclm_score",
                     "fwedu_rounded", "token_count"]].rename(columns={"doc_idx": "__doc_idx"})
    feat = feat.merge(ann_slim, on="__doc_idx", how="left")

    # synth heuristics
    synth = pd.read_parquet(M2_DATA / "features_synth" / f"{stem}.parquet")
    synth = synth.rename(columns={"doc_idx": "__doc_idx"})
    feat = feat.merge(synth, on="__doc_idx", how="left")

    # entities
    ent_path = M2_DATA / "entities" / f"{stem}.npz"
    if ent_path.exists():
        z = np.load(ent_path)
        ent_cols = ["entity_density", "entity_diversity", "entity_per_sentence"]
        ent_n = z[ent_cols[0]].shape[0]
        if ent_n == n:
            for c in ent_cols:
                feat[c] = z[c]
        else:
            # length mismatch (shouldn't happen if catchup worked)
            for c in ent_cols:
                feat[c] = np.nan
    else:
        for c in ["entity_density", "entity_diversity", "entity_per_sentence"]:
            feat[c] = np.nan

    # QuRater (n, 4)
    qr_path = M2_DATA / "scores_qurater" / f"{stem}.npy"
    if qr_path.exists():
        qr = np.load(qr_path)
        if qr.shape[0] == n:
            for i, axis in enumerate(QURATER_AXES):
                feat[f"qr_{axis}"] = qr[:, i]
        else:
            for axis in QURATER_AXES:
                feat[f"qr_{axis}"] = np.nan
    else:
        for axis in QURATER_AXES:
            feat[f"qr_{axis}"] = np.nan

    # perplexity-corr scores
    for m in PC_MODELS:
        p = M2_DATA / "scores_pc" / m / f"{stem}.npy"
        if p.exists():
            arr = np.load(p)
            if len(arr) == n:
                feat[f"pc_{m}"] = arr
            else:
                feat[f"pc_{m}"] = np.nan
        else:
            feat[f"pc_{m}"] = np.nan

    # textbook
    tb = M2_DATA / "scores_textbook_quality" / f"{stem}.npy"
    if tb.exists():
        arr = np.load(tb)
        feat["textbook_quality"] = arr if len(arr) == n else np.nan
    else:
        feat["textbook_quality"] = np.nan

    # ai-generated
    ai = M2_DATA / "scores_ai_generated" / f"{stem}.npy"
    if ai.exists():
        arr = np.load(ai)
        feat["ai_generated"] = arr if len(arr) == n else np.nan
    else:
        feat["ai_generated"] = np.nan

    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=MASTER_PARQUET)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shards", type=int, nargs="*",
                    help="optionally process only these shards (debug)")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"{args.output} exists; --overwrite to rebuild.")
        return

    split = json.loads(VAL_SHARDS_JSON.read_text())
    train_shards = split["train"]
    val_shards = split["val"]
    ordered = train_shards + val_shards
    if args.shards:
        ordered = [s for s in ordered if s in args.shards]
    n_train = sum(1 for s in ordered if s in set(train_shards))

    print(f"loading annotations.parquet ...")
    ann_full = pd.read_parquet(M2_DATA / "annotations.parquet")
    ann_by_shard = {s: g.reset_index(drop=True) for s, g in ann_full.groupby("shard_idx")}

    rows = []
    miss_counts = {}
    for s in tqdm(ordered, desc="joining"):
        df = load_shard(s, ann_by_shard)
        df["__in_train_pool"] = s in set(train_shards)
        rows.append(df)
    master = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}: {master.shape}")
    print(f"  columns: {list(master.columns)}")
    nans = master.isna().sum()
    nans = nans[nans > 0]
    if len(nans):
        print("  NaN counts:")
        for c, v in nans.items():
            print(f"    {c}: {v:,} ({100*v/len(master):.2f}%)")


if __name__ == "__main__":
    main()
