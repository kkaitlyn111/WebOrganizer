from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    ANNOTATIONS_PARQUET,
    ARTIFACTS_DIR,
    FEATURES_DIR,
    MODEL_FRAME_PARQUET,
    MODEL_FRAME_SUMMARY_JSON,
    PC_MODELS,
    PC_SCORES_DIR,
    QURATER_AXES,
    QURATER_DIR,
    SAMPLED_SHARDS_JSON,
    shard_stem,
)


def load_shards(max_shards: int | None, explicit_shards: list[int] | None) -> list[int]:
    if explicit_shards:
        shards = explicit_shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if max_shards is not None:
        shards = shards[:max_shards]
    return [int(s) for s in shards]


def _empty_summary() -> dict[str, Any]:
    return {
        "rows": 0,
        "shards_requested": 0,
        "shards_joined": 0,
        "skipped_shards": [],
        "pc_score_shards": {model: 0 for model in PC_MODELS},
        "qurater_shards": 0,
    }


def build_shard_frame(
    shard_idx: int,
    annotations_by_shard: dict[int, pd.DataFrame],
    include_qurater: bool,
) -> tuple[pd.DataFrame | None, dict[str, int | bool]]:
    stem = shard_stem(shard_idx)
    status: dict[str, int | bool] = {
        "has_annotations": False,
        "has_features": False,
        "has_qurater": False,
    }
    for model in PC_MODELS:
        status[f"has_pc_{model}"] = False

    annotations = annotations_by_shard.get(shard_idx)
    if annotations is None or annotations.empty:
        return None, status
    status["has_annotations"] = True

    feature_path = FEATURES_DIR / f"{stem}.parquet"
    if not feature_path.exists():
        return None, status
    features = pd.read_parquet(feature_path)
    status["has_features"] = True

    df = annotations.merge(features, on="doc_idx", how="inner", validate="one_to_one")
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))

    doc_idx = df["doc_idx"].to_numpy()
    for model in PC_MODELS:
        score_path = PC_SCORES_DIR / model / f"{stem}.npy"
        col = f"pc_{model}"
        if score_path.exists():
            scores = np.load(score_path)
            if len(scores) != len(annotations):
                raise ValueError(
                    f"{score_path} has length {len(scores)} but annotations for shard "
                    f"{shard_idx} have length {len(annotations)}"
                )
            df[col] = scores[doc_idx].astype(np.float32)
            status[f"has_pc_{model}"] = True
        else:
            df[col] = np.nan

    if include_qurater:
        qurater_path = QURATER_DIR / f"{stem}.npy"
        if qurater_path.exists():
            qr = np.load(qurater_path)
            if qr.shape != (len(annotations), len(QURATER_AXES)):
                raise ValueError(
                    f"{qurater_path} has shape {qr.shape}; expected "
                    f"({len(annotations)}, {len(QURATER_AXES)})"
                )
            for i, axis in enumerate(QURATER_AXES):
                df[f"qr_{axis}"] = qr[doc_idx, i].astype(np.float32)
            status["has_qurater"] = True
        else:
            for axis in QURATER_AXES:
                df[f"qr_{axis}"] = np.nan

    return df, status


def write_summary(
    frame: pd.DataFrame,
    statuses: dict[int, dict[str, int | bool]],
    output: Path,
    summary_output: Path,
) -> None:
    summary = _empty_summary()
    summary["rows"] = int(len(frame))
    summary["shards_requested"] = int(len(statuses))
    summary["shards_joined"] = int(frame["shard_idx"].nunique()) if len(frame) else 0
    summary["skipped_shards"] = [
        int(s)
        for s, status in statuses.items()
        if not (status.get("has_annotations") and status.get("has_features"))
    ]
    for model in PC_MODELS:
        summary["pc_score_shards"][model] = int(
            sum(bool(status.get(f"has_pc_{model}")) for status in statuses.values())
        )
    summary["qurater_shards"] = int(sum(bool(status.get("has_qurater")) for status in statuses.values()))
    summary["columns"] = list(frame.columns)
    summary["missing_counts"] = {
        col: int(value)
        for col, value in frame.isna().sum().items()
        if int(value) > 0
    }
    summary["output"] = str(output)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Milestone 3 joined modeling frame.")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--shards", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=MODEL_FRAME_PARQUET)
    parser.add_argument("--summary-output", type=Path, default=MODEL_FRAME_SUMMARY_JSON)
    parser.add_argument("--include-qurater", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite to rebuild.")
    if not ANNOTATIONS_PARQUET.exists():
        raise SystemExit(f"{ANNOTATIONS_PARQUET} not found.")

    shards = load_shards(args.max_shards, args.shards)
    annotations = pd.read_parquet(ANNOTATIONS_PARQUET)
    annotations_by_shard = {
        int(shard_idx): group.reset_index(drop=True)
        for shard_idx, group in annotations.groupby("shard_idx", sort=False)
    }

    frames: list[pd.DataFrame] = []
    statuses: dict[int, dict[str, int | bool]] = {}
    for shard_idx in tqdm(shards, desc="joining shards"):
        frame, status = build_shard_frame(shard_idx, annotations_by_shard, args.include_qurater)
        statuses[shard_idx] = status
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise SystemExit("No shards joined. Check root data/ artifacts.")

    model_frame = pd.concat(frames, ignore_index=True)
    model_frame["row_id"] = np.arange(len(model_frame), dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_parquet(args.output, index=False)
    write_summary(model_frame, statuses, args.output, args.summary_output)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(model_frame):,}")
    print(f"Columns: {len(model_frame.columns)}")
    print(f"Shards joined: {model_frame['shard_idx'].nunique()} / {len(shards)}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()

