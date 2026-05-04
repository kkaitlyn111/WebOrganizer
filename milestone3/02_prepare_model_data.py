from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    HEURISTIC_COLS,
    LOG1P_FEATURE_COLS,
    MODEL_FRAME_PARQUET,
    MODEL_SCORE_COLUMNS,
    PREPARED_MODEL_NPZ,
    PREPARED_MODEL_SUMMARY_JSON,
    PREPARED_SAMPLE_METADATA_PARQUET,
    PROBABILITY_SCORE_COLUMNS,
)


BASE_COLUMNS = [
    "row_id",
    "shard_idx",
    "doc_idx",
    "topic",
    "format",
    "url_registered_domain",
    "token_count",
]
FEATURE_COLUMNS = ["token_count", *HEURISTIC_COLS]
EPS = 1e-4
OTHER_DOMAIN = "__OTHER__"
MISSING_DOMAIN = "__MISSING__"


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    return value


def read_model_frame(path: Path) -> pd.DataFrame:
    columns = list(dict.fromkeys(BASE_COLUMNS + FEATURE_COLUMNS + MODEL_SCORE_COLUMNS))
    return pd.read_parquet(path, columns=columns)


def choose_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=sample_size, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def make_train_mask(n_rows: int, test_fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed + 17)
    return rng.random(n_rows) >= test_fraction


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def standardize(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train_mask].mean(axis=0)
    std = values[train_mask].std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std, mean, std


def transform_scores(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    transformed = []
    raw_ranges: dict[str, dict[str, float]] = {}

    for col in MODEL_SCORE_COLUMNS:
        x = df[col].astype("float64").to_numpy()
        raw_ranges[col] = {
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
        }
        if col in PROBABILITY_SCORE_COLUMNS:
            x = logit(x)
        transformed.append(x)

    y_raw = np.column_stack(transformed)
    y, mean, std = standardize(y_raw, train_mask)
    details = {
        "raw_ranges": raw_ranges,
        "train_transformed_mean": dict(zip(MODEL_SCORE_COLUMNS, mean)),
        "train_transformed_std": dict(zip(MODEL_SCORE_COLUMNS, std)),
        "probability_clip_epsilon": EPS,
    }
    return y.astype(np.float32), details


def transform_features(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x_df = df[FEATURE_COLUMNS].astype("float64").copy()
    for col in LOG1P_FEATURE_COLS:
        if col in x_df:
            x_df[col] = np.log1p(np.clip(x_df[col], 0.0, None))

    train_df = x_df.loc[train_mask]
    fill_values = train_df.median(axis=0, skipna=True).fillna(0.0)
    x_filled = x_df.fillna(fill_values)
    x, mean, std = standardize(x_filled.to_numpy(), train_mask)

    details = {
        "feature_columns": FEATURE_COLUMNS,
        "log1p_feature_columns": [col for col in LOG1P_FEATURE_COLS if col in FEATURE_COLUMNS],
        "fill_values": dict(zip(FEATURE_COLUMNS, fill_values.to_numpy())),
        "train_mean": dict(zip(FEATURE_COLUMNS, mean)),
        "train_std": dict(zip(FEATURE_COLUMNS, std)),
    }
    return x.astype(np.float32), details


def encode_domains(df: pd.DataFrame, train_mask: np.ndarray, top_domains: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    domains = df["url_registered_domain"].fillna(MISSING_DOMAIN).astype(str)
    top = domains.loc[train_mask].value_counts().head(top_domains).index.to_list()
    domain_names = np.array([OTHER_DOMAIN, *top], dtype=object)
    domain_to_code = {domain: i + 1 for i, domain in enumerate(top)}
    codes = domains.map(domain_to_code).fillna(0).astype("int32").to_numpy()

    details = {
        "top_domains_requested": top_domains,
        "domain_levels": int(len(domain_names)),
        "other_code": 0,
        "top_20_train_domains": domains.loc[train_mask].value_counts().head(20).to_dict(),
    }
    return codes, domain_names, details


def write_metadata(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    domain_codes: np.ndarray,
    domain_names: np.ndarray,
    output: Path,
) -> None:
    metadata = df[["row_id", "shard_idx", "doc_idx", "topic", "format", "token_count", *MODEL_SCORE_COLUMNS]].copy()
    metadata["split"] = np.where(train_mask, "train", "test")
    metadata["domain_code"] = domain_codes
    metadata["domain_name"] = [domain_names[code] for code in domain_codes]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output, index=False)


def build_summary(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    rows_available: int,
    df: pd.DataFrame,
    train_mask: np.ndarray,
    score_details: dict[str, Any],
    feature_details: dict[str, Any],
    domain_details: dict[str, Any],
    sample_size: int | None,
    seed: int,
    test_fraction: float,
) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "output": str(output_path),
        "metadata_output": str(metadata_path),
        "seed": seed,
        "sample_size_requested": sample_size,
        "rows_available": int(rows_available),
        "rows_sampled": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "test_fraction": float(test_fraction),
        "score_columns": MODEL_SCORE_COLUMNS,
        "probability_score_columns": PROBABILITY_SCORE_COLUMNS,
        "feature_details": feature_details,
        "score_details": score_details,
        "domain_details": domain_details,
        "topic_counts": df["topic"].value_counts().sort_index().to_dict(),
        "format_counts": df["format"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare transformed Milestone 3 model matrices.")
    parser.add_argument("--input", type=Path, default=MODEL_FRAME_PARQUET)
    parser.add_argument("--output", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--summary-output", type=Path, default=PREPARED_MODEL_SUMMARY_JSON)
    parser.add_argument("--metadata-output", type=Path, default=PREPARED_SAMPLE_METADATA_PARQUET)
    parser.add_argument("--sample-size", type=int, default=120_000)
    parser.add_argument("--top-domains", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.summary_output, args.metadata_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run 01_build_model_frame.py first.")

    frame = read_model_frame(args.input)
    frame = frame.dropna(subset=MODEL_SCORE_COLUMNS).reset_index(drop=True)
    rows_available = len(frame)
    sample = choose_sample(frame, args.sample_size, args.seed)
    train_mask = make_train_mask(len(sample), args.test_fraction, args.seed)

    y, score_details = transform_scores(sample, train_mask)
    x, feature_details = transform_features(sample, train_mask)
    domain_codes, domain_names, domain_details = encode_domains(sample, train_mask, args.top_domains)
    topic = sample["topic"].astype("int32").to_numpy()
    fmt = sample["format"].astype("int32").to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        Y=y,
        X=x,
        topic=topic,
        format=fmt,
        domain=domain_codes,
        train_mask=train_mask,
        test_mask=~train_mask,
        row_id=sample["row_id"].astype("int64").to_numpy(),
        shard_idx=sample["shard_idx"].astype("int32").to_numpy(),
        doc_idx=sample["doc_idx"].astype("int32").to_numpy(),
        token_count=sample["token_count"].astype("float64").to_numpy(),
        score_names=np.array(MODEL_SCORE_COLUMNS, dtype=object),
        feature_names=np.array(FEATURE_COLUMNS, dtype=object),
        domain_names=domain_names,
    )
    write_metadata(sample, train_mask, domain_codes, domain_names, args.metadata_output)

    summary = build_summary(
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata_output,
        rows_available=rows_available,
        df=sample,
        train_mask=train_mask,
        score_details=score_details,
        feature_details=feature_details,
        domain_details=domain_details,
        sample_size=args.sample_size,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    args.summary_output.write_text(json.dumps(as_jsonable(summary), indent=2))

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata_output}")
    print(f"Rows sampled: {len(sample):,} / {rows_available:,}")
    print(f"Train/test: {train_mask.sum():,} / {(~train_mask).sum():,}")
    print(f"Y shape: {y.shape}; X shape: {x.shape}; domain levels: {len(domain_names)}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
