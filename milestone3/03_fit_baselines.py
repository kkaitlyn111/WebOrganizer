from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import BASELINE_PREDICTIONS_NPZ, BASELINE_RESULTS_JSON, PREPARED_MODEL_NPZ


DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


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


def one_hot(codes: np.ndarray, n_levels: int) -> np.ndarray:
    if n_levels <= 1:
        return np.empty((len(codes), 0), dtype=np.float64)
    out = np.zeros((len(codes), n_levels - 1), dtype=np.float64)
    keep = codes > 0
    rows = np.flatnonzero(keep)
    out[rows, codes[keep] - 1] = 1.0
    return out


def design_matrix(data: np.lib.npyio.NpzFile, spec: str) -> tuple[np.ndarray, list[str]]:
    n = len(data["topic"])
    parts = [np.ones((n, 1), dtype=np.float64)]
    names = ["intercept"]

    if spec in {"heuristics", "topic_format_heuristics"}:
        x = data["X"].astype(np.float64)
        parts.append(x)
        names.extend(str(name) for name in data["feature_names"])

    if spec in {"topic_format", "topic_format_heuristics"}:
        topic = data["topic"].astype(int)
        fmt = data["format"].astype(int)
        n_topics = int(topic.max()) + 1
        n_formats = int(fmt.max()) + 1
        topic_oh = one_hot(topic, n_topics)
        format_oh = one_hot(fmt, n_formats)
        parts.extend([topic_oh, format_oh])
        names.extend([f"topic_{i}" for i in range(1, n_topics)])
        names.extend([f"format_{i}" for i in range(1, n_formats)])

    return np.column_stack(parts), names


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    return np.linalg.solve(lhs, rhs)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, score_names: np.ndarray) -> dict[str, Any]:
    residual = y_true - y_pred
    mse = np.mean(residual**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual), axis=0)
    sse = np.sum(residual**2, axis=0)
    centered = y_true - y_true.mean(axis=0)
    sst = np.sum(centered**2, axis=0)
    r2 = 1.0 - sse / np.where(sst <= 1e-12, np.nan, sst)
    return {
        "rmse_by_score": dict(zip(score_names, rmse)),
        "mae_by_score": dict(zip(score_names, mae)),
        "r2_by_score": dict(zip(score_names, r2)),
        "mean_rmse": float(rmse.mean()),
        "mean_mae": float(mae.mean()),
        "mean_r2": float(np.nanmean(r2)),
    }


def validation_split(train_mask: np.ndarray, seed: int, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(train_idx)) < val_fraction
    subtrain = train_idx[~is_val]
    val = train_idx[is_val]
    return subtrain, val


def choose_alpha(
    x: np.ndarray,
    y: np.ndarray,
    subtrain_idx: np.ndarray,
    val_idx: np.ndarray,
    alphas: list[float],
) -> tuple[float, list[dict[str, float]]]:
    scores = []
    best_alpha = alphas[0]
    best_rmse = np.inf
    for alpha in alphas:
        beta = fit_ridge(x[subtrain_idx], y[subtrain_idx], alpha)
        pred = x[val_idx] @ beta
        rmse = float(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
        scores.append({"alpha": float(alpha), "validation_rmse": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha, scores


def fit_models(
    data: np.lib.npyio.NpzFile,
    alphas: list[float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = data["Y"].astype(np.float64)
    train_mask = data["train_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    score_names = np.array([str(name) for name in data["score_names"]])
    subtrain_idx, val_idx = validation_split(train_mask, seed)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {"Y_test": y[test_idx].astype(np.float32)}

    mean_pred = np.broadcast_to(y[train_idx].mean(axis=0), y[test_idx].shape)
    results["mean"] = {
        "description": "Train-set mean for each standardized score.",
        "n_parameters": int(y.shape[1]),
        "test_metrics": evaluate(y[test_idx], mean_pred, score_names),
    }
    predictions["pred_mean"] = mean_pred.astype(np.float32)

    for spec in ["topic_format", "heuristics", "topic_format_heuristics"]:
        x, names = design_matrix(data, spec)
        best_alpha, validation_scores = choose_alpha(x, y, subtrain_idx, val_idx, alphas)
        beta = fit_ridge(x[train_idx], y[train_idx], best_alpha)
        pred = x[test_idx] @ beta
        results[spec] = {
            "description": {
                "topic_format": "Intercept plus topic and format fixed effects.",
                "heuristics": "Intercept plus standardized heuristic features.",
                "topic_format_heuristics": "Intercept plus standardized heuristic features, topic, and format fixed effects.",
            }[spec],
            "n_parameters": int(beta.size),
            "n_predictors": int(x.shape[1]),
            "best_alpha": float(best_alpha),
            "validation_scores": validation_scores,
            "test_metrics": evaluate(y[test_idx], pred, score_names),
        }
        predictions[f"pred_{spec}"] = pred.astype(np.float32)

    return results, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Milestone 3 baseline models.")
    parser.add_argument("--input", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--output", type=Path, default=BASELINE_RESULTS_JSON)
    parser.add_argument("--predictions-output", type=Path, default=BASELINE_PREDICTIONS_NPZ)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.predictions_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run 02_prepare_model_data.py first.")

    data = np.load(args.input, allow_pickle=True)
    results, predictions = fit_models(data, args.alphas, args.seed)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "predictions_output": str(args.predictions_output),
        "score_names": [str(name) for name in data["score_names"]],
        "models": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))
    np.savez_compressed(
        args.predictions_output,
        test_row_id=data["row_id"][data["test_mask"].astype(bool)],
        score_names=np.array([str(name) for name in data["score_names"]], dtype=object),
        **predictions,
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.predictions_output}")
    for name, result in results.items():
        metrics = result["test_metrics"]
        print(f"{name}: RMSE={metrics['mean_rmse']:.4f}, R2={metrics['mean_r2']:.4f}")


if __name__ == "__main__":
    main()
