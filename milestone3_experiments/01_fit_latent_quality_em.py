from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import LATENT_EM_DRAWS_NPZ, LATENT_EM_RESULTS_JSON, PREPARED_MODEL_NPZ


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


def build_design(data: np.lib.npyio.NpzFile, mask: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    if mask is None:
        mask = np.ones(len(data["topic"]), dtype=bool)
    x = data["X"][mask].astype(np.float64)
    topic = data["topic"][mask].astype(int)
    fmt = data["format"][mask].astype(int)
    domain = data["domain"][mask].astype(int)
    n_topics = int(data["topic"].max()) + 1
    n_formats = int(data["format"].max()) + 1
    n_domains = int(len(data["domain_names"]))

    parts = [np.ones((len(topic), 1), dtype=np.float64), x]
    names = ["intercept", *[str(name) for name in data["feature_names"]]]

    topic_oh = one_hot(topic, n_topics)
    parts.append(topic_oh)
    names.extend([f"topic_{i}" for i in range(1, n_topics)])

    format_oh = one_hot(fmt, n_formats)
    parts.append(format_oh)
    names.extend([f"format_{i}" for i in range(1, n_formats)])

    domain_oh = one_hot(domain, n_domains)
    parts.append(domain_oh)
    names.extend([f"domain_{i}" for i in range(1, n_domains)])

    return np.column_stack(parts), names


def ridge_regression(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def initialize_parameters(
    z: np.ndarray,
    y: np.ndarray,
    ridge: float,
    positive_loadings: bool,
    min_loading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = y.mean(axis=1)
    q = (q - q.mean()) / max(q.std(ddof=0), 1e-8)
    gamma = ridge_regression(z, q, ridge)
    q_mean = z @ gamma
    q_second = q_mean**2 + 1.0
    alpha, loading, psi = update_measurement_model(
        y,
        q_mean,
        q_second,
        min_psi=1e-3,
        positive_loadings=positive_loadings,
        min_loading=min_loading,
    )
    return gamma, alpha, loading, psi


def posterior_q(
    z: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    loading: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    prior_mean = z @ gamma
    precision = 1.0 + np.sum((loading**2) / psi)
    posterior_var = float(1.0 / precision)
    weighted_residual = ((y - alpha) * (loading / psi)).sum(axis=1)
    posterior_mean = posterior_var * (prior_mean + weighted_residual)
    posterior_second = posterior_var + posterior_mean**2
    return posterior_mean, posterior_second, posterior_var


def update_measurement_model(
    y: np.ndarray,
    q_mean: np.ndarray,
    q_second: np.ndarray,
    min_psi: float,
    positive_loadings: bool,
    min_loading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = y.shape
    sum_q = float(q_mean.sum())
    sum_q2 = float(q_second.sum())
    lhs = np.array([[n, sum_q], [sum_q, sum_q2]], dtype=np.float64)
    alpha = np.empty(k, dtype=np.float64)
    loading = np.empty(k, dtype=np.float64)
    psi = np.empty(k, dtype=np.float64)

    for j in range(k):
        rhs = np.array([y[:, j].sum(), np.dot(y[:, j], q_mean)], dtype=np.float64)
        alpha[j], loading[j] = np.linalg.solve(lhs, rhs)
        if positive_loadings:
            loading[j] = max(loading[j], min_loading)
        resid_second = (
            y[:, j] ** 2
            - 2.0 * alpha[j] * y[:, j]
            - 2.0 * loading[j] * y[:, j] * q_mean
            + alpha[j] ** 2
            + 2.0 * alpha[j] * loading[j] * q_mean
            + loading[j] ** 2 * q_second
        )
        psi[j] = max(float(resid_second.mean()), min_psi)

    return alpha, loading, psi


def marginal_loglik(
    y: np.ndarray,
    prior_mean: np.ndarray,
    alpha: np.ndarray,
    loading: np.ndarray,
    psi: np.ndarray,
) -> float:
    mean = alpha[None, :] + np.outer(prior_mean, loading)
    cov = np.diag(psi) + np.outer(loading, loading)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    inv_cov = np.linalg.inv(cov)
    residual = y - mean
    quad = np.einsum("nk,kl,nl->n", residual, inv_cov, residual)
    return float(-0.5 * (len(y) * (len(alpha) * np.log(2.0 * np.pi) + logdet) + quad.sum()))


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, score_names: np.ndarray) -> dict[str, Any]:
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(residual**2, axis=0))
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


def orient_latent_quality(
    gamma: np.ndarray,
    loading: np.ndarray,
    q_mean: np.ndarray,
    q_second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if float(loading.mean()) >= 0.0:
        return gamma, loading, q_mean, q_second
    gamma = -gamma
    loading = -loading
    q_mean = -q_mean
    return gamma, loading, q_mean, q_second


def fit_em(
    z: np.ndarray,
    y: np.ndarray,
    iterations: int,
    ridge: float,
    min_psi: float,
    positive_loadings: bool,
    min_loading: float,
    tol: float,
) -> dict[str, Any]:
    gamma, alpha, loading, psi = initialize_parameters(z, y, ridge, positive_loadings, min_loading)
    history: list[float] = []
    converged = False

    for iteration in range(iterations):
        q_mean, q_second, q_var = posterior_q(z, y, gamma, alpha, loading, psi)
        gamma, loading, q_mean, q_second = orient_latent_quality(gamma, loading, q_mean, q_second)
        alpha, loading, psi = update_measurement_model(
            y,
            q_mean,
            q_second,
            min_psi,
            positive_loadings=positive_loadings,
            min_loading=min_loading,
        )
        if not positive_loadings and loading.mean() < 0:
            gamma = -gamma
            loading = -loading

        q_mean, q_second, q_var = posterior_q(z, y, gamma, alpha, loading, psi)
        gamma = ridge_regression(z, q_mean, ridge)
        prior_mean = z @ gamma
        loglik = marginal_loglik(y, prior_mean, alpha, loading, psi)
        history.append(loglik)
        if iteration > 3 and abs(history[-1] - history[-2]) < tol * (1.0 + abs(history[-2])):
            converged = True
            break

    q_mean, q_second, q_var = posterior_q(z, y, gamma, alpha, loading, psi)
    gamma, loading, q_mean, q_second = orient_latent_quality(gamma, loading, q_mean, q_second)
    return {
        "gamma": gamma,
        "alpha": alpha,
        "loading": loading,
        "psi": psi,
        "posterior_q_train": q_mean,
        "posterior_q_var": q_var,
        "prior_q_train": z @ gamma,
        "loglik_history": np.array(history, dtype=np.float64),
        "converged": converged,
        "iterations_run": len(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a latent-quality factor model with EM.")
    parser.add_argument("--input", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--output", type=Path, default=LATENT_EM_RESULTS_JSON)
    parser.add_argument("--npz-output", type=Path, default=LATENT_EM_DRAWS_NPZ)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--min-psi", type=float, default=1e-3)
    parser.add_argument("--positive-loadings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-loading", type=float, default=1e-4)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.npz_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run the stable milestone3 pipeline first.")

    data = np.load(args.input, allow_pickle=True)
    y = data["Y"].astype(np.float64)
    train_mask = data["train_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    score_names = np.array([str(name) for name in data["score_names"]])

    z_train, design_names = build_design(data, train_mask)
    z_test, _ = build_design(data, test_mask)
    z_all, _ = build_design(data)
    fit = fit_em(
        z=z_train,
        y=y[train_mask],
        iterations=args.iterations,
        ridge=args.ridge,
        min_psi=args.min_psi,
        positive_loadings=args.positive_loadings,
        min_loading=args.min_loading,
        tol=args.tol,
    )

    prior_q_all = z_all @ fit["gamma"]
    posterior_q_all, _, posterior_q_var = posterior_q(
        z_all,
        y,
        fit["gamma"],
        fit["alpha"],
        fit["loading"],
        fit["psi"],
    )
    y_pred_test_prior = fit["alpha"][None, :] + np.outer(z_test @ fit["gamma"], fit["loading"])
    y_pred_test_posterior = fit["alpha"][None, :] + np.outer(posterior_q_all[test_mask], fit["loading"])
    y_test = y[test_mask]

    composite_quality = y_test.mean(axis=1)
    posterior_quality_corr = float(np.corrcoef(posterior_q_all[test_mask], composite_quality)[0, 1])
    prior_quality_corr = float(np.corrcoef(prior_q_all[test_mask], composite_quality)[0, 1])
    average_score = y_test.mean(axis=1)
    average_quality_corr = float(np.corrcoef(average_score, composite_quality)[0, 1])

    cov = np.diag(fit["psi"]) + np.outer(fit["loading"], fit["loading"])
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "npz_output": str(args.npz_output),
        "model": "One-factor latent document quality model with covariate-dependent prior mean",
        "em": {
            "iterations_requested": args.iterations,
            "iterations_run": fit["iterations_run"],
            "converged": fit["converged"],
            "ridge": args.ridge,
            "min_psi": args.min_psi,
            "positive_loadings": args.positive_loadings,
            "min_loading": args.min_loading,
            "tol": args.tol,
            "initial_loglik": float(fit["loglik_history"][0]),
            "final_loglik": float(fit["loglik_history"][-1]),
            "last_loglik_delta": float(fit["loglik_history"][-1] - fit["loglik_history"][-2])
            if len(fit["loglik_history"]) > 1
            else None,
        },
        "dimensions": {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "scores": int(y.shape[1]),
            "predictors": int(z_train.shape[1]),
        },
        "score_names": score_names,
        "measurement": {
            "alpha": dict(zip(score_names, fit["alpha"])),
            "loading": dict(zip(score_names, fit["loading"])),
            "unique_variance": dict(zip(score_names, fit["psi"])),
            "loading_share_of_marginal_variance": dict(
                zip(score_names, (fit["loading"] ** 2) / (fit["loading"] ** 2 + fit["psi"]))
            ),
        },
        "latent_correlations_on_test": {
            "posterior_q_vs_composite_quality": posterior_quality_corr,
            "prior_q_vs_composite_quality": prior_quality_corr,
            "average_score_vs_composite_quality": average_quality_corr,
        },
        "test_prediction_from_covariates": evaluate_prediction(y_test, y_pred_test_prior, score_names),
        "test_reconstruction_from_filters": evaluate_prediction(y_test, y_pred_test_posterior, score_names),
        "implied_score_correlation": dict(
            zip(score_names, [dict(zip(score_names, row)) for row in corr])
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))
    np.savez_compressed(
        args.npz_output,
        gamma=fit["gamma"].astype(np.float32),
        alpha=fit["alpha"].astype(np.float32),
        loading=fit["loading"].astype(np.float32),
        psi=fit["psi"].astype(np.float32),
        prior_q=prior_q_all.astype(np.float32),
        posterior_q=posterior_q_all.astype(np.float32),
        posterior_q_var=np.array([posterior_q_var], dtype=np.float32),
        train_mask=train_mask,
        test_mask=test_mask,
        row_id=data["row_id"],
        score_names=score_names.astype(object),
        design_names=np.array(design_names, dtype=object),
        test_pred_prior=y_pred_test_prior.astype(np.float32),
        test_pred_posterior=y_pred_test_posterior.astype(np.float32),
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.npz_output}")
    print(f"EM iterations: {fit['iterations_run']} converged={fit['converged']}")
    print(f"Final log likelihood: {fit['loglik_history'][-1]:.2f}")
    print(f"Covariate-only test RMSE: {summary['test_prediction_from_covariates']['mean_rmse']:.4f}")
    print(f"Posterior reconstruction RMSE: {summary['test_reconstruction_from_filters']['mean_rmse']:.4f}")
    print(f"posterior q vs composite quality corr: {posterior_quality_corr:.4f}")


if __name__ == "__main__":
    main()
