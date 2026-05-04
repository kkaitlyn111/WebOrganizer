from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import FACTOR_EM_OUTPUTS_NPZ, FACTOR_EM_RESULTS_JSON, PREPARED_MODEL_NPZ


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
    format_oh = one_hot(fmt, n_formats)
    domain_oh = one_hot(domain, n_domains)
    parts.extend([topic_oh, format_oh, domain_oh])
    names.extend([f"topic_{i}" for i in range(1, n_topics)])
    names.extend([f"format_{i}" for i in range(1, n_formats)])
    names.extend([f"domain_{i}" for i in range(1, n_domains)])
    return np.column_stack(parts), names


def ridge_regression(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def posterior_factors(
    z: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    loading: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prior_mean = z @ gamma
    precision = np.eye(loading.shape[1]) + loading.T @ ((1.0 / psi)[:, None] * loading)
    posterior_cov = np.linalg.inv(precision)
    residual_term = (y - alpha) @ ((1.0 / psi)[:, None] * loading)
    posterior_mean = (prior_mean + residual_term) @ posterior_cov.T
    return posterior_mean, posterior_cov


def update_measurement_model(
    y: np.ndarray,
    h_mean: np.ndarray,
    h_cov: np.ndarray,
    min_psi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = y.shape
    r = h_mean.shape[1]
    sum_h = h_mean.sum(axis=0)
    sum_hh = h_mean.T @ h_mean + n * h_cov
    lhs = np.empty((r + 1, r + 1), dtype=np.float64)
    lhs[0, 0] = n
    lhs[0, 1:] = sum_h
    lhs[1:, 0] = sum_h
    lhs[1:, 1:] = sum_hh

    alpha = np.empty(k, dtype=np.float64)
    loading = np.empty((k, r), dtype=np.float64)
    psi = np.empty(k, dtype=np.float64)
    for j in range(k):
        rhs = np.concatenate(([y[:, j].sum()], y[:, j] @ h_mean))
        coef = np.linalg.solve(lhs, rhs)
        alpha[j] = coef[0]
        loading[j] = coef[1:]

        linear = h_mean @ loading[j]
        quad = np.einsum("r,rs,s->", loading[j], h_cov, loading[j]) + linear**2
        resid_second = (
            y[:, j] ** 2
            - 2.0 * alpha[j] * y[:, j]
            - 2.0 * y[:, j] * linear
            + alpha[j] ** 2
            + 2.0 * alpha[j] * linear
            + quad
        )
        psi[j] = max(float(resid_second.mean()), min_psi)
    return alpha, loading, psi


def initialize_parameters(
    z: np.ndarray,
    y: np.ndarray,
    n_factors: int,
    ridge: float,
    min_psi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cov = np.cov(y, rowvar=False, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    components = eigvecs[:, np.argsort(eigvals)[::-1][:n_factors]]
    h = y @ components
    h = (h - h.mean(axis=0)) / np.maximum(h.std(axis=0, ddof=0), 1e-8)
    gamma = ridge_regression(z, h, ridge)
    h_prior = z @ gamma
    h_cov = np.eye(n_factors)
    alpha, loading, psi = update_measurement_model(y, h_prior, h_cov, min_psi)
    return gamma, alpha, loading, psi


def orient_factors(
    gamma: np.ndarray,
    loading: np.ndarray,
    h_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    for r in range(loading.shape[1]):
        if loading[:, r].sum() < 0:
            gamma[:, r] = -gamma[:, r]
            loading[:, r] = -loading[:, r]
            if h_mean is not None:
                h_mean[:, r] = -h_mean[:, r]
    return gamma, loading, h_mean


def marginal_loglik(
    y: np.ndarray,
    prior_mean: np.ndarray,
    alpha: np.ndarray,
    loading: np.ndarray,
    psi: np.ndarray,
) -> float:
    mean = alpha[None, :] + prior_mean @ loading.T
    cov = np.diag(psi) + loading @ loading.T
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    inv_cov = np.linalg.inv(cov)
    residual = y - mean
    quad = np.einsum("nk,kl,nl->n", residual, inv_cov, residual)
    return float(-0.5 * (len(y) * (y.shape[1] * np.log(2.0 * np.pi) + logdet) + quad.sum()))


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


def fit_em(
    z: np.ndarray,
    y: np.ndarray,
    n_factors: int,
    iterations: int,
    ridge: float,
    min_psi: float,
    tol: float,
) -> dict[str, Any]:
    gamma, alpha, loading, psi = initialize_parameters(z, y, n_factors, ridge, min_psi)
    gamma, loading, _ = orient_factors(gamma, loading)
    history: list[float] = []
    converged = False

    for iteration in range(iterations):
        h_mean, h_cov = posterior_factors(z, y, gamma, alpha, loading, psi)
        gamma, loading, h_mean = orient_factors(gamma, loading, h_mean)
        alpha, loading, psi = update_measurement_model(y, h_mean, h_cov, min_psi)
        gamma = ridge_regression(z, h_mean, ridge)
        gamma, loading, _ = orient_factors(gamma, loading)

        loglik = marginal_loglik(y, z @ gamma, alpha, loading, psi)
        history.append(loglik)
        if iteration > 3 and abs(history[-1] - history[-2]) < tol * (1.0 + abs(history[-2])):
            converged = True
            break

    h_mean, h_cov = posterior_factors(z, y, gamma, alpha, loading, psi)
    gamma, loading, h_mean = orient_factors(gamma, loading, h_mean)
    return {
        "gamma": gamma,
        "alpha": alpha,
        "loading": loading,
        "psi": psi,
        "posterior_factors_train": h_mean,
        "posterior_factor_cov": h_cov,
        "prior_factors_train": z @ gamma,
        "loglik_history": np.array(history, dtype=np.float64),
        "converged": converged,
        "iterations_run": len(history),
    }


def factor_summary(
    y: np.ndarray,
    factor_scores: np.ndarray,
    score_names: np.ndarray,
) -> dict[str, Any]:
    composite = y.mean(axis=1)
    out: dict[str, Any] = {}
    for r in range(factor_scores.shape[1]):
        score = factor_scores[:, r]
        out[f"factor_{r + 1}"] = {
            "corr_with_composite": float(np.corrcoef(score, composite)[0, 1]),
            "corr_by_score": dict(zip(score_names, [float(np.corrcoef(score, y[:, j])[0, 1]) for j in range(y.shape[1])])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a multi-factor latent-quality model with EM.")
    parser.add_argument("--input", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--output", type=Path, default=FACTOR_EM_RESULTS_JSON)
    parser.add_argument("--npz-output", type=Path, default=FACTOR_EM_OUTPUTS_NPZ)
    parser.add_argument("--n-factors", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--min-psi", type=float, default=1e-3)
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
        z_train,
        y[train_mask],
        n_factors=args.n_factors,
        iterations=args.iterations,
        ridge=args.ridge,
        min_psi=args.min_psi,
        tol=args.tol,
    )

    prior_factors_all = z_all @ fit["gamma"]
    posterior_factors_all, posterior_factor_cov = posterior_factors(
        z_all,
        y,
        fit["gamma"],
        fit["alpha"],
        fit["loading"],
        fit["psi"],
    )
    latent_weight = fit["loading"].mean(axis=0)
    latent_prior_score = prior_factors_all @ latent_weight
    latent_posterior_score = posterior_factors_all @ latent_weight
    y_pred_test_prior = fit["alpha"][None, :] + (z_test @ fit["gamma"]) @ fit["loading"].T
    y_pred_test_posterior = fit["alpha"][None, :] + posterior_factors_all[test_mask] @ fit["loading"].T
    y_test = y[test_mask]

    cov = np.diag(fit["psi"]) + fit["loading"] @ fit["loading"].T
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    composite_quality = y_test.mean(axis=1)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "npz_output": str(args.npz_output),
        "model": "Multi-factor latent filter model with covariate-dependent factor prior means",
        "em": {
            "n_factors": args.n_factors,
            "iterations_requested": args.iterations,
            "iterations_run": fit["iterations_run"],
            "converged": fit["converged"],
            "ridge": args.ridge,
            "min_psi": args.min_psi,
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
        "loading": {
            score: {f"factor_{r + 1}": float(fit["loading"][j, r]) for r in range(args.n_factors)}
            for j, score in enumerate(score_names)
        },
        "unique_variance": dict(zip(score_names, fit["psi"])),
        "factor_summary_on_test": factor_summary(y_test, posterior_factors_all[test_mask], score_names),
        "latent_score_correlations_on_test": {
            "posterior_latent_score_vs_composite_quality": float(
                np.corrcoef(latent_posterior_score[test_mask], composite_quality)[0, 1]
            ),
            "prior_latent_score_vs_composite_quality": float(
                np.corrcoef(latent_prior_score[test_mask], composite_quality)[0, 1]
            ),
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
        posterior_factor_cov=posterior_factor_cov.astype(np.float32),
        prior_factors=prior_factors_all.astype(np.float32),
        posterior_factors=posterior_factors_all.astype(np.float32),
        latent_prior_score=latent_prior_score.astype(np.float32),
        latent_posterior_score=latent_posterior_score.astype(np.float32),
        train_mask=train_mask,
        test_mask=test_mask,
        row_id=data["row_id"],
        score_names=score_names.astype(object),
        design_names=np.array(design_names, dtype=object),
        factor_names=np.array([f"factor_{r + 1}" for r in range(args.n_factors)], dtype=object),
        test_pred_prior=y_pred_test_prior.astype(np.float32),
        test_pred_posterior=y_pred_test_posterior.astype(np.float32),
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.npz_output}")
    print(f"EM iterations: {fit['iterations_run']} converged={fit['converged']}")
    print(f"Final log likelihood: {fit['loglik_history'][-1]:.2f}")
    print(f"Covariate-only test RMSE: {summary['test_prediction_from_covariates']['mean_rmse']:.4f}")
    print(f"Posterior reconstruction RMSE: {summary['test_reconstruction_from_filters']['mean_rmse']:.4f}")
    print(
        "posterior latent score vs composite quality corr: "
        f"{summary['latent_score_correlations_on_test']['posterior_latent_score_vs_composite_quality']:.4f}"
    )


if __name__ == "__main__":
    main()
