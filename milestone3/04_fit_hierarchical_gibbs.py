from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import invwishart
from tqdm import trange

from config import BASELINE_RESULTS_JSON, GIBBS_DRAWS_NPZ, GIBBS_RESULTS_JSON, PREPARED_MODEL_NPZ


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


def build_design(
    x: np.ndarray,
    topic: np.ndarray,
    fmt: np.ndarray,
    domain: np.ndarray,
    feature_names: np.ndarray,
    n_topics: int,
    n_formats: int,
    n_domains: int,
) -> tuple[np.ndarray, list[str], dict[str, slice]]:
    parts = [np.ones((len(topic), 1), dtype=np.float64), x.astype(np.float64)]
    names = ["intercept", *[str(name) for name in feature_names]]
    block_slices: dict[str, slice] = {}
    start = 0
    block_slices["intercept"] = slice(start, start + 1)
    start += 1
    block_slices["features"] = slice(start, start + x.shape[1])
    start += x.shape[1]

    topic_oh = one_hot(topic.astype(int), n_topics)
    parts.append(topic_oh)
    names.extend([f"topic_{i}" for i in range(1, n_topics)])
    block_slices["topic"] = slice(start, start + topic_oh.shape[1])
    start += topic_oh.shape[1]

    format_oh = one_hot(fmt.astype(int), n_formats)
    parts.append(format_oh)
    names.extend([f"format_{i}" for i in range(1, n_formats)])
    block_slices["format"] = slice(start, start + format_oh.shape[1])
    start += format_oh.shape[1]

    domain_oh = one_hot(domain.astype(int), n_domains)
    parts.append(domain_oh)
    names.extend([f"domain_{i}" for i in range(1, n_domains)])
    block_slices["domain"] = slice(start, start + domain_oh.shape[1])

    design = np.column_stack(parts)
    return design, names, block_slices


def make_prior_precision(
    block_slices: dict[str, slice],
    p: int,
    intercept_sd: float,
    feature_sd: float,
    topic_sd: float,
    format_sd: float,
    domain_sd: float,
) -> tuple[np.ndarray, dict[str, float]]:
    prior_sd = np.empty(p, dtype=np.float64)
    prior_sd[block_slices["intercept"]] = intercept_sd
    prior_sd[block_slices["features"]] = feature_sd
    prior_sd[block_slices["topic"]] = topic_sd
    prior_sd[block_slices["format"]] = format_sd
    prior_sd[block_slices["domain"]] = domain_sd
    prior_precision = 1.0 / (prior_sd**2)
    return prior_precision, {
        "intercept": intercept_sd,
        "features": feature_sd,
        "topic": topic_sd,
        "format": format_sd,
        "domain": domain_sd,
    }


def symmetrize_positive_definite(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    min_eig = float(np.linalg.eigvalsh(matrix).min())
    if min_eig < jitter:
        matrix = matrix + np.eye(matrix.shape[0]) * (jitter - min_eig)
    return matrix


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, score_names: np.ndarray) -> dict[str, Any]:
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


def residual_correlation(sigma: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(sigma))
    return sigma / np.outer(sd, sd)


def rhat(chains: np.ndarray) -> float:
    chains = np.asarray(chains, dtype=np.float64)
    m, n = chains.shape
    if m < 2 or n < 2:
        return float("nan")
    chain_means = chains.mean(axis=1)
    b = n * np.var(chain_means, ddof=1)
    w = np.mean(np.var(chains, axis=1, ddof=1))
    if w <= 1e-12:
        return float("nan")
    var_hat = ((n - 1) / n) * w + b / n
    return float(np.sqrt(var_hat / w))


def ess(chains: np.ndarray, max_lag: int = 100) -> float:
    chains = np.asarray(chains, dtype=np.float64)
    m, n = chains.shape
    if m < 1 or n < 3:
        return float("nan")
    centered = chains - chains.mean(axis=1, keepdims=True)
    variances = np.mean(centered**2, axis=1)
    valid = variances > 1e-12
    if not np.any(valid):
        return float("nan")
    max_lag = min(max_lag, n - 2)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        autocorrs = []
        for chain, var in zip(centered[valid], variances[valid], strict=True):
            autocorrs.append(float(np.mean(chain[:-lag] * chain[lag:]) / var))
        rho = float(np.mean(autocorrs))
        if rho < 0:
            break
        rho_sum += rho
    return float((m * n) / (1.0 + 2.0 * rho_sum))


def weighted_level_variance(effects: np.ndarray, codes: np.ndarray, n_levels: int) -> np.ndarray:
    weights = np.bincount(codes.astype(int), minlength=n_levels).astype(np.float64)
    weights = weights / weights.sum()
    mean = weights @ effects
    second = weights @ (effects**2)
    return np.maximum(second - mean**2, 0.0)


def block_variances(
    beta: np.ndarray,
    block_slices: dict[str, slice],
    feature_cov: np.ndarray,
    topic_codes: np.ndarray,
    format_codes: np.ndarray,
    domain_codes: np.ndarray,
    n_topics: int,
    n_formats: int,
    n_domains: int,
) -> dict[str, np.ndarray]:
    feature_beta = beta[block_slices["features"]]
    feature_var = np.einsum("pk,pq,qk->k", feature_beta, feature_cov, feature_beta)

    zeros = np.zeros((1, beta.shape[1]), dtype=np.float64)
    topic_effects = np.vstack([zeros, beta[block_slices["topic"]]])
    format_effects = np.vstack([zeros, beta[block_slices["format"]]])
    domain_effects = np.vstack([zeros, beta[block_slices["domain"]]])
    return {
        "features": np.maximum(feature_var, 0.0),
        "topic": weighted_level_variance(topic_effects, topic_codes, n_topics),
        "format": weighted_level_variance(format_effects, format_codes, n_formats),
        "domain": weighted_level_variance(domain_effects, domain_codes, n_domains),
    }


def scalar_diagnostics(sigma_draws: np.ndarray, block_var_draws: np.ndarray, score_names: np.ndarray) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    block_names = ["features", "topic", "format", "domain"]
    for k, score in enumerate(score_names):
        diagnostics[f"sigma_{score}"] = {
            "rhat": rhat(sigma_draws[:, :, k, k]),
            "ess": ess(sigma_draws[:, :, k, k]),
        }
        for b, block in enumerate(block_names):
            diagnostics[f"var_{block}_{score}"] = {
                "rhat": rhat(block_var_draws[:, :, b, k]),
                "ess": ess(block_var_draws[:, :, b, k]),
            }
    mean_corr = []
    for chain in range(sigma_draws.shape[0]):
        chain_corr = []
        for draw in range(sigma_draws.shape[1]):
            corr = residual_correlation(sigma_draws[chain, draw])
            off_diag = corr[np.triu_indices_from(corr, k=1)]
            chain_corr.append(float(off_diag.mean()))
        mean_corr.append(chain_corr)
    diagnostics["mean_residual_correlation"] = {
        "rhat": rhat(np.array(mean_corr)),
        "ess": ess(np.array(mean_corr)),
    }
    return diagnostics


def summarize_baseline_comparison(gibbs_metrics: dict[str, Any]) -> dict[str, Any] | None:
    if not BASELINE_RESULTS_JSON.exists():
        return None
    baseline = json.loads(BASELINE_RESULTS_JSON.read_text())
    models = baseline["models"]
    best_name, best_model = min(
        models.items(),
        key=lambda item: item[1]["test_metrics"]["mean_rmse"],
    )
    best_rmse = best_model["test_metrics"]["mean_rmse"]
    best_r2 = best_model["test_metrics"]["mean_r2"]
    return {
        "best_baseline": best_name,
        "best_baseline_mean_rmse": best_rmse,
        "best_baseline_mean_r2": best_r2,
        "gibbs_mean_rmse": gibbs_metrics["mean_rmse"],
        "gibbs_mean_r2": gibbs_metrics["mean_r2"],
        "rmse_improvement_vs_best_baseline": best_rmse - gibbs_metrics["mean_rmse"],
        "r2_improvement_vs_best_baseline": gibbs_metrics["mean_r2"] - best_r2,
    }


def run_gibbs(
    data: np.lib.npyio.NpzFile,
    iterations: int,
    burn: int,
    thin: int,
    chains: int,
    seed: int,
    prior_precision: np.ndarray,
    block_slices: dict[str, slice],
    design_train: np.ndarray,
    y_train: np.ndarray,
    score_names: np.ndarray,
    feature_cov: np.ndarray,
    topic_train: np.ndarray,
    format_train: np.ndarray,
    domain_train: np.ndarray,
    n_topics: int,
    n_formats: int,
    n_domains: int,
    iw_df_prior: float,
    iw_scale_prior: np.ndarray,
) -> dict[str, Any]:
    n, p = design_train.shape
    k = y_train.shape[1]
    keep_per_chain = (iterations - burn + thin - 1) // thin
    if keep_per_chain <= 0:
        raise ValueError("No posterior draws retained. Check iterations, burn, and thin.")

    xtx = design_train.T @ design_train
    xty = design_train.T @ y_train
    yty = y_train.T @ y_train
    precision = xtx + np.diag(prior_precision)
    posterior_cov = np.linalg.inv(precision)
    posterior_cov = symmetrize_positive_definite(posterior_cov)
    posterior_mean = posterior_cov @ xty
    chol_v = np.linalg.cholesky(posterior_cov)
    df_sigma = iw_df_prior + n + p

    sigma_draws = np.empty((chains, keep_per_chain, k, k), dtype=np.float64)
    block_var_draws = np.empty((chains, keep_per_chain, 4, k), dtype=np.float64)
    coef_mean = np.zeros((p, k), dtype=np.float64)
    coef_m2 = np.zeros((p, k), dtype=np.float64)
    draw_count = 0

    init_ss = yty - 2.0 * posterior_mean.T @ xty + posterior_mean.T @ xtx @ posterior_mean
    sigma_init = symmetrize_positive_definite((iw_scale_prior + init_ss) / max(n - k, 1))

    for chain in range(chains):
        rng = np.random.default_rng(seed + 1000 * chain)
        sigma = sigma_init.copy()
        kept = 0
        for it in trange(iterations, desc=f"chain {chain + 1}/{chains}", leave=False):
            chol_sigma = np.linalg.cholesky(symmetrize_positive_definite(sigma))
            z = rng.standard_normal((p, k))
            beta = posterior_mean + chol_v @ z @ chol_sigma.T

            resid_ss = yty - 2.0 * beta.T @ xty + beta.T @ xtx @ beta
            prior_ss = (beta * prior_precision[:, None]).T @ beta
            scale = symmetrize_positive_definite(iw_scale_prior + resid_ss + prior_ss)
            sigma = invwishart.rvs(df=df_sigma, scale=scale, random_state=rng)
            sigma = np.atleast_2d(sigma)

            if it >= burn and (it - burn) % thin == 0:
                sigma_draws[chain, kept] = sigma
                vars_by_block = block_variances(
                    beta,
                    block_slices,
                    feature_cov,
                    topic_train,
                    format_train,
                    domain_train,
                    n_topics,
                    n_formats,
                    n_domains,
                )
                for b, block in enumerate(["features", "topic", "format", "domain"]):
                    block_var_draws[chain, kept, b] = vars_by_block[block]

                draw_count += 1
                delta = beta - coef_mean
                coef_mean += delta / draw_count
                coef_m2 += delta * (beta - coef_mean)
                kept += 1

    coef_sd = np.sqrt(coef_m2 / max(draw_count - 1, 1))
    return {
        "coef_mean": coef_mean,
        "coef_sd": coef_sd,
        "sigma_draws": sigma_draws,
        "block_var_draws": block_var_draws,
        "posterior_mean": posterior_mean,
        "posterior_cov_diag": np.diag(posterior_cov),
        "draw_count": draw_count,
        "score_names": score_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the Milestone 3 hierarchical model with Gibbs sampling.")
    parser.add_argument("--input", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--output", type=Path, default=GIBBS_RESULTS_JSON)
    parser.add_argument("--draws-output", type=Path, default=GIBBS_DRAWS_NPZ)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--burn", type=int, default=400)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--intercept-prior-sd", type=float, default=10.0)
    parser.add_argument("--feature-prior-sd", type=float, default=1.5)
    parser.add_argument("--topic-prior-sd", type=float, default=1.0)
    parser.add_argument("--format-prior-sd", type=float, default=1.0)
    parser.add_argument("--domain-prior-sd", type=float, default=0.5)
    parser.add_argument("--iw-df-prior", type=float, default=10.0)
    parser.add_argument("--iw-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.draws_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run 02_prepare_model_data.py first.")
    if args.burn >= args.iterations:
        raise SystemExit("--burn must be smaller than --iterations.")

    data = np.load(args.input, allow_pickle=True)
    y = data["Y"].astype(np.float64)
    x = data["X"].astype(np.float64)
    topic = data["topic"].astype(int)
    fmt = data["format"].astype(int)
    domain = data["domain"].astype(int)
    train_mask = data["train_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    score_names = np.array([str(name) for name in data["score_names"]])
    feature_names = np.array([str(name) for name in data["feature_names"]])

    n_topics = int(topic.max()) + 1
    n_formats = int(fmt.max()) + 1
    n_domains = int(len(data["domain_names"]))

    design_train, design_names, block_slices = build_design(
        x[train_mask],
        topic[train_mask],
        fmt[train_mask],
        domain[train_mask],
        feature_names,
        n_topics,
        n_formats,
        n_domains,
    )
    design_test, _, _ = build_design(
        x[test_mask],
        topic[test_mask],
        fmt[test_mask],
        domain[test_mask],
        feature_names,
        n_topics,
        n_formats,
        n_domains,
    )
    prior_precision, prior_sds = make_prior_precision(
        block_slices,
        design_train.shape[1],
        args.intercept_prior_sd,
        args.feature_prior_sd,
        args.topic_prior_sd,
        args.format_prior_sd,
        args.domain_prior_sd,
    )
    feature_train = x[train_mask]
    feature_cov = (feature_train.T @ feature_train) / max(len(feature_train), 1)
    iw_scale_prior = np.eye(len(score_names), dtype=np.float64) * args.iw_scale

    fit = run_gibbs(
        data=data,
        iterations=args.iterations,
        burn=args.burn,
        thin=args.thin,
        chains=args.chains,
        seed=args.seed,
        prior_precision=prior_precision,
        block_slices=block_slices,
        design_train=design_train,
        y_train=y[train_mask],
        score_names=score_names,
        feature_cov=feature_cov,
        topic_train=topic[train_mask],
        format_train=fmt[train_mask],
        domain_train=domain[train_mask],
        n_topics=n_topics,
        n_formats=n_formats,
        n_domains=n_domains,
        iw_df_prior=args.iw_df_prior,
        iw_scale_prior=iw_scale_prior,
    )

    coef_mean = fit["coef_mean"]
    test_pred = design_test @ coef_mean
    test_metrics = evaluate(y[test_mask], test_pred, score_names)
    sigma_mean = fit["sigma_draws"].reshape(-1, len(score_names), len(score_names)).mean(axis=0)
    sigma_corr_mean = residual_correlation(sigma_mean)
    block_var_mean = fit["block_var_draws"].reshape(-1, 4, len(score_names)).mean(axis=0)
    diagnostics = scalar_diagnostics(fit["sigma_draws"], fit["block_var_draws"], score_names)
    baseline_comparison = summarize_baseline_comparison(test_metrics)

    block_names = ["features", "topic", "format", "domain"]
    variance_summary = {
        block: dict(zip(score_names, block_var_mean[i]))
        for i, block in enumerate(block_names)
    }
    variance_summary["residual"] = dict(zip(score_names, np.diag(sigma_mean)))

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "draws_output": str(args.draws_output),
        "model": "Multivariate Gaussian hierarchical shrinkage regression",
        "mcmc": {
            "chains": args.chains,
            "iterations": args.iterations,
            "burn": args.burn,
            "thin": args.thin,
            "retained_draws": int(fit["draw_count"]),
            "seed": args.seed,
        },
        "dimensions": {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "scores": int(len(score_names)),
            "predictors": int(design_train.shape[1]),
            "topics": n_topics,
            "formats": n_formats,
            "domains": n_domains,
        },
        "priors": {
            "coefficient_prior_sd_by_block": prior_sds,
            "inverse_wishart_df_prior": args.iw_df_prior,
            "inverse_wishart_scale_diagonal": args.iw_scale,
        },
        "score_names": score_names,
        "test_metrics": test_metrics,
        "baseline_comparison": baseline_comparison,
        "residual_covariance_mean": dict(
            zip(score_names, [dict(zip(score_names, row)) for row in sigma_mean])
        ),
        "residual_correlation_mean": dict(
            zip(score_names, [dict(zip(score_names, row)) for row in sigma_corr_mean])
        ),
        "variance_summary": variance_summary,
        "diagnostics": diagnostics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))
    np.savez_compressed(
        args.draws_output,
        coef_mean=coef_mean.astype(np.float32),
        coef_sd=fit["coef_sd"].astype(np.float32),
        sigma_draws=fit["sigma_draws"].astype(np.float32),
        block_var_draws=fit["block_var_draws"].astype(np.float32),
        design_names=np.array(design_names, dtype=object),
        score_names=score_names.astype(object),
        block_names=np.array(block_names, dtype=object),
        test_pred_mean=test_pred.astype(np.float32),
        test_row_id=data["row_id"][test_mask],
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.draws_output}")
    print(f"Retained draws: {fit['draw_count']}")
    print(f"Test RMSE={test_metrics['mean_rmse']:.4f}, R2={test_metrics['mean_r2']:.4f}")
    if baseline_comparison:
        print(
            "Best baseline "
            f"{baseline_comparison['best_baseline']}: "
            f"RMSE improvement={baseline_comparison['rmse_improvement_vs_best_baseline']:.4f}, "
            f"R2 improvement={baseline_comparison['r2_improvement_vs_best_baseline']:.4f}"
        )


if __name__ == "__main__":
    main()
