from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import BASELINE_RESULTS_JSON, FIGURES_DIR, GIBBS_RESULTS_JSON, POSTERIOR_SUMMARY_JSON


SCORE_LABELS = {
    "dclm_score": "DCLM",
    "fwedu_rounded": "FW-Edu",
    "pc_arc_easy": "ARC",
    "pc_piqa": "PIQA",
    "pc_sciq": "SciQ",
    "pc_lambada": "LAMBADA",
}
BLOCK_LABELS = {
    "features": "Heuristics",
    "topic": "Topic",
    "format": "Format",
    "domain": "Domain",
    "residual": "Residual",
}
COLORS = {
    "features": "#3B6C8C",
    "topic": "#D98C48",
    "format": "#5E8C61",
    "domain": "#8E5E7D",
    "residual": "#C7C2B8",
}


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    return json.loads(path.read_text())


def figure_model_comparison(baseline: dict[str, Any], gibbs: dict[str, Any], output: Path) -> None:
    model_names = list(baseline["models"].keys()) + ["hierarchical"]
    rmse = [baseline["models"][name]["test_metrics"]["mean_rmse"] for name in baseline["models"]]
    rmse.append(gibbs["test_metrics"]["mean_rmse"])
    r2 = [baseline["models"][name]["test_metrics"]["mean_r2"] for name in baseline["models"]]
    r2.append(gibbs["test_metrics"]["mean_r2"])

    labels = ["Mean", "Topic+\nformat", "Heuristics", "TF+\nheur.", "Hier.\nBayes"]
    colors = ["#B9B0A2", "#A8B6A0", "#93A9B5", "#6D8EA0", "#B85C38"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    axes[0].bar(labels, rmse, color=colors)
    axes[0].set_ylabel("Held-out RMSE")
    axes[0].set_title("Prediction error")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylim(0, max(rmse) * 1.12)

    axes[1].bar(labels, r2, color=colors)
    axes[1].set_ylabel("Held-out R2")
    axes[1].set_title("Explained variance")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylim(min(0, min(r2)) - 0.03, max(r2) * 1.18)
    fig.suptitle("Hierarchical model adds modest predictive lift", fontsize=11)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def figure_variance_decomposition(gibbs: dict[str, Any], output: Path) -> dict[str, dict[str, float]]:
    scores = gibbs["score_names"]
    components = ["features", "topic", "format", "domain", "residual"]
    raw = gibbs["variance_summary"]
    normalized: dict[str, dict[str, float]] = {}
    matrix = np.zeros((len(components), len(scores)))
    for j, score in enumerate(scores):
        total = sum(float(raw[component][score]) for component in components)
        normalized[score] = {}
        for i, component in enumerate(components):
            value = float(raw[component][score]) / total if total > 0 else np.nan
            matrix[i, j] = value
            normalized[score][component] = value

    x = np.arange(len(scores))
    bottom = np.zeros(len(scores))
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for i, component in enumerate(components):
        ax.bar(
            x,
            matrix[i],
            bottom=bottom,
            label=BLOCK_LABELS[component],
            color=COLORS[component],
        )
        bottom += matrix[i]
    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS.get(score, score) for score in scores], rotation=20, ha="right")
    ax.set_ylabel("Share of modeled variance")
    ax.set_ylim(0, 1.0)
    ax.set_title("Most remaining variation is document-level residual")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return normalized


def figure_residual_correlation(gibbs: dict[str, Any], output: Path) -> None:
    scores = gibbs["score_names"]
    corr = np.array([[gibbs["residual_correlation_mean"][row][col] for col in scores] for row in scores])
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(scores)))
    ax.set_yticks(np.arange(len(scores)))
    labels = [SCORE_LABELS.get(score, score) for score in scores]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(scores)):
        for j in range(len(scores)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Residual score correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def figure_diagnostics(gibbs: dict[str, Any], output: Path) -> dict[str, float]:
    diagnostics = gibbs["diagnostics"]
    rhats = np.array([float(v["rhat"]) for v in diagnostics.values() if v["rhat"] is not None])
    esses = np.array([float(v["ess"]) for v in diagnostics.values() if v["ess"] is not None])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    axes[0].hist(rhats[np.isfinite(rhats)], bins=14, color="#6D8EA0", edgecolor="white")
    axes[0].axvline(1.05, color="#B85C38", linestyle="--", linewidth=1)
    axes[0].set_title("R-hat summaries")
    axes[0].set_xlabel("R-hat")
    axes[0].set_ylabel("Count")

    axes[1].hist(esses[np.isfinite(esses)], bins=14, color="#5E8C61", edgecolor="white")
    axes[1].set_title("Approx. ESS summaries")
    axes[1].set_xlabel("ESS")
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return {
        "max_rhat": float(np.nanmax(rhats)),
        "min_ess": float(np.nanmin(esses)),
        "median_ess": float(np.nanmedian(esses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create posterior summary tables and figures.")
    parser.add_argument("--baseline", type=Path, default=BASELINE_RESULTS_JSON)
    parser.add_argument("--gibbs", type=Path, default=GIBBS_RESULTS_JSON)
    parser.add_argument("--output", type=Path, default=POSTERIOR_SUMMARY_JSON)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite to rebuild.")

    baseline = load_json(args.baseline)
    gibbs = load_json(args.gibbs)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    model_comparison_path = args.figures_dir / "model_comparison.png"
    variance_path = args.figures_dir / "variance_decomposition.png"
    residual_corr_path = args.figures_dir / "residual_correlation.png"
    diagnostics_path = args.figures_dir / "diagnostics.png"

    figure_model_comparison(baseline, gibbs, model_comparison_path)
    normalized_variance = figure_variance_decomposition(gibbs, variance_path)
    figure_residual_correlation(gibbs, residual_corr_path)
    diagnostic_summary = figure_diagnostics(gibbs, diagnostics_path)

    summary = {
        "baseline_models": {
            name: model["test_metrics"]
            for name, model in baseline["models"].items()
        },
        "hierarchical_model": gibbs["test_metrics"],
        "baseline_comparison": gibbs["baseline_comparison"],
        "normalized_variance": normalized_variance,
        "diagnostic_summary": diagnostic_summary,
        "figures": {
            "model_comparison": str(model_comparison_path),
            "variance_decomposition": str(variance_path),
            "residual_correlation": str(residual_corr_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))

    print(f"Wrote {args.output}")
    for path in summary["figures"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
