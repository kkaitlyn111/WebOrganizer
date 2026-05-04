from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, LATENT_POLICY_SELECTIONS_PARQUET, ROBUSTNESS_RESULTS_JSON


POLICY_ORDER = [
    "global_dclm",
    "hierarchical_topic_floor",
    "one_factor_posterior",
    "two_factor_posterior",
    "two_factor_topic_floor",
    "avg_score_oracle",
]
POLICY_LABELS = {
    "global_dclm": "Global DCLM",
    "hierarchical_topic_floor": "Hier. floor",
    "one_factor_posterior": "1-factor",
    "two_factor_posterior": "2-factor",
    "two_factor_topic_floor": "2-factor floor",
    "avg_score_oracle": "Avg filters",
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


def summarize_draws(draws: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(draws.mean()),
        "sd": float(draws.std(ddof=1)),
        "q025": float(np.quantile(draws, 0.025)),
        "q500": float(np.quantile(draws, 0.5)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def bootstrap_policy_quality(
    frame: pd.DataFrame,
    policies: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    quality = frame["observed_quality"].to_numpy()
    selected = {
        policy: frame[f"selected_{policy}"].to_numpy(dtype=bool)
        for policy in policies
    }
    n = len(frame)
    draws = {policy: np.empty(n_bootstrap, dtype=np.float64) for policy in policies}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        q_b = quality[idx]
        for policy in policies:
            mask_b = selected[policy][idx]
            draws[policy][b] = q_b[mask_b].mean()
    return draws


def figure_ci(summary: dict[str, Any], output: Path) -> None:
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    means = np.array([summary["quality"][p]["mean"] for p in POLICY_ORDER])
    lows = np.array([summary["quality"][p]["q025"] for p in POLICY_ORDER])
    highs = np.array([summary["quality"][p]["q975"] for p in POLICY_ORDER])
    colors = ["#A85E46", "#4F7F8F", "#7C6A9E", "#3B6C8C", "#5E8C61", "#2F2F2F"]
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    ax.bar(labels, means, color=colors)
    ax.errorbar(labels, means, yerr=[means - lows, highs - means], fmt="none", ecolor="black", capsize=3)
    ax.axhline(summary["quality"]["global_dclm"]["mean"], color="#A85E46", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean held-out quality")
    ax.set_title("Bootstrap 95% intervals")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap robustness checks for latent data-mix policies.")
    parser.add_argument("--selections", type=Path, default=LATENT_POLICY_SELECTIONS_PARQUET)
    parser.add_argument("--output", type=Path, default=ROBUSTNESS_RESULTS_JSON)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite to rebuild.")
    if not args.selections.exists():
        raise SystemExit(f"{args.selections} not found. Run 03_evaluate_latent_policies.py first.")

    frame = pd.read_parquet(args.selections)
    draws = bootstrap_policy_quality(frame, POLICY_ORDER, args.n_bootstrap, args.seed)
    quality_summary = {policy: summarize_draws(draws[policy]) for policy in POLICY_ORDER}
    diff_vs_dclm = {
        policy: summarize_draws(draws[policy] - draws["global_dclm"])
        for policy in POLICY_ORDER
        if policy != "global_dclm"
    }
    summary = {
        "selections": str(args.selections),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "quality": quality_summary,
        "difference_vs_global_dclm": diff_vs_dclm,
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figures_dir / "policy_robustness_ci.png"
    figure_ci(summary, figure_path)
    summary["figures"] = {"policy_robustness_ci": str(figure_path)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {figure_path}")
    for policy, stats in diff_vs_dclm.items():
        print(
            f"{policy} - DCLM: mean={stats['mean']:.3f}, "
            f"95% CI=({stats['q025']:.3f}, {stats['q975']:.3f})"
        )


if __name__ == "__main__":
    main()
