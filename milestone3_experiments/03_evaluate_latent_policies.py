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

from config import (
    FACTOR_EM_OUTPUTS_NPZ,
    FIGURES_DIR,
    GIBBS_DRAWS_NPZ,
    LATENT_EM_DRAWS_NPZ,
    LATENT_POLICY_RESULTS_JSON,
    LATENT_POLICY_SELECTIONS_PARQUET,
    PREPARED_MODEL_NPZ,
)


POLICY_LABELS = {
    "random": "Random",
    "global_dclm": "Global DCLM",
    "within_topic_dclm": "Topic DCLM",
    "avg_score_oracle": "Avg filters",
    "hierarchical_pred": "Hier. pred.",
    "hierarchical_topic_floor": "Hier. floor",
    "one_factor_posterior": "1-factor",
    "one_factor_prior": "1-factor prior",
    "two_factor_posterior": "2-factor",
    "two_factor_topic_floor": "2-factor floor",
    "two_factor_prior": "2-factor prior",
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


def top_n(scores: np.ndarray, n_select: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argpartition(scores, -n_select)[-n_select:]] = True
    return selected


def within_group_top_fraction(scores: np.ndarray, groups: np.ndarray, fraction: float) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        n_group = max(1, int(round(fraction * len(idx))))
        chosen = idx[np.argpartition(scores[idx], -n_group)[-n_group:]]
        selected[chosen] = True
    return selected


def group_floor_then_global_fill(
    scores: np.ndarray,
    groups: np.ndarray,
    selection_fraction: float,
    floor_fraction: float,
) -> np.ndarray:
    n_select = int(round(selection_fraction * len(scores)))
    selected = within_group_top_fraction(scores, groups, floor_fraction)
    if selected.sum() < n_select:
        remaining = np.flatnonzero(~selected)
        n_add = n_select - int(selected.sum())
        selected[remaining[np.argpartition(scores[remaining], -n_add)[-n_add:]]] = True
    elif selected.sum() > n_select:
        chosen = np.flatnonzero(selected)
        keep = chosen[np.argpartition(scores[chosen], -n_select)[-n_select:]]
        selected = np.zeros(len(scores), dtype=bool)
        selected[keep] = True
    return selected


def entropy(codes: np.ndarray, n_levels: int) -> tuple[float, float]:
    counts = np.bincount(codes.astype(int), minlength=n_levels)
    probs = counts[counts > 0] / counts.sum()
    value = float(-np.sum(probs * np.log(probs)))
    normalized = value / np.log(n_levels) if n_levels > 1 else 0.0
    return value, normalized


def policy_metrics(
    selected: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    topic: np.ndarray,
    fmt: np.ndarray,
    token_count: np.ndarray,
    score_names: np.ndarray,
    n_topics: int,
    n_formats: int,
) -> dict[str, Any]:
    topic_entropy, topic_entropy_normalized = entropy(topic[selected], n_topics)
    format_entropy, format_entropy_normalized = entropy(fmt[selected], n_formats)
    selected_quality = quality[selected]
    return {
        "selected_count": int(selected.sum()),
        "mean_observed_quality": float(selected_quality.mean()),
        "se_observed_quality": float(selected_quality.std(ddof=1) / np.sqrt(selected.sum())),
        "mean_scores": dict(zip(score_names, y[selected].mean(axis=0))),
        "topic_entropy": topic_entropy,
        "topic_entropy_normalized": topic_entropy_normalized,
        "format_entropy": format_entropy,
        "format_entropy_normalized": format_entropy_normalized,
        "topic_coverage": int(np.unique(topic[selected]).size),
        "format_coverage": int(np.unique(fmt[selected]).size),
        "mean_token_count": float(token_count[selected].mean()),
        "total_token_count": float(token_count[selected].sum()),
    }


def build_policies(
    y: np.ndarray,
    topic: np.ndarray,
    score_names: np.ndarray,
    scores: dict[str, np.ndarray],
    selection_fraction: float,
    topic_floor_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    n_select = int(round(selection_fraction * len(y)))
    rng = np.random.default_rng(seed)
    random = np.zeros(len(y), dtype=bool)
    random[rng.choice(len(y), size=n_select, replace=False)] = True
    dclm_idx = int(np.where(score_names == "dclm_score")[0][0])
    return {
        "random": random,
        "global_dclm": top_n(y[:, dclm_idx], n_select),
        "within_topic_dclm": within_group_top_fraction(y[:, dclm_idx], topic, selection_fraction),
        "avg_score_oracle": top_n(scores["avg_score"], n_select),
        "hierarchical_pred": top_n(scores["hierarchical_pred"], n_select),
        "hierarchical_topic_floor": group_floor_then_global_fill(
            scores["hierarchical_pred"],
            topic,
            selection_fraction,
            topic_floor_fraction,
        ),
        "one_factor_posterior": top_n(scores["one_factor_posterior"], n_select),
        "one_factor_prior": top_n(scores["one_factor_prior"], n_select),
        "two_factor_posterior": top_n(scores["two_factor_posterior"], n_select),
        "two_factor_topic_floor": group_floor_then_global_fill(
            scores["two_factor_posterior"],
            topic,
            selection_fraction,
            topic_floor_fraction,
        ),
        "two_factor_prior": top_n(scores["two_factor_prior"], n_select),
    }


def write_selection_frame(
    path: Path,
    row_id: np.ndarray,
    topic: np.ndarray,
    fmt: np.ndarray,
    quality: np.ndarray,
    scores: dict[str, np.ndarray],
    policies: dict[str, np.ndarray],
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": row_id,
            "topic": topic,
            "format": fmt,
            "observed_quality": quality,
        }
    )
    for name, score in scores.items():
        frame[f"score_{name}"] = score
    for name, selected in policies.items():
        frame[f"selected_{name}"] = selected
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def figure_policy_bars(metrics: dict[str, Any], output: Path) -> None:
    policy_order = [
        "random",
        "global_dclm",
        "hierarchical_topic_floor",
        "one_factor_posterior",
        "two_factor_posterior",
        "two_factor_topic_floor",
        "avg_score_oracle",
    ]
    labels = [POLICY_LABELS[name] for name in policy_order]
    quality = np.array([metrics[name]["mean_observed_quality"] for name in policy_order])
    se = np.array([metrics[name]["se_observed_quality"] for name in policy_order])
    topic_entropy = np.array([metrics[name]["topic_entropy_normalized"] for name in policy_order])
    format_entropy = np.array([metrics[name]["format_entropy_normalized"] for name in policy_order])
    colors = ["#B9B0A2", "#A85E46", "#4F7F8F", "#7C6A9E", "#3B6C8C", "#5E8C61", "#2F2F2F"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    axes[0].bar(labels, quality, yerr=1.96 * se, color=colors)
    axes[0].set_ylabel("Mean held-out quality")
    axes[0].set_title("Quality")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)

    axes[1].bar(labels, topic_entropy, color=colors)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Normalized entropy")
    axes[1].set_title("Topic diversity")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)

    axes[2].bar(labels, format_entropy, color=colors)
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel("Normalized entropy")
    axes[2].set_title("Format diversity")
    axes[2].tick_params(axis="x", rotation=35, labelsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent-quality data-mix policies.")
    parser.add_argument("--model-data", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--gibbs-draws", type=Path, default=GIBBS_DRAWS_NPZ)
    parser.add_argument("--one-factor", type=Path, default=LATENT_EM_DRAWS_NPZ)
    parser.add_argument("--two-factor", type=Path, default=FACTOR_EM_OUTPUTS_NPZ)
    parser.add_argument("--output", type=Path, default=LATENT_POLICY_RESULTS_JSON)
    parser.add_argument("--selections-output", type=Path, default=LATENT_POLICY_SELECTIONS_PARQUET)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--topic-floor-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.selections_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    for path in [args.model_data, args.gibbs_draws, args.one_factor, args.two_factor]:
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    data = np.load(args.model_data, allow_pickle=True)
    gibbs = np.load(args.gibbs_draws, allow_pickle=True)
    one_factor = np.load(args.one_factor, allow_pickle=True)
    two_factor = np.load(args.two_factor, allow_pickle=True)
    test_mask = data["test_mask"].astype(bool)
    y = data["Y"][test_mask].astype(np.float64)
    topic = data["topic"][test_mask].astype(int)
    fmt = data["format"][test_mask].astype(int)
    token_count = data["token_count"][test_mask].astype(float)
    row_id = data["row_id"][test_mask]
    score_names = np.array([str(name) for name in data["score_names"]])
    quality = y.mean(axis=1)

    scores = {
        "avg_score": quality,
        "hierarchical_pred": gibbs["test_pred_mean"].astype(np.float64).mean(axis=1),
        "one_factor_posterior": one_factor["posterior_q"][test_mask].astype(np.float64),
        "one_factor_prior": one_factor["prior_q"][test_mask].astype(np.float64),
        "two_factor_posterior": two_factor["latent_posterior_score"][test_mask].astype(np.float64),
        "two_factor_prior": two_factor["latent_prior_score"][test_mask].astype(np.float64),
    }
    policies = build_policies(
        y,
        topic,
        score_names,
        scores,
        args.selection_fraction,
        args.topic_floor_fraction,
        args.seed,
    )
    n_topics = int(data["topic"].max()) + 1
    n_formats = int(data["format"].max()) + 1
    metrics = {
        name: policy_metrics(
            selected,
            y,
            quality,
            topic,
            fmt,
            token_count,
            score_names,
            n_topics,
            n_formats,
        )
        for name, selected in policies.items()
    }

    dclm_quality = metrics["global_dclm"]["mean_observed_quality"]
    dclm_topic_entropy = metrics["global_dclm"]["topic_entropy_normalized"]
    comparisons = {
        name: {
            "quality_minus_global_dclm": metrics[name]["mean_observed_quality"] - dclm_quality,
            "topic_entropy_minus_global_dclm": metrics[name]["topic_entropy_normalized"] - dclm_topic_entropy,
        }
        for name in metrics
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figures_dir / "latent_policy_comparison.png"
    figure_policy_bars(metrics, figure_path)
    write_selection_frame(args.selections_output, row_id, topic, fmt, quality, scores, policies)

    summary = {
        "model_data": str(args.model_data),
        "gibbs_draws": str(args.gibbs_draws),
        "one_factor": str(args.one_factor),
        "two_factor": str(args.two_factor),
        "selection_fraction": args.selection_fraction,
        "topic_floor_fraction": args.topic_floor_fraction,
        "test_rows": int(len(y)),
        "score_names": score_names,
        "metrics": metrics,
        "comparisons_vs_global_dclm": comparisons,
        "figures": {"latent_policy_comparison": str(figure_path)},
        "selections_output": str(args.selections_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))

    print(f"Wrote {args.output}")
    print(f"Wrote {args.selections_output}")
    print(f"Wrote {figure_path}")
    for name in metrics:
        print(
            f"{name}: quality={metrics[name]['mean_observed_quality']:.3f}, "
            f"topic_entropy={metrics[name]['topic_entropy_normalized']:.3f}, "
            f"delta_vs_dclm={comparisons[name]['quality_minus_global_dclm']:.3f}"
        )


if __name__ == "__main__":
    main()
