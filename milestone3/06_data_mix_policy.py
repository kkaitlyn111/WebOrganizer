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
    DATA_MIX_RESULTS_JSON,
    DATA_MIX_SELECTIONS_PARQUET,
    FIGURES_DIR,
    GIBBS_DRAWS_NPZ,
    PREPARED_MODEL_NPZ,
)


POLICY_LABELS = {
    "random": "Random",
    "global_dclm": "Global DCLM",
    "within_topic_dclm": "Topic DCLM",
    "model_score": "Model score",
    "model_topic_floor": "Model floor",
    "model_within_topic": "Model topic mix",
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
    idx = np.argpartition(scores, -n_select)[-n_select:]
    selected[idx] = True
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
        add = remaining[np.argpartition(scores[remaining], -n_add)[-n_add:]]
        selected[add] = True
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
    pred_quality: np.ndarray,
    observed_quality: np.ndarray,
    topic: np.ndarray,
    fmt: np.ndarray,
    token_count: np.ndarray,
    score_names: np.ndarray,
    n_topics: int,
    n_formats: int,
) -> dict[str, Any]:
    topic_entropy, topic_entropy_norm = entropy(topic[selected], n_topics)
    format_entropy, format_entropy_norm = entropy(fmt[selected], n_formats)
    quality = observed_quality[selected]
    return {
        "selected_count": int(selected.sum()),
        "mean_observed_quality": float(quality.mean()),
        "se_observed_quality": float(quality.std(ddof=1) / np.sqrt(len(quality))),
        "mean_predicted_quality": float(pred_quality[selected].mean()),
        "mean_scores": dict(zip(score_names, y[selected].mean(axis=0))),
        "topic_entropy": topic_entropy,
        "topic_entropy_normalized": topic_entropy_norm,
        "format_entropy": format_entropy,
        "format_entropy_normalized": format_entropy_norm,
        "topic_coverage": int(np.unique(topic[selected]).size),
        "format_coverage": int(np.unique(fmt[selected]).size),
        "mean_token_count": float(token_count[selected].mean()),
        "total_token_count": float(token_count[selected].sum()),
        "topic_proportions": {
            int(i): float(v)
            for i, v in enumerate(np.bincount(topic[selected].astype(int), minlength=n_topics) / selected.sum())
        },
        "format_proportions": {
            int(i): float(v)
            for i, v in enumerate(np.bincount(fmt[selected].astype(int), minlength=n_formats) / selected.sum())
        },
    }


def make_policies(
    y: np.ndarray,
    pred_quality: np.ndarray,
    topic: np.ndarray,
    fraction: float,
    topic_floor_fraction: float,
    seed: int,
    score_names: np.ndarray,
) -> dict[str, np.ndarray]:
    n_select = int(round(fraction * len(y)))
    rng = np.random.default_rng(seed)
    random_selected = np.zeros(len(y), dtype=bool)
    random_selected[rng.choice(len(y), size=n_select, replace=False)] = True
    dclm_idx = int(np.where(score_names == "dclm_score")[0][0])
    return {
        "random": random_selected,
        "global_dclm": top_n(y[:, dclm_idx], n_select),
        "within_topic_dclm": within_group_top_fraction(y[:, dclm_idx], topic, fraction),
        "model_score": top_n(pred_quality, n_select),
        "model_topic_floor": group_floor_then_global_fill(
            pred_quality,
            topic,
            selection_fraction=fraction,
            floor_fraction=topic_floor_fraction,
        ),
        "model_within_topic": within_group_top_fraction(pred_quality, topic, fraction),
    }


def write_selection_frame(
    output: Path,
    row_id: np.ndarray,
    topic: np.ndarray,
    fmt: np.ndarray,
    token_count: np.ndarray,
    observed_quality: np.ndarray,
    pred_quality: np.ndarray,
    policies: dict[str, np.ndarray],
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": row_id,
            "topic": topic,
            "format": fmt,
            "token_count": token_count,
            "observed_quality": observed_quality,
            "predicted_quality": pred_quality,
        }
    )
    for name, selected in policies.items():
        frame[f"selected_{name}"] = selected
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def figure_policy_comparison(metrics: dict[str, Any], output: Path) -> None:
    names = list(metrics)
    labels = [POLICY_LABELS[name] for name in names]
    quality = np.array([metrics[name]["mean_observed_quality"] for name in names])
    se = np.array([metrics[name]["se_observed_quality"] for name in names])
    topic_entropy = np.array([metrics[name]["topic_entropy_normalized"] for name in names])
    format_entropy = np.array([metrics[name]["format_entropy_normalized"] for name in names])
    colors = ["#B9B0A2", "#A85E46", "#D98C48", "#3B6C8C", "#4F7F8F", "#5E8C61"]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)
    axes[0].bar(labels, quality, yerr=1.96 * se, color=colors)
    axes[0].set_ylabel("Mean held-out quality")
    axes[0].set_title("Quality lift")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)

    axes[1].bar(labels, topic_entropy, color=colors)
    axes[1].set_ylabel("Normalized entropy")
    axes[1].set_title("Topic diversity")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].set_ylim(0, 1.0)

    axes[2].bar(labels, format_entropy, color=colors)
    axes[2].set_ylabel("Normalized entropy")
    axes[2].set_title("Format diversity")
    axes[2].tick_params(axis="x", rotation=30, labelsize=8)
    axes[2].set_ylim(0, 1.0)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def figure_topic_mix(metrics: dict[str, Any], n_topics: int, output: Path) -> None:
    policy_names = ["random", "global_dclm", "model_score", "model_topic_floor", "model_within_topic"]
    topic_props = np.array([
        [metrics[policy]["topic_proportions"][topic] for topic in range(n_topics)]
        for policy in policy_names
    ])
    fig, ax = plt.subplots(figsize=(7.4, 2.8), constrained_layout=True)
    im = ax.imshow(topic_props, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(policy_names)))
    ax.set_yticklabels([POLICY_LABELS[name] for name in policy_names])
    ax.set_xticks(np.arange(n_topics))
    ax.set_xticklabels([str(i) for i in range(n_topics)], fontsize=7)
    ax.set_xlabel("Topic")
    ax.set_title("Selected topic proportions")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-based data-mix policies.")
    parser.add_argument("--model-data", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--gibbs-draws", type=Path, default=GIBBS_DRAWS_NPZ)
    parser.add_argument("--output", type=Path, default=DATA_MIX_RESULTS_JSON)
    parser.add_argument("--selections-output", type=Path, default=DATA_MIX_SELECTIONS_PARQUET)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--topic-floor-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.output, args.selections_output]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists. Pass --overwrite to rebuild.")
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit("--fraction must be between 0 and 1.")
    if not args.model_data.exists():
        raise SystemExit(f"{args.model_data} not found.")
    if not args.gibbs_draws.exists():
        raise SystemExit(f"{args.gibbs_draws} not found.")

    data = np.load(args.model_data, allow_pickle=True)
    draws = np.load(args.gibbs_draws, allow_pickle=True)
    test_mask = data["test_mask"].astype(bool)
    y = data["Y"][test_mask].astype(np.float64)
    pred = draws["test_pred_mean"].astype(np.float64)
    topic = data["topic"][test_mask].astype(int)
    fmt = data["format"][test_mask].astype(int)
    token_count = data["token_count"][test_mask].astype(float)
    row_id = data["row_id"][test_mask]
    score_names = np.array([str(name) for name in data["score_names"]])

    observed_quality = y.mean(axis=1)
    pred_quality = pred.mean(axis=1)
    policies = make_policies(
        y,
        pred_quality,
        topic,
        args.fraction,
        args.topic_floor_fraction,
        args.seed,
        score_names,
    )
    n_topics = int(data["topic"].max()) + 1
    n_formats = int(data["format"].max()) + 1
    metrics = {
        name: policy_metrics(
            selected,
            y,
            pred_quality,
            observed_quality,
            topic,
            fmt,
            token_count,
            score_names,
            n_topics,
            n_formats,
        )
        for name, selected in policies.items()
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    policy_fig = args.figures_dir / "data_mix_comparison.png"
    topic_fig = args.figures_dir / "data_mix_topic_mix.png"
    figure_policy_comparison(metrics, policy_fig)
    figure_topic_mix(metrics, n_topics, topic_fig)
    write_selection_frame(
        args.selections_output,
        row_id,
        topic,
        fmt,
        token_count,
        observed_quality,
        pred_quality,
        policies,
    )

    summary = {
        "model_data": str(args.model_data),
        "gibbs_draws": str(args.gibbs_draws),
        "fraction": args.fraction,
        "topic_floor_fraction": args.topic_floor_fraction,
        "test_rows": int(len(y)),
        "score_names": score_names,
        "metrics": metrics,
        "figures": {
            "data_mix_comparison": str(policy_fig),
            "topic_mix": str(topic_fig),
        },
        "selections_output": str(args.selections_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))

    print(f"Wrote {args.output}")
    print(f"Wrote {args.selections_output}")
    print(f"Wrote {policy_fig}")
    print(f"Wrote {topic_fig}")
    for name in metrics:
        print(
            f"{name}: quality={metrics[name]['mean_observed_quality']:.3f}, "
            f"topic_entropy={metrics[name]['topic_entropy_normalized']:.3f}, "
            f"format_entropy={metrics[name]['format_entropy_normalized']:.3f}"
        )


if __name__ == "__main__":
    main()
