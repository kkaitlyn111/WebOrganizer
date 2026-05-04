from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import (
    FACTOR_EM_OUTPUTS_NPZ,
    FIGURES_DIR,
    FRONTIER_RESULTS_JSON,
    GIBBS_DRAWS_NPZ,
    PREPARED_MODEL_NPZ,
)


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
        n_group = max(1, int(round(fraction * len(idx)))) if fraction > 0 else 0
        if n_group > 0:
            selected[idx[np.argpartition(scores[idx], -n_group)[-n_group:]]] = True
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


def entropy(codes: np.ndarray, n_levels: int) -> float:
    counts = np.bincount(codes.astype(int), minlength=n_levels)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(n_levels))


def metrics(selected: np.ndarray, quality: np.ndarray, topic: np.ndarray, fmt: np.ndarray) -> dict[str, float | int]:
    return {
        "selected_count": int(selected.sum()),
        "mean_quality": float(quality[selected].mean()),
        "se_quality": float(quality[selected].std(ddof=1) / np.sqrt(selected.sum())),
        "topic_entropy": entropy(topic[selected], int(topic.max()) + 1),
        "format_entropy": entropy(fmt[selected], int(fmt.max()) + 1),
        "topic_coverage": int(np.unique(topic[selected]).size),
        "format_coverage": int(np.unique(fmt[selected]).size),
    }


def figure_frontier(
    frontier: list[dict[str, Any]],
    hierarchical_frontier: list[dict[str, Any]],
    references: dict[str, Any],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    floor = np.array([point["topic_floor_fraction"] for point in frontier])
    x = np.array([point["topic_entropy"] for point in frontier])
    y = np.array([point["mean_quality"] for point in frontier])
    sc = ax.scatter(x, y, c=floor, cmap="viridis", s=46, label="2-factor frontier")
    ax.plot(x, y, color="#3B6C8C", linewidth=1.3)
    hx = np.array([point["topic_entropy"] for point in hierarchical_frontier])
    hy = np.array([point["mean_quality"] for point in hierarchical_frontier])
    ax.plot(hx, hy, color="#777777", linewidth=1.1, linestyle="--", label="Hier. frontier")
    for name, ref in references.items():
        ax.scatter(ref["topic_entropy"], ref["mean_quality"], marker=ref["marker"], s=70, color=ref["color"], label=ref["label"])
    ax.set_xlabel("Topic entropy")
    ax.set_ylabel("Mean held-out quality")
    ax.set_title("Quality-diversity frontier")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Topic floor fraction")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-diversity frontiers for data-mix policies.")
    parser.add_argument("--model-data", type=Path, default=PREPARED_MODEL_NPZ)
    parser.add_argument("--gibbs-draws", type=Path, default=GIBBS_DRAWS_NPZ)
    parser.add_argument("--two-factor", type=Path, default=FACTOR_EM_OUTPUTS_NPZ)
    parser.add_argument("--output", type=Path, default=FRONTIER_RESULTS_JSON)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--floor-grid", type=float, nargs="+", default=[0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite to rebuild.")
    for path in [args.model_data, args.gibbs_draws, args.two_factor]:
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    data = np.load(args.model_data, allow_pickle=True)
    gibbs = np.load(args.gibbs_draws, allow_pickle=True)
    factor = np.load(args.two_factor, allow_pickle=True)
    test_mask = data["test_mask"].astype(bool)
    y = data["Y"][test_mask].astype(np.float64)
    topic = data["topic"][test_mask].astype(int)
    fmt = data["format"][test_mask].astype(int)
    score_names = np.array([str(name) for name in data["score_names"]])
    dclm_idx = int(np.where(score_names == "dclm_score")[0][0])
    quality = y.mean(axis=1)
    two_factor_score = factor["latent_posterior_score"][test_mask].astype(np.float64)
    hierarchical_score = gibbs["test_pred_mean"].astype(np.float64).mean(axis=1)
    n_select = int(round(args.selection_fraction * len(y)))

    frontier = []
    for floor_fraction in args.floor_grid:
        selected = group_floor_then_global_fill(
            two_factor_score,
            topic,
            args.selection_fraction,
            floor_fraction,
        )
        point = metrics(selected, quality, topic, fmt)
        point["topic_floor_fraction"] = float(floor_fraction)
        point["policy"] = "two_factor_topic_floor"
        frontier.append(point)

    hierarchical_frontier = []
    for floor_fraction in args.floor_grid:
        selected = group_floor_then_global_fill(
            hierarchical_score,
            topic,
            args.selection_fraction,
            floor_fraction,
        )
        point = metrics(selected, quality, topic, fmt)
        point["topic_floor_fraction"] = float(floor_fraction)
        point["policy"] = "hierarchical_topic_floor"
        hierarchical_frontier.append(point)

    random_selected = np.zeros(len(y), dtype=bool)
    random_selected[np.random.default_rng(305).choice(len(y), size=n_select, replace=False)] = True
    references = {
        "random": metrics(random_selected, quality, topic, fmt)
        | {"label": "Random", "marker": "o", "color": "#B9B0A2"},
        "global_dclm": metrics(top_n(y[:, dclm_idx], n_select), quality, topic, fmt)
        | {"label": "Global DCLM", "marker": "s", "color": "#A85E46"},
        "avg_score_oracle": metrics(top_n(quality, n_select), quality, topic, fmt)
        | {"label": "Avg filters", "marker": "*", "color": "#2F2F2F"},
    }
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figures_dir / "quality_diversity_frontier.png"
    figure_frontier(frontier, hierarchical_frontier, references, figure_path)

    summary = {
        "model_data": str(args.model_data),
        "two_factor": str(args.two_factor),
        "selection_fraction": args.selection_fraction,
        "floor_grid": args.floor_grid,
        "two_factor_frontier": frontier,
        "hierarchical_frontier": hierarchical_frontier,
        "references": references,
        "figures": {"quality_diversity_frontier": str(figure_path)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_jsonable(summary), indent=2))

    print(f"Wrote {args.output}")
    print(f"Wrote {figure_path}")
    best = max(frontier, key=lambda point: point["mean_quality"])
    most_diverse = max(frontier, key=lambda point: point["topic_entropy"])
    print(
        "Best two-factor frontier point: "
        f"floor={best['topic_floor_fraction']:.3f}, "
        f"quality={best['mean_quality']:.3f}, topic_entropy={best['topic_entropy']:.3f}"
    )
    print(
        "Most diverse two-factor frontier point: "
        f"floor={most_diverse['topic_floor_fraction']:.3f}, "
        f"quality={most_diverse['mean_quality']:.3f}, topic_entropy={most_diverse['topic_entropy']:.3f}"
    )


if __name__ == "__main__":
    main()
