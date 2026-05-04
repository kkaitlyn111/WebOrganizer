from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import FACTOR_EM_OUTPUTS_NPZ, FIGURES_DIR


SCORE_LABELS = {
    "dclm_score": "DCLM",
    "fwedu_rounded": "FW-Edu",
    "pc_arc_easy": "ARC",
    "pc_piqa": "PIQA",
    "pc_sciq": "SciQ",
    "pc_lambada": "LAMBADA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Make report-ready experiment figures.")
    parser.add_argument("--factor", type=Path, default=FACTOR_EM_OUTPUTS_NPZ)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    if not args.factor.exists():
        raise SystemExit(f"{args.factor} not found. Run 02_fit_factor_quality_em.py first.")
    data = np.load(args.factor, allow_pickle=True)
    loading = data["loading"].astype(float)
    score_names = [str(name) for name in data["score_names"]]
    labels = [SCORE_LABELS.get(name, name) for name in score_names]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.0), constrained_layout=True)
    ax.bar(x - width / 2, loading[:, 0], width=width, label="Factor 1", color="#3B6C8C")
    ax.bar(x + width / 2, loading[:, 1], width=width, label="Factor 2", color="#D98C48")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title("Two latent filter axes")
    ax.legend(frameon=False, ncol=2)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    output = args.figures_dir / "factor_loadings.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
