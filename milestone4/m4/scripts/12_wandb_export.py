"""Pull all diagnostic + validation runs from wandb into one CSV for analysis.

Outputs:
  data/wandb_diagnostic.csv  — one row per diagnostic run (config + val_loss + summary)
  data/wandb_validation.csv  — same for validation
  data/figures/wandb_overview.pdf — scatter of val_loss vs n_selected, colored by type
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import DATA_DIR  # noqa


def fetch(project, entity=None):
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    rows = []
    for r in runs:
        row = {"name": r.name, "id": r.id, "state": r.state,
                "created": str(r.created_at), "url": r.url}
        row.update({f"cfg.{k}": v for k, v in r.config.items()})
        row.update({f"sum.{k}": v for k, v in r.summary.items() if not k.startswith("_")})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=== diagnostic ===")
    diag = fetch("m4-diagnostic")
    print(f"  {len(diag)} runs")
    diag.to_csv(DATA_DIR / "wandb_diagnostic.csv", index=False)
    print(f"  wrote {DATA_DIR / 'wandb_diagnostic.csv'}")

    print("=== validation ===")
    try:
        val = fetch("m4-validation")
        print(f"  {len(val)} runs")
        val.to_csv(DATA_DIR / "wandb_validation.csv", index=False)
    except Exception as e:
        print(f"  no validation project yet ({e})")
        val = pd.DataFrame()

    # Overview scatter
    if len(diag) and "sum.val_loss" in diag.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for typ, group in diag.groupby("cfg.type"):
            ax.scatter(group["cfg.n_selected"], group["sum.val_loss"],
                       label=typ, alpha=0.6, s=15)
        ax.set_xscale("log")
        ax.set_xlabel("n_selected docs"); ax.set_ylabel("val_loss")
        ax.set_title("Diagnostic runs: val_loss vs selected docs")
        ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        out = DATA_DIR / "figures" / "wandb_overview.pdf"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out); plt.close()
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
