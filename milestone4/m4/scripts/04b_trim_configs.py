"""Drop configs whose selection has < U unique tokens.

Reads diagnostic_run_configs.json, computes selection size in unique tokens,
writes diagnostic_run_configs_trimmed.json (kept configs only) and
diagnostic_run_configs_dropped.json (dropped, for the record).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import MASTER_PARQUET, DATA_DIR, DIAGNOSTIC_CONFIGS_JSON  # noqa
from run_diagnostic import apply_selection  # noqa

U = 150_000_000


def main():
    df = pd.read_parquet(MASTER_PARQUET)
    df = df[df["__in_train_pool"]].reset_index(drop=True)
    print(f"train pool: {len(df):,} docs, {df['token_count'].sum():,} tokens")

    cfgs = json.loads(Path(DIAGNOSTIC_CONFIGS_JSON).read_text())
    print(f"input configs: {len(cfgs)}")

    kept, dropped = [], []
    by_type_kept = {}
    by_type_drop = {}
    for c in cfgs:
        try:
            mask = apply_selection(df, c)
            n_uniq = int(df.loc[mask, "token_count"].sum())
        except Exception as e:
            print(f"  ERROR run {c.get('run_id')}: {e}")
            n_uniq = 0
        c2 = dict(c)
        c2["n_unique_tokens"] = n_uniq
        if n_uniq >= U:
            kept.append(c2)
            by_type_kept[c["type"]] = by_type_kept.get(c["type"], 0) + 1
        else:
            dropped.append(c2)
            by_type_drop[c["type"]] = by_type_drop.get(c["type"], 0) + 1

    print(f"\nU = {U:,} unique tokens cutoff")
    print(f"  kept:    {len(kept)} / {len(cfgs)}")
    print(f"  dropped: {len(dropped)} / {len(cfgs)}")
    print(f"\nKept by type:")
    for typ, n in sorted(by_type_kept.items()):
        print(f"  {typ:<28s} {n}")
    print(f"Dropped by type:")
    for typ, n in sorted(by_type_drop.items()):
        print(f"  {typ:<28s} {n}")

    out_kept = DATA_DIR / "diagnostic_run_configs_trimmed.json"
    out_drop = DATA_DIR / "diagnostic_run_configs_dropped.json"
    out_kept.write_text(json.dumps(kept, indent=2))
    out_drop.write_text(json.dumps(dropped, indent=2))
    print(f"\nWrote {out_kept}")
    print(f"Wrote {out_drop}")


if __name__ == "__main__":
    main()
