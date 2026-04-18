# simple script that samples N_SHARDS indexes from range(TOTAL_SHARDS) with fixed seed.
# also writes data/sampled_shards.json and prints if .npy annotations r present

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from config import (
    SEED, N_SHARDS, TOTAL_SHARDS,
    DATA_ROOT, SAMPLED_SHARDS_JSON,
    ANNOT_SPECS, annot_path, doc_path,
)


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    shards = sorted(rng.choice(TOTAL_SHARDS, size=N_SHARDS, replace=False).tolist())

    payload = {
        "seed": SEED,
        "n_shards": N_SHARDS,
        "total_shards": TOTAL_SHARDS,
        "shards": shards,
    }
    SAMPLED_SHARDS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {SAMPLED_SHARDS_JSON} ({len(shards)} shards)")
    print(f"First 5: {shards[:5]}")
    print(f"Last 5:  {shards[-5:]}")

    # availability report
    keys = list(ANNOT_SPECS.keys()) + ["documents"]
    counts = {k: 0 for k in keys}
    for s in shards:
        if doc_path(s).exists():
            counts["documents"] += 1
        for k in ANNOT_SPECS:
            if annot_path(s, k).exists():
                counts[k] += 1

    print("\nAvailability of sampled shards:")
    print(f"  {'subdir':32s}  present / {N_SHARDS}")
    for k in keys:
        print(f"  {k:32s}  {counts[k]:6d}")

    missing_any = [
        s for s in shards
        if not doc_path(s).exists()
        or not all(annot_path(s, k).exists() for k in ANNOT_SPECS)
    ]
    print(f"\nShards missing at least one file: {len(missing_any)}")
    if missing_any[:5]:
        print(f"  first few: {missing_any[:5]}")


if __name__ == "__main__":
    main()
