"""Split the 100 sampled shards into 90 train pool + 10 LM val."""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import VAL_SHARDS_JSON, SEED, N_LM_VAL_SHARDS, M2_DATA  # noqa: E402

SAMPLED = M2_DATA / "sampled_shards.json"


def main():
    shards = json.loads(SAMPLED.read_text())["shards"]
    assert len(shards) == 100
    rng = random.Random(SEED)
    val = sorted(rng.sample(shards, N_LM_VAL_SHARDS))
    train = sorted(s for s in shards if s not in set(val))
    VAL_SHARDS_JSON.write_text(json.dumps(
        {"seed": SEED, "val": val, "train": train}, indent=2))
    print(f"val:   {len(val)} shards -> {val}")
    print(f"train: {len(train)} shards")
    print(f"Saved to {VAL_SHARDS_JSON}")


if __name__ == "__main__":
    main()
