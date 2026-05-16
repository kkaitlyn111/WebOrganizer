"""Smoke test: run extract_synth_features on a handful of docs from one shard."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "milestone2"))

from config import SAMPLED_SHARDS_JSON  # type: ignore
from _shared_io import iter_docs  # type: ignore
from config import doc_path  # type: ignore

from heuristics_synth import extract_synth_features, SYNTH_FEATURE_COLS

import json


def main():
    shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    s = shards[0]
    print(f"Reading first 5 docs from shard {s} -> {doc_path(s)}")
    for i, (doc_idx, obj) in enumerate(iter_docs(doc_path(s), max_docs=5)):
        text = obj.get("text", "") or ""
        feats = extract_synth_features(text)
        print(f"\n--- doc {doc_idx} (len={len(text)} chars) ---")
        for k in SYNTH_FEATURE_COLS:
            v = feats[k]
            print(f"  {k:30s} = {v!r}")


if __name__ == "__main__":
    main()
