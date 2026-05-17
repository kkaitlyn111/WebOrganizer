"""Phase 4 setup: pick 100 fresh shard indices + download from HF.

Output:
  data/fresh_shards.json
  Corpus-200B/<subdir>/CC_shard_NNNNNNNN_processed.<ext>  (existing layout)
    documents/*.jsonl.zst, scores_dclm-fasttext/*.npy, ...
"""
from __future__ import annotations
import json
import random
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import DATA_DIR, REPO_ROOT  # noqa

FRESH_SEED = 1234
N_FRESH = 100
TOTAL_SHARDS = 9888


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="Actually download files (otherwise just plan)")
    ap.add_argument("--n", type=int, default=N_FRESH)
    args = ap.parse_args()

    sampled = json.loads((REPO_ROOT / "milestone2" / "data" / "sampled_shards.json").read_text())
    excluded = set(sampled["shards"])
    rng = random.Random(FRESH_SEED)
    candidates = [i for i in range(TOTAL_SHARDS) if i not in excluded]
    fresh = sorted(rng.sample(candidates, args.n))

    out = {"seed": FRESH_SEED, "shards": fresh, "n": len(fresh)}
    (DATA_DIR / "fresh_shards.json").write_text(json.dumps(out, indent=2))
    print(f"Picked {len(fresh)} fresh shards. Saved -> data/fresh_shards.json")
    print(f"First 5: {fresh[:5]}, last 5: {fresh[-5:]}")
    if not args.download:
        print("\nDry run. Pass --download to fetch.")
        return

    # Download via huggingface_hub
    from huggingface_hub import hf_hub_download
    REPO = "WebOrganizer/Corpus-200B"
    out_root = REPO_ROOT / "Corpus-200B"
    subdirs = [
        ("documents", ".jsonl.zst"),
        ("scores_dclm-fasttext", ".npy"),
        ("scores_fineweb-edu__rounded", "__rounded.npy"),
        ("domains_topics", "__choice.npy"),
        ("domains_formats", "__choice.npy"),
        ("domains_clusters-k24", ".npy"),
        ("tokens", ".npy"),
    ]
    from tqdm import tqdm
    for s in tqdm(fresh, desc="shards"):
        stem = f"CC_shard_{s:08d}_processed"
        for subdir, suffix in subdirs:
            fname = f"{subdir}/{stem}{suffix}"
            try:
                hf_hub_download(repo_id=REPO, filename=fname,
                                 repo_type="dataset",
                                 local_dir=str(out_root))
            except Exception as e:
                tqdm.write(f"  shard {s} {subdir}: {e}")


if __name__ == "__main__":
    main()
