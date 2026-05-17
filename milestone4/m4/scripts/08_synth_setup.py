"""Phase 5: download synthetic datasets and extract features.

Datasets:
  HuggingFaceTB/cosmopedia       (50K docs)
  teknium/OpenHermes-2.5         (50K docs)
  HuggingFaceTB/ultrachat_200k   (50K docs)

Saves each dataset as a JSONL file under data/synth/<name>.jsonl
so the milestone4 feature extractors can be re-pointed at it.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import DATA_DIR  # noqa

SYNTH_DIR = DATA_DIR / "synth"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "auto_math_text", "train", "text"),
    # openhermes is conversation-formatted; we'll concatenate turns
    "openhermes": ("teknium/OpenHermes-2.5", None, "train", "conversations"),
    "ultrachat": ("HuggingFaceTB/ultrachat_200k", None, "train_sft", "messages"),
}


def render_conversation(c):
    if isinstance(c, list):
        return "\n".join((m.get("value") or m.get("content") or "") for m in c if isinstance(m, dict))
    return str(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    for name, (repo, config, split, field) in SOURCES.items():
        if args.only and name not in args.only:
            continue
        out = SYNTH_DIR / f"{name}.jsonl"
        if out.exists():
            print(f"  {name}: already at {out}, skipping")
            continue
        print(f"  loading {repo} ({split}) ...")
        if config:
            ds = load_dataset(repo, config, split=split, streaming=True)
        else:
            ds = load_dataset(repo, split=split, streaming=True)
        with open(out, "w", encoding="utf-8") as fh:
            for i, ex in enumerate(ds):
                if i >= args.n:
                    break
                val = ex.get(field)
                if val is None:
                    continue
                if field in ("conversations", "messages"):
                    text = render_conversation(val)
                else:
                    text = str(val)
                fh.write(json.dumps({"id": i, "text": text}) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
