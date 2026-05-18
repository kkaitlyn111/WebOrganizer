"""Load a saved checkpoint and run downstream benchmarks.

Usage:
  python eval_checkpoint_bench.py --ckpt path/to/V3_ckpt.pt --rid V3 --out-dir validation_results_eq_60m_300M
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import torch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from train_lm import build_model, evaluate, MODEL_PRESETS  # noqa
from bench_eval import eval_benchmarks  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--rid", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bench-subset", type=int, default=None)
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    preset = obj.get("preset", "15m")
    vocab_size = obj.get("vocab_size", 50304)
    print(f"loading {args.ckpt}  preset={preset}  vocab={vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(vocab_size, preset=preset).to(device)
    model.load_state_dict({k: v.float() for k, v in obj["state_dict"].items()})
    model.eval()
    backbone = model.model
    lm_head_weight = model.lm_head.weight

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    t0 = time.time()
    bench = eval_benchmarks(backbone, lm_head_weight, tok, device,
                             bf16=True, subset=args.bench_subset)
    print(f"benchmarks done in {time.time()-t0:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # merge into existing results JSON if present
    rp = args.out_dir / f"{args.rid}.json"
    if rp.exists():
        existing = json.loads(rp.read_text())
    else:
        existing = {"rid": args.rid}
    existing.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in bench.items()})
    rp.write_text(json.dumps(existing, indent=2, default=str))
    print(f"wrote {rp}")
    for k in sorted(bench):
        print(f"  {k:<20s} {bench[k]}")


if __name__ == "__main__":
    main()
