"""
A4: AI-generated text detection via roberta-base-openai-detector
    openai-community/roberta-base-openai-detector

For each doc, compute P(FAKE) length-weighted over 512-token chunks.
Single GPU. Saves milestone2/data/scores_ai_generated/CC_shard_XXXXXXXX.npy.

Mirrors structure of milestone2/05_extra_scores.py toxicity scoring.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "milestone2"))

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path  # type: ignore
from _shared_io import iter_docs  # type: ignore

REPO = "openai-community/roberta-base-openai-detector"
CTX = 512
OUT_DIR = DATA_ROOT / "scores_ai_generated"


def _load_model(device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, use_fast=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(REPO, torch_dtype=dtype)
    model.eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    # canonical: 0=Real, 1=Fake
    fake_idx = next((i for i, n in id2label.items() if "fake" in n), 1)
    return model, tok, fake_idx


def _score_shard(shard_idx: int, model, tok, device: str, batch_size: int,
                  fake_idx: int, max_docs: int | None) -> np.ndarray:
    import torch
    per_doc_chunks: list[list[list[int]]] = []
    per_doc_lens: list[list[int]] = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx), max_docs=max_docs):
        text = obj.get("text", "") or ""
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            per_doc_chunks.append([[tok.cls_token_id or 0]])
            per_doc_lens.append([1])
            continue
        win = CTX - 2
        chunks = [ids[i:i + win] for i in range(0, len(ids), win)]
        per_doc_chunks.append(chunks)
        per_doc_lens.append([len(c) for c in chunks])

    flat = []
    for d, (chunks, lens) in enumerate(zip(per_doc_chunks, per_doc_lens)):
        for c, L in zip(chunks, lens):
            flat.append((d, c, L))
    order = sorted(range(len(flat)), key=lambda i: flat[i][2])

    n_docs = len(per_doc_chunks)
    sum_p = np.zeros(n_docs, dtype=np.float64)
    sum_w = np.zeros(n_docs, dtype=np.float64)
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else 0
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else 2
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            items = [flat[i] for i in idxs]
            max_len = max(it[2] for it in items) + 2
            input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(items), max_len), dtype=torch.long)
            for i, (_d, ids, L) in enumerate(items):
                input_ids[i, 0] = cls_id
                input_ids[i, 1:1 + L] = torch.tensor(ids, dtype=torch.long)
                input_ids[i, 1 + L] = sep_id
                attn[i, :2 + L] = 1
            input_ids = input_ids.to(device); attn = attn.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, fake_idx].cpu().numpy()
            for (d, _ids, L), p in zip(items, probs):
                sum_p[d] += float(p) * L
                sum_w[d] += L

    sum_w = np.maximum(sum_w, 1.0)
    return (sum_p / sum_w).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading {REPO}")
    model, tok, fake_idx = _load_model(device)
    print(f"fake_idx={fake_idx}  id2label={model.config.id2label}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for s in tqdm(shards, desc="ai_detector"):
        out_path = args.output_dir / f"{shard_stem(s)}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        t0 = time.time()
        arr = _score_shard(s, model, tok, device, args.batch_size, fake_idx,
                            args.max_docs)
        np.save(out_path, arr)
        tqdm.write(f"  shard {s}: {arr.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
