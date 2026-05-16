# extra per-doc scores beyond the 10 main quality filters:
#   - flesch_kincaid: Flesch-Kincaid grade level (textstat, CPU, instant)
#   - toxicity:       s-nlp/roberta_toxicity_classifier (GPU, binary)
#
# output:
#   data/scores_flesch_kincaid/{stem}.npy   shape (n_docs,) float32, grade level
#   data/scores_toxicity/{stem}.npy         shape (n_docs,) float32, P(toxic)
#
# usage:
#   ../.venv/bin/python3 05_extra_scores.py --task flesch --shards 431 --max-docs 50
#   ../.venv/bin/python3 05_extra_scores.py --task toxicity --shards 431 --max-docs 50
#   ../.venv/bin/python3 05_extra_scores.py --task flesch   # full run, CPU pool
#   ../.venv/bin/python3 05_extra_scores.py --task toxicity # full run, single GPU

from __future__ import annotations
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path
from _shared_io import iter_docs

FK_DIR = DATA_ROOT / "scores_flesch_kincaid"
TOX_DIR = DATA_ROOT / "scores_toxicity"

TOX_REPO = "s-nlp/roberta_toxicity_classifier"
TOX_CTX = 512  # roberta max


# ---------- flesch-kincaid (CPU) ----------

def _fk_score_one(text: str) -> float:
    import textstat
    t = (text or "").strip()
    if not t:
        return float("nan")
    try:
        return float(textstat.flesch_kincaid_grade(t))
    except Exception:
        return float("nan")


def _fk_process_shard(args) -> tuple[int, int, float]:
    shard_idx, out_dir, overwrite = args
    out_path = out_dir / f"{shard_stem(shard_idx)}.npy"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0
    t0 = time.time()
    vals = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx)):
        vals.append(_fk_score_one(obj.get("text", "") or ""))
    arr = np.asarray(vals, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return shard_idx, len(arr), time.time() - t0


def run_flesch(shards: list[int], n_workers: int, overwrite: bool,
               max_docs: int | None):
    # max_docs isn't threaded through the pool path to keep the worker func
    # pickle-simple; for smoke testing pass --n-workers 1.
    out_dir = FK_DIR
    if max_docs is not None:
        if n_workers != 1:
            print(f"[warn] --max-docs forces --n-workers 1")
            n_workers = 1
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in tqdm(shards, desc="flesch_kincaid (smoke)"):
            out_path = out_dir / f"{shard_stem(s)}.npy"
            if out_path.exists() and not overwrite:
                continue
            t0 = time.time()
            vals = []
            for _doc_idx, obj in iter_docs(doc_path(s), max_docs=max_docs):
                vals.append(_fk_score_one(obj.get("text", "") or ""))
            arr = np.asarray(vals, dtype=np.float32)
            np.save(out_path, arr)
            tqdm.write(f"  shard {s}: {arr.shape} in {time.time()-t0:.1f}s")
        return

    work = [(s, out_dir, overwrite) for s in shards]
    print(f"\n=== flesch_kincaid: {len(work)} shards, {n_workers} workers ===")
    totals = [0, 0.0]
    if n_workers == 1:
        for w in tqdm(work, desc="flesch_kincaid"):
            _, n, el = _fk_process_shard(w)
            if n >= 0:
                totals[0] += n; totals[1] += el
    else:
        with Pool(n_workers) as pool:
            for _, n, el in tqdm(pool.imap_unordered(_fk_process_shard, work),
                                 total=len(work), desc="flesch_kincaid"):
                if n >= 0:
                    totals[0] += n; totals[1] += el
    print(f"  flesch_kincaid: {totals[0]:,} docs, "
          f"{totals[0] / max(totals[1], 1e-9):.0f} docs/sec/worker")


# ---------- toxicity (GPU) ----------

def _load_tox_model(device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOX_REPO, use_fast=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        TOX_REPO, torch_dtype=dtype
    )
    model.eval().to(device)
    # verify label mapping: expect {0: neutral, 1: toxic}
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    toxic_idx = next((i for i, n in id2label.items() if "toxic" in n), 1)
    return model, tok, toxic_idx


def score_shard_toxicity(shard_idx: int, model, tok, device: str,
                         batch_size: int, toxic_idx: int,
                         max_docs: int | None = None) -> np.ndarray:
    import torch
    # collect per-doc chunked token ids (first 512-tok window only).
    # toxicity is usually evident from any portion; we length-weight-average
    # over chunks of length <=512 to cover the whole doc cheaply.
    per_doc_chunks: list[list[list[int]]] = []
    per_doc_lens: list[list[int]] = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx), max_docs=max_docs):
        text = obj.get("text", "") or ""
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            per_doc_chunks.append([[tok.cls_token_id or 0]])
            per_doc_lens.append([1])
            continue
        # leave room for <s> and </s> special tokens
        win = TOX_CTX - 2
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
            max_len = max(it[2] for it in items) + 2  # +<s>+</s>
            input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(items), max_len), dtype=torch.long)
            for i, (_d, ids, L) in enumerate(items):
                input_ids[i, 0] = cls_id
                input_ids[i, 1:1 + L] = torch.tensor(ids, dtype=torch.long)
                input_ids[i, 1 + L] = sep_id
                attn[i, :2 + L] = 1
            input_ids = input_ids.to(device); attn = attn.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, toxic_idx].cpu().numpy()
            for (d, _ids, L), p in zip(items, probs):
                sum_p[d] += float(p) * L
                sum_w[d] += L

    sum_w = np.maximum(sum_w, 1.0)
    return (sum_p / sum_w).astype(np.float32)


def run_toxicity(shards: list[int], batch_size: int, overwrite: bool,
                 max_docs: int | None, output_dir: Path):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading {TOX_REPO} ...")
    model, tok, toxic_idx = _load_tox_model(device)
    print(f"model loaded. toxic_idx={toxic_idx}  id2label={model.config.id2label}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for s in tqdm(shards, desc="toxicity"):
        out_path = output_dir / f"{shard_stem(s)}.npy"
        if out_path.exists() and not overwrite:
            continue
        t0 = time.time()
        arr = score_shard_toxicity(s, model, tok, device, batch_size, toxic_idx,
                                    max_docs)
        np.save(out_path, arr)
        tqdm.write(f"  shard {s}: {arr.shape} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["flesch", "toxicity", "both"], required=True)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Per-shard doc cap for smoke testing")
    ap.add_argument("--overwrite", action="store_true")
    # flesch only
    ap.add_argument("--n-workers", type=int, default=4)
    # toxicity only
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tox-output-dir", type=Path, default=TOX_DIR)
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    if args.task in ("flesch", "both"):
        run_flesch(shards, args.n_workers, args.overwrite, args.max_docs)
    if args.task in ("toxicity", "both"):
        run_toxicity(shards, args.batch_size, args.overwrite, args.max_docs,
                     args.tox_output_dir)


if __name__ == "__main__":
    main()
