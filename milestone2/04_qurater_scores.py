# score sampled shards with princeton-nlp/QuRater-1.3B on 4 quality axes

# per README:
#- logit order: 0 writing_style, 1 required_expertise, 2 facts_and_trivia, 3 educational_value
#- model trained on sequences up to 512 tokens -> chunk longer docs into 512-tok
#  windows and average logits weighted by window token count.

# output = one .npy per shard at data/scores_qurater/CC_shard_XXXXXXXX_processed.npy of shape (n_docs, 4) float32

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DATA_ROOT, SAMPLED_SHARDS_JSON, shard_stem, doc_path
from _shared_io import iter_docs

QURATER_REPO = "princeton-nlp/QuRater-1.3B"
AXIS_NAMES = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]
SCORES_DIR = DATA_ROOT / "scores_qurater"
CTX = 512  # per README


def load_model_and_tokenizer(device: str = "cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(QURATER_REPO, use_fast=True)
    tok.pad_token_id = 0
    model = AutoModelForSequenceClassification.from_pretrained(
        QURATER_REPO, torch_dtype=dtype
    )
    model.config.pad_token_id = 0
    model.eval()
    model.to(device)
    return model, tok


@torch.inference_mode()
def score_shard(shard_idx: int, model, tok, device: str, batch_size: int,
                max_docs: int | None = None) -> np.ndarray:
    # collect all chunk token-id lists, grouped by doc
    per_doc_chunks: list[list[list[int]]] = []
    per_doc_lens: list[list[int]] = []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx), max_docs=max_docs):
        text = obj.get("text", "") or ""
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            per_doc_chunks.append([[0]]); per_doc_lens.append([1])
            continue
        chunks = [ids[i:i + CTX] for i in range(0, len(ids), CTX)]
        per_doc_chunks.append(chunks)
        per_doc_lens.append([len(c) for c in chunks])

    # flatten into a single list with (doc_idx, chunk_tokens, chunk_len) and
    # score in batches sorted by length (reduces padding).
    flat = []
    for d, (chunks, lens) in enumerate(zip(per_doc_chunks, per_doc_lens)):
        for c, L in zip(chunks, lens):
            flat.append((d, c, L))
    order = sorted(range(len(flat)), key=lambda i: flat[i][2])

    n_docs = len(per_doc_chunks)
    sum_logits = np.zeros((n_docs, 4), dtype=np.float64)
    sum_weights = np.zeros(n_docs, dtype=np.float64)

    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batch_items = [flat[i] for i in idxs]
        max_len = max(it[2] for it in batch_items)
        input_ids = torch.zeros((len(batch_items), max_len), dtype=torch.long)
        attn = torch.zeros((len(batch_items), max_len), dtype=torch.long)
        for i, (_d, ids, L) in enumerate(batch_items):
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn[i, :L] = 1
        input_ids = input_ids.to(device); attn = attn.to(device)
        out = model(input_ids=input_ids, attention_mask=attn).logits  # (B, 4)
        out = out.float().cpu().numpy()
        for (d, _ids, L), row in zip(batch_items, out):
            sum_logits[d] += row * L
            sum_weights[d] += L

    sum_weights = np.maximum(sum_weights, 1)
    return (sum_logits / sum_weights[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Per-shard doc cap for smoke testing")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=SCORES_DIR)
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading {QURATER_REPO} ...")
    model, tok = load_model_and_tokenizer(device=device)
    print("model loaded.")

    for s in tqdm(shards, desc="shards"):
        out_path = args.output_dir / f"{shard_stem(s)}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        t0 = time.time()
        arr = score_shard(s, model, tok, device, args.batch_size, args.max_docs)
        np.save(out_path, arr)
        tqdm.write(f"  shard {s}: {arr.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
