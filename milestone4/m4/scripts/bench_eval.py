"""Zero-shot downstream benchmark eval using completion log-likelihood.

For each (context, [completion_1, ..., completion_k]) example we compute the
per-token mean cross-entropy of each completion (conditional on the context).
Argmin = predicted choice. Mean loss across examples is reported as a
continuous metric.

Benchmarks (all in HF datasets, all CPU-loadable):
  HellaSwag      validation split, 4-way multiple choice
  PIQA           validation split, 2-way multiple choice
  ARC-Easy       test split, 4-way multiple choice
  LAMBADA        test split, last-word prediction (loss-only)

For the multiple-choice tasks, we follow the lm-eval-harness convention:
  - context = the question / setup
  - completions = the k options
  - score(option) = sum over completion-token log-probs (NOT length normalized)
  - prediction = argmax score
  - accuracy = (prediction == gold)
  - per-token loss = -mean over all completion tokens of all options

For LAMBADA: target = last word, loss = -log P(target tokens | rest).

Designed to plug into train_lm.py.evaluate() — returns a dict.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


_BENCH_CACHE: dict = {}


def _load_bench(name: str, n_examples: int | None = None):
    """Return list[(context_str, [completion_str_1..k], gold_idx)]."""
    if name in _BENCH_CACHE:
        return _BENCH_CACHE[name]

    from datasets import load_dataset
    out: List[Tuple[str, List[str], int]] = []

    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for ex in ds:
            ctx = (ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"]).strip()
            comps = [" " + c for c in ex["endings"]]
            gold = int(ex["label"])
            out.append((ctx, comps, gold))
    elif name == "piqa":
        # ybisk/piqa requires legacy dataset scripts; use a parquet mirror.
        try:
            ds = load_dataset("ehovy/piqa", split="validation")
        except Exception:
            ds = load_dataset("piqa", split="validation", trust_remote_code=True)
        for ex in ds:
            ctx = "Question: " + ex["goal"] + "\nAnswer:"
            comps = [" " + ex["sol1"], " " + ex["sol2"]]
            gold = int(ex["label"])
            out.append((ctx, comps, gold))
    elif name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        for ex in ds:
            ctx = "Question: " + ex["question"] + "\nAnswer:"
            comps = [" " + c for c in ex["choices"]["text"]]
            try:
                gold = ex["choices"]["label"].index(ex["answerKey"])
            except ValueError:
                continue
            out.append((ctx, comps, gold))
    elif name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
        for ex in ds:
            txt = ex["text"]
            words = txt.rsplit(" ", 1)
            if len(words) != 2:
                continue
            ctx, tgt = words[0], " " + words[1]
            out.append((ctx, [tgt], 0))   # always idx 0 since only one option
    else:
        raise ValueError(f"unknown bench {name}")

    if n_examples is not None:
        out = out[:n_examples]
    _BENCH_CACHE[name] = out
    return out


def _completion_loss(backbone, lm_head_weight, tokenizer, device,
                      context: str, completion: str,
                      bf16: bool, max_len: int = 1024) -> tuple[float, int]:
    """Return (sum-of-token-CE for completion tokens, n_completion_tokens)."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + completion, add_special_tokens=False)
    # The completion tokens are the tail.
    n_ctx = len(ctx_ids)
    n_comp = len(full_ids) - n_ctx
    if n_comp <= 0:
        return 0.0, 0
    # Truncate from left if too long (keep all completion tokens)
    if len(full_ids) > max_len:
        keep_from = len(full_ids) - max_len
        full_ids = full_ids[keep_from:]
        n_ctx = max(0, n_ctx - keep_from)
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
            h = backbone(input_ids=x).last_hidden_state[0]   # (T, H)
            logits = h @ lm_head_weight.t()                  # (T, V)
        # log-prob of token at position t+1 = log_softmax(logits[t])[full_ids[t+1]]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # completion tokens are positions n_ctx ... len(full_ids)-1
        # predicted from positions n_ctx-1 ... len(full_ids)-2
        targets = torch.tensor(full_ids[n_ctx:], device=device)
        pred_pos = log_probs[n_ctx - 1: len(full_ids) - 1]
        token_logp = pred_pos.gather(1, targets.unsqueeze(1)).squeeze(1)
        total_loss = float(-token_logp.sum().item())
    return total_loss, n_comp


def eval_benchmarks(backbone, lm_head_weight, tokenizer, device,
                    bf16: bool = True, subset: int | None = None,
                    verbose: bool = True) -> dict:
    """Score a model on the 4 benchmarks. Returns a flat dict of
    {bench}_acc and {bench}_loss numbers."""
    results = {}
    benches = [("hellaswag", subset or 2000),
               ("arc_easy",  subset or 1500),
               ("lambada",   subset or 1500)]
    for bname, default_n in benches:
        # Use n_examples cap to keep runtime down
        examples = _load_bench(bname, n_examples=default_n)
        n_correct = 0
        n_total = 0
        total_loss = 0.0
        total_toks = 0
        for ctx, comps, gold in examples:
            losses = []
            for c in comps:
                l, nt = _completion_loss(backbone, lm_head_weight, tokenizer, device, ctx, c, bf16)
                losses.append(l)
                total_loss += l
                total_toks += nt
            pred = int(np.argmin(losses))
            n_total += 1
            if pred == gold:
                n_correct += 1
        acc = n_correct / max(1, n_total)
        avg_loss = total_loss / max(1, total_toks)
        results[f"{bname}_acc"] = acc
        results[f"{bname}_loss"] = avg_loss
        results[f"{bname}_n"] = n_total
        if verbose:
            print(f"  bench {bname:<10s}: n={n_total} acc={acc:.4f} loss={avg_loss:.4f}")
    return results
