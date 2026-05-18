"""Train a 15M Llama-style LM on selected tokens and report val loss.

This module is invoked by `run_diagnostic.py` and by validation runs.
It exposes a single function: train_and_eval(token_ids_array, val_sequences, cfg).

Architecture (~15M params):
  hidden=128, intermediate=384 (SwiGLU), 4 heads, 8 layers, RoPE base=10000

Training:
  AdamW (beta=(0.9, 0.95)), lr=0.007 cosine + 10% warmup, batch=64, seq=2048
  Token budget: 1B tokens -> 1B / (64 * 2048) ~= 7629 steps
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM


@dataclass
class TrainConfig:
    batch_size: int = 64          # effective batch size (token-rate target)
    micro_batch_size: int = 16    # actual forward batch (per accumulation step)
    seq_len: int = 1024
    total_tokens: int = 1_000_000_000
    lr: float = 7e-3
    warmup_frac: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    vocab_size: int = 50304       # GPT-NeoX rounded to multiple of 64
    seed: int = 0
    eval_every: int = 0           # if >0, eval val_loss every N steps
    log_every: int = 200
    eval_micro_batch: int = 16
    compile_model: bool = True    # only compiles the backbone (not the head)
    bf16: bool = True             # auto-disabled if unsupported
    ce_chunk_size: int = 16384    # chunk size for chunked CE (tokens)
    model_preset: str = "15m"     # '15m' | '60m' | '120m'
    run_bench: bool = False       # if True, run HellaSwag/PIQA/ARC/LAMBADA at end
    bench_subset: int | None = None  # cap per-bench examples (None = preset cap)
    save_checkpoint: str | None = None  # path to save bf16 state_dict (None = don't save)


MODEL_PRESETS = {
    "15m": dict(hidden_size=192, intermediate_size=576, num_hidden_layers=12,
                 num_attention_heads=6, num_key_value_heads=6),
    "60m": dict(hidden_size=384, intermediate_size=1024, num_hidden_layers=16,
                 num_attention_heads=8, num_key_value_heads=8),
    "120m": dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=18,
                 num_attention_heads=8, num_key_value_heads=8),
}


def build_model(vocab_size: int, preset: str = "15m") -> LlamaForCausalLM:
    """preset='15m' (default) keeps the original 15M-param shape; '60m'/'120m'
    scale hidden/inter/layers for larger comparisons."""
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except Exception:
        pass
    p = MODEL_PRESETS[preset]
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=p["hidden_size"],
        intermediate_size=p["intermediate_size"],
        num_hidden_layers=p["num_hidden_layers"],
        num_attention_heads=p["num_attention_heads"],
        num_key_value_heads=p["num_key_value_heads"],
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        use_cache=False,
        rms_norm_eps=1e-5,
        attn_implementation=attn_impl,
    )
    return LlamaForCausalLM(cfg)


def _supports_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _cosine_lr(step, max_steps, base_lr, warmup_steps, min_frac=0.1):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def prepare_token_stream(selected_token_arrays, total_tokens, seq_len,
                          max_repeat=8, rng=None):
    """Take a list/iterable of per-doc token arrays, concat, repeat to total_tokens,
    pack into (n_seqs, seq_len+1) sequences (last token is the shifted target).
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(selected_token_arrays, np.ndarray):
        stream = selected_token_arrays
    else:
        stream = np.concatenate(selected_token_arrays) if len(selected_token_arrays) else np.zeros(0, dtype=np.uint32)
    if len(stream) == 0:
        raise ValueError("No tokens selected")
    if len(stream) < total_tokens:
        n_repeats = min(max_repeat, int(np.ceil(total_tokens / len(stream))))
        stream = np.tile(stream, n_repeats)
    stream = stream[:total_tokens]
    chunk = seq_len + 1
    n_seqs = len(stream) // chunk
    stream = stream[: n_seqs * chunk].reshape(n_seqs, chunk)
    perm = rng.permutation(n_seqs)
    return stream[perm]


def chunked_loss(backbone, lm_head_weight, input_ids, labels,
                  bf16: bool, chunk_size: int = 4096,
                  reduction: str = "mean"):
    """Memory-efficient cross-entropy that NEVER materializes the full
    (B*T, V) logits or its gradient.

    Computes hidden = backbone(input_ids), then loops over chunks of (B*T)
    tokens, projecting hidden->logits and CE per chunk. The largest tensor
    materialized is (chunk_size, V) — at chunk=4096 and V=50304 in bf16
    that's only ~400 MB, vs. the 6.6 GB at (B*T=65536, V) for micro=32 seq=2048.

    Backward then accumulates grad w.r.t. lm_head_weight & hidden chunk-by-chunk.

    reduction: "mean" returns scalar loss = sum/total_tokens.
               "sum"  returns scalar loss = sum.
    """
    with torch.amp.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
        hidden = backbone(input_ids=input_ids).last_hidden_state  # (B, T, H)
    H = hidden.size(-1)
    flat_hidden = hidden.reshape(-1, H)
    flat_labels = labels.reshape(-1)
    N = flat_hidden.size(0)
    total_sum = None
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        with torch.amp.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
            chunk_logits = flat_hidden[i:j] @ lm_head_weight.t()  # (c, V)
            chunk_loss = F.cross_entropy(chunk_logits, flat_labels[i:j],
                                          reduction="sum")
        total_sum = chunk_loss if total_sum is None else total_sum + chunk_loss
    if reduction == "mean":
        return total_sum / N
    return total_sum


def evaluate(backbone, lm_head_weight, val_sequences, device, bf16,
              batch_size, chunk_size=4096):
    backbone.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i + batch_size]
            batch_t = torch.from_numpy(batch.astype(np.int64)).to(device)
            input_ids = batch_t[:, :-1]
            labels = batch_t[:, 1:]
            loss_sum = chunked_loss(backbone, lm_head_weight, input_ids, labels,
                                     bf16=bf16, chunk_size=chunk_size,
                                     reduction="sum")
            total_loss += loss_sum.item()
            total_n += labels.numel()
    backbone.train()
    return float(total_loss / total_n)


def train_and_eval(packed_train_sequences: np.ndarray,
                   val_sequences: np.ndarray,
                   cfg: TrainConfig,
                   device: torch.device,
                   log_callback=None,
                   wandb_run=None) -> dict:
    """packed_train_sequences: (N, seq_len + 1) uint32 array of token ids.
    val_sequences:           (M, seq_len + 1) uint32 array.
    Returns dict with val_loss and training metadata.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    bf16 = cfg.bf16 and _supports_bf16()
    model = build_model(cfg.vocab_size, preset=cfg.model_preset).to(device)
    print(f"  attn_impl: {getattr(model.config, '_attn_implementation', '?')}")

    # Split lm_head from the backbone so we can use a chunked CE.
    # The lm_head weight is tied to the input embedding, so grads accumulate
    # to the shared parameter correctly when we use lm_head.weight directly.
    backbone = model.model           # LlamaModel — outputs hidden states
    lm_head_weight = model.lm_head.weight  # (V, H), tied to embed_tokens.weight
    compiled = False
    if cfg.compile_model:
        try:
            backbone = torch.compile(backbone, mode="default", fullgraph=False)
            compiled = True
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )

    assert cfg.batch_size % cfg.micro_batch_size == 0, "batch must be divisible by micro"
    accum_steps = cfg.batch_size // cfg.micro_batch_size
    tokens_per_step = cfg.batch_size * cfg.seq_len
    max_steps = cfg.total_tokens // tokens_per_step
    warmup_steps = int(cfg.warmup_frac * max_steps)
    print(f"  model params: {n_params/1e6:.2f}M, bf16={bf16}")
    print(f"  effective batch {cfg.batch_size} = micro {cfg.micro_batch_size} x accum {accum_steps}")
    print(f"  steps: {max_steps}, warmup: {warmup_steps}, tokens/step: {tokens_per_step}")
    print(f"  packed train seqs available: {len(packed_train_sequences)}")

    n_seqs = len(packed_train_sequences)
    # uint32 -> int32 view: zero-copy, vocab fits in positive int32 range.
    if packed_train_sequences.dtype == np.uint32:
        seqs_i32 = packed_train_sequences.view(np.int32)
    else:
        seqs_i32 = packed_train_sequences.astype(np.int32, copy=False)
    perm = np.random.permutation(n_seqs)
    seq_cursor = 0

    def _fetch(idx):
        # int32 on CPU -> int32 on GPU -> .long() on GPU (cheap)
        batch_np = np.ascontiguousarray(seqs_i32[idx])
        t = torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True).long()
        return t

    t0 = time.time()
    last_log_t = t0
    loss_tensors = []  # accumulate detached losses (one per step), sync only at log

    # Prefetch first microbatch
    if seq_cursor + cfg.micro_batch_size > n_seqs:
        perm = np.random.permutation(n_seqs); seq_cursor = 0
    next_batch = _fetch(perm[seq_cursor:seq_cursor + cfg.micro_batch_size])
    seq_cursor += cfg.micro_batch_size

    for step in range(max_steps):
        lr = _cosine_lr(step, max_steps, cfg.lr, warmup_steps)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for micro in range(accum_steps):
            batch_t = next_batch
            # prefetch next while compute runs
            if step + 1 < max_steps or micro + 1 < accum_steps:
                if seq_cursor + cfg.micro_batch_size > n_seqs:
                    perm = np.random.permutation(n_seqs); seq_cursor = 0
                next_batch = _fetch(perm[seq_cursor:seq_cursor + cfg.micro_batch_size])
                seq_cursor += cfg.micro_batch_size
            input_ids = batch_t[:, :-1]
            labels = batch_t[:, 1:]
            loss = chunked_loss(backbone, lm_head_weight, input_ids, labels,
                                 bf16=bf16, chunk_size=cfg.ce_chunk_size,
                                 reduction="mean") / accum_steps
            loss.backward()
            step_loss = step_loss + loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        loss_tensors.append(step_loss)
        if (step + 1) % cfg.log_every == 0 or step == 0:
            now = time.time()
            # ONE sync point: stack and move to CPU at log boundary
            recent_vals = torch.stack(loss_tensors[-cfg.log_every:]).float().cpu().numpy()
            recent = float(recent_vals.mean())
            toks_per_s = (cfg.log_every * tokens_per_step) / max(now - last_log_t, 1e-9)
            print(f"  step {step+1:5d}/{max_steps}  loss={recent:.4f}  "
                  f"lr={lr:.2e}  toks/s={toks_per_s/1e3:.1f}k", flush=True)
            last_log_t = now
            if log_callback:
                log_callback(step + 1, recent, lr)
            if wandb_run is not None:
                wandb_run.log({"train_loss": recent, "lr": lr,
                                "toks_per_s": toks_per_s, "step": step + 1})

    train_time = time.time() - t0
    print(f"  training done in {train_time:.1f}s, evaluating...")
    # Sync final losses
    all_losses = torch.stack(loss_tensors).float().cpu().numpy() if loss_tensors else np.array([])
    final_train_loss = float(all_losses[-100:].mean()) if len(all_losses) else None
    val_loss = evaluate(backbone, lm_head_weight, val_sequences, device, bf16,
                         batch_size=cfg.eval_micro_batch,
                         chunk_size=cfg.ce_chunk_size)
    print(f"  val_loss = {val_loss:.4f}")
    if wandb_run is not None:
        wandb_run.summary["val_loss"] = float(val_loss)
        wandb_run.summary["train_time_s"] = float(train_time)
        wandb_run.summary["final_train_loss"] = final_train_loss
        wandb_run.summary["compiled"] = compiled

    if cfg.save_checkpoint:
        ckpt_path = cfg.save_checkpoint
        import os
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        # Save the original (uncompiled) model state_dict in bf16.
        # 'model' is the LlamaForCausalLM wrapper around `backbone`.
        sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": sd, "preset": cfg.model_preset,
                     "vocab_size": cfg.vocab_size, "val_loss": val_loss},
                    ckpt_path)
        print(f"  saved checkpoint to {ckpt_path}  ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)")

    bench_result: dict = {}
    if cfg.run_bench:
        print("  running downstream benchmarks...")
        from transformers import AutoTokenizer
        from bench_eval import eval_benchmarks
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        t_b = time.time()
        bench_result = eval_benchmarks(backbone, lm_head_weight, tok, device,
                                        bf16=bf16, subset=cfg.bench_subset)
        print(f"  benchmarks done in {time.time()-t_b:.1f}s")
        if wandb_run is not None:
            for k, v in bench_result.items():
                wandb_run.summary[k] = v

    return {
        "val_loss": val_loss,
        "n_params": int(n_params),
        "train_time_s": train_time,
        "steps": max_steps,
        "final_train_loss": final_train_loss,
        "bf16": bf16,
        "compiled": compiled,
        **bench_result,
    }
