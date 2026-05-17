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
    seq_len: int = 2048
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
    compile_model: bool = True
    bf16: bool = True             # auto-disabled if unsupported


def build_model(vocab_size: int) -> LlamaForCausalLM:
    # Hidden=192, inter=576, 6 heads, 12 layers => ~15M params (embed dominates).
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except Exception:
        pass
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=192,
        intermediate_size=576,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_key_value_heads=6,
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


def _chunked_ce_sum(logits, labels, chunk=128):
    """Compute sum cross-entropy without materializing huge logits gradients.
    logits: (B, T, V) (possibly bf16), labels: (B, T) int64.
    """
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    total = 0.0
    for i in range(0, flat_logits.size(0), chunk * logits.size(1)):
        # We chunk by rows of (B*T)
        j = min(i + chunk * logits.size(1), flat_logits.size(0))
        l = F.cross_entropy(flat_logits[i:j].float(), flat_labels[i:j], reduction="sum")
        total = total + l.item() if isinstance(total, float) else total + l.item()
    return total


def evaluate(model, val_sequences, device, bf16, batch_size):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i + batch_size]
            batch_t = torch.from_numpy(batch.astype(np.int64)).to(device)
            input_ids = batch_t[:, :-1]
            labels = batch_t[:, 1:]
            with torch.amp.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
            total_loss += loss.item()
            total_n += labels.numel()
    model.train()
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
    model = build_model(cfg.vocab_size).to(device)
    print(f"  attn_impl: {getattr(model.config, '_attn_implementation', '?')}")
    compiled = False
    if cfg.compile_model:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
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
            with torch.amp.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                ) / accum_steps
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
    val_loss = evaluate(model, val_sequences, device, bf16,
                         batch_size=cfg.eval_micro_batch)
    print(f"  val_loss = {val_loss:.4f}")
    if wandb_run is not None:
        wandb_run.summary["val_loss"] = float(val_loss)
        wandb_run.summary["train_time_s"] = float(train_time)
        wandb_run.summary["final_train_loss"] = final_train_loss
        wandb_run.summary["compiled"] = compiled

    return {
        "val_loss": val_loss,
        "n_params": int(n_params),
        "train_time_s": train_time,
        "steps": max_steps,
        "final_train_loss": final_train_loss,
        "bf16": bf16,
        "compiled": compiled,
    }
