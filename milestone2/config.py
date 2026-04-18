from __future__ import annotations
from pathlib import Path

SEED = 42
N_SHARDS = 100
TOTAL_SHARDS = 9888

REPO_ROOT = Path("/juice5b/scr5b/kaitwang/stats305c/WebOrganizer")
CORPUS_ROOT = REPO_ROOT / "Corpus-200B"
M2_ROOT = REPO_ROOT / "milestone2"
DATA_ROOT = M2_ROOT / "data"
FEATURES_DIR = DATA_ROOT / "features"

SAMPLED_SHARDS_JSON = DATA_ROOT / "sampled_shards.json"
ANNOTATIONS_PARQUET = DATA_ROOT / "annotations.parquet"
MASTER_PARQUET = DATA_ROOT / "master.parquet"

# Per-subdir filename conventions. Each entry maps a logical "key" to
# (subdir_name, suffix). Files are named "{stem}{suffix}.npy" where
# stem = CC_shard_NNNNNNNN_processed.
ANNOT_SPECS = {
    "tokens":        ("tokens",                       ""),
    "dclm_score":    ("scores_dclm-fasttext",         ""),
    "fwedu_score":   ("scores_fineweb-edu",           ""),
    "fwedu_rounded": ("scores_fineweb-edu__rounded",  "__rounded"),
    "topic":         ("domains_topics",               "__choice"),
    "topic_logits":  ("domains_topics__logits",       "__logits"),
    "format":        ("domains_formats",              "__choice"),
    "format_logits": ("domains_formats__logits",      "__logits"),
    "cluster24":     ("domains_clusters-k24",         ""),
}

DOC_SUBDIR = "documents"
DOC_SUFFIX = ".jsonl.zst"


def shard_stem(shard_idx: int) -> str:
    return f"CC_shard_{shard_idx:08d}_processed"


def annot_path(shard_idx: int, key: str) -> Path:
    subdir, suffix = ANNOT_SPECS[key]
    return CORPUS_ROOT / subdir / f"{shard_stem(shard_idx)}{suffix}.npy"


def doc_path(shard_idx: int) -> Path:
    return CORPUS_ROOT / DOC_SUBDIR / f"{shard_stem(shard_idx)}{DOC_SUFFIX}"
