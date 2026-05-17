"""Shared paths for Milestone 4 experiments."""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
M4_ROOT = REPO_ROOT / "milestone4" / "m4"
M2_DATA = REPO_ROOT / "milestone2" / "data"

# Outputs
DATA_DIR = M4_ROOT / "data"
TOKENS_DIR = DATA_DIR / "tokens"
MASKS_DIR = DATA_DIR / "selection_masks"
RESULTS_DIR = DATA_DIR / "diagnostic_results"
LOGS_DIR = M4_ROOT / "logs"
CONFIGS_DIR = DATA_DIR / "configs"
MODELS_DIR = DATA_DIR / "models"

MASTER_PARQUET = DATA_DIR / "master.parquet"
VAL_SHARDS_JSON = DATA_DIR / "val_shard_indices.json"
DIAGNOSTIC_CONFIGS_JSON = DATA_DIR / "diagnostic_run_configs.json"

# Pretokenization outputs
TRAIN_TOKEN_IDS = TOKENS_DIR / "train_pool_token_ids.bin"
TRAIN_DOC_OFFSETS = TOKENS_DIR / "train_pool_doc_offsets.npy"
TRAIN_DOC_INDEX = TOKENS_DIR / "train_pool_doc_index.npy"  # global doc idx per token-row
VAL_TOKEN_IDS = TOKENS_DIR / "val_token_ids.bin"
VAL_DOC_OFFSETS = TOKENS_DIR / "val_doc_offsets.npy"
VAL_SEQUENCES = TOKENS_DIR / "val_sequences_1024.npy"
VAL_SEQUENCES_EVAL = TOKENS_DIR / "val_sequences_eval_1024.npy"  # subsample for fast eval
TOKENS_META = TOKENS_DIR / "tokens_meta.json"

# Per-shard pretokenized files
PERSHARD_TOKENS_DIR = TOKENS_DIR / "pershard"

# Constants
SEED = 42
SEQ_LEN = 1024
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
N_LM_VAL_SHARDS = 10

# Add milestone2 to path
sys.path.insert(0, str(REPO_ROOT / "milestone2"))

for d in (DATA_DIR, TOKENS_DIR, MASKS_DIR, RESULTS_DIR, LOGS_DIR, CONFIGS_DIR,
          MODELS_DIR, PERSHARD_TOKENS_DIR):
    d.mkdir(parents=True, exist_ok=True)
