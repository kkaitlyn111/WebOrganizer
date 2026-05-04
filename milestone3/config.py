from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
M3_ROOT = REPO_ROOT / "milestone3"
DATA_ROOT = REPO_ROOT / "data"
ARTIFACTS_DIR = M3_ROOT / "artifacts"

SAMPLED_SHARDS_JSON = DATA_ROOT / "sampled_shards.json"
ANNOTATIONS_PARQUET = DATA_ROOT / "annotations.parquet"
FEATURES_DIR = DATA_ROOT / "features"
PC_SCORES_DIR = DATA_ROOT / "scores_pc"
QURATER_DIR = DATA_ROOT / "scores_qurater"

MODEL_FRAME_PARQUET = ARTIFACTS_DIR / "model_frame.parquet"
MODEL_FRAME_SUMMARY_JSON = ARTIFACTS_DIR / "model_frame_summary.json"
PREPARED_MODEL_NPZ = ARTIFACTS_DIR / "prepared_model_data.npz"
PREPARED_MODEL_SUMMARY_JSON = ARTIFACTS_DIR / "prepared_model_data_summary.json"
PREPARED_SAMPLE_METADATA_PARQUET = ARTIFACTS_DIR / "prepared_sample_metadata.parquet"
BASELINE_RESULTS_JSON = ARTIFACTS_DIR / "baseline_results.json"
BASELINE_PREDICTIONS_NPZ = ARTIFACTS_DIR / "baseline_predictions.npz"
GIBBS_RESULTS_JSON = ARTIFACTS_DIR / "gibbs_results.json"
GIBBS_DRAWS_NPZ = ARTIFACTS_DIR / "gibbs_draws.npz"

PC_MODELS = ["arc_easy", "piqa", "sciq", "lambada", "lambada_es"]
MODEL_SCORE_COLUMNS = [
    "dclm_score",
    "fwedu_rounded",
    "pc_arc_easy",
    "pc_piqa",
    "pc_sciq",
    "pc_lambada",
]
PROBABILITY_SCORE_COLUMNS = [
    "dclm_score",
    "pc_arc_easy",
    "pc_piqa",
    "pc_sciq",
    "pc_lambada",
]
QURATER_AXES = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
]

HEURISTIC_COLS = [
    "char_count",
    "word_count",
    "mean_word_length",
    "frac_alpha",
    "frac_digit",
    "frac_punctuation",
    "frac_uppercase",
    "frac_lines_terminal_punct",
    "frac_lines_bullet",
    "type_token_ratio",
    "stopword_fraction",
    "ngram_rep_2",
    "ngram_rep_3",
    "ngram_rep_4",
    "num_lines",
    "mean_line_length",
]

URL_COLS = ["url", "url_netloc", "url_registered_domain"]
LOG1P_FEATURE_COLS = [
    "token_count",
    "char_count",
    "word_count",
    "num_lines",
    "mean_line_length",
]


def shard_stem(shard_idx: int) -> str:
    return f"CC_shard_{shard_idx:08d}_processed"


def parse_shard_idx_from_stem(stem: str) -> int:
    return int(stem.split("_")[2])
