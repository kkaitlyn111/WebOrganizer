from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / "milestone3_experiments"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"

STABLE_ARTIFACTS_DIR = REPO_ROOT / "milestone3" / "artifacts"
PREPARED_MODEL_NPZ = STABLE_ARTIFACTS_DIR / "prepared_model_data.npz"
GIBBS_DRAWS_NPZ = STABLE_ARTIFACTS_DIR / "gibbs_draws.npz"

LATENT_EM_RESULTS_JSON = ARTIFACTS_DIR / "latent_em_results.json"
LATENT_EM_DRAWS_NPZ = ARTIFACTS_DIR / "latent_em_outputs.npz"
FACTOR_EM_RESULTS_JSON = ARTIFACTS_DIR / "factor_em_results.json"
FACTOR_EM_OUTPUTS_NPZ = ARTIFACTS_DIR / "factor_em_outputs.npz"
LATENT_POLICY_RESULTS_JSON = ARTIFACTS_DIR / "latent_policy_results.json"
LATENT_POLICY_SELECTIONS_PARQUET = ARTIFACTS_DIR / "latent_policy_selections.parquet"
FRONTIER_RESULTS_JSON = ARTIFACTS_DIR / "quality_diversity_frontier.json"
ROBUSTNESS_RESULTS_JSON = ARTIFACTS_DIR / "policy_robustness.json"

REPORT_DIR = REPO_ROOT / "Stats305C_Project_AngikarGhosal_KaitlynWang"
