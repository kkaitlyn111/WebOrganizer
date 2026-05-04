# Milestone 3: Model and Inference

This directory contains the reproducible Milestone 3 pipeline. It assumes Kaitlyn's local
`data/` folder is present at the repository root. Raw data and generated artifacts are
ignored by git.

## Pipeline

Run from the repository root:

```bash
uv run python milestone3/01_build_model_frame.py --overwrite
uv run python milestone3/02_prepare_model_data.py --overwrite
uv run python milestone3/03_fit_baselines.py --overwrite
uv run python milestone3/04_fit_hierarchical_gibbs.py --iterations 300 --burn 150 --thin 3 --chains 4 --overwrite
uv run python milestone3/05_summarize_posterior.py --overwrite
uv run python milestone3/06_data_mix_policy.py --overwrite
```

Then compile the report:

```bash
cd Stats305C_Project_AngikarGhosal_KaitlynWang
pdflatex -interaction=nonstopmode week6.tex
pdflatex -interaction=nonstopmode week6.tex
```

## Main Outputs

- `milestone3/artifacts/model_frame.parquet`: joined annotations, features, PC scores, and partial QuRater scores.
- `milestone3/artifacts/prepared_model_data.npz`: transformed model matrices and train/test split.
- `milestone3/artifacts/baseline_results.json`: fixed-effect baseline results.
- `milestone3/artifacts/gibbs_results.json`: hierarchical Gibbs sampler metrics, diagnostics, posterior summaries.
- `milestone3/artifacts/data_mix_results.json`: model-based data-mix policy evaluation.
- `Stats305C_Project_AngikarGhosal_KaitlynWang/week6.pdf`: two-page Milestone 3 report.

## Modeling Scope

The main model uses the six complete scores:

- DCLM
- rounded FineWeb-Edu
- PC ARC-Easy
- PC PIQA
- PC SciQ
- PC LAMBADA

QuRater is not included in the main likelihood because it only covers 44 of the 100 sampled shards.
