# Milestone 3 Experiments

This folder is separate from the stable `milestone3/` pipeline. It explores a
stronger model-and-decision story for Milestone 3:

- infer a shared latent document-quality score from multiple quality filters;
- compare latent-quality policies against DCLM and the existing hierarchical
  regression policy;
- produce quality-diversity frontiers for actionable data-mix decisions.

Generated outputs are local-only:

- `milestone3_experiments/artifacts/`
- `milestone3_experiments/figures/`
- `milestone3_experiments/reports/`

The stable `week6.tex` and `week6.pdf` remain untouched unless a v2 report
clearly improves the submission.

## Run Order

Run from the repository root after the stable `milestone3/` artifacts exist:

```bash
uv run python milestone3_experiments/01_fit_latent_quality_em.py --iterations 200 --overwrite
uv run python milestone3_experiments/02_fit_factor_quality_em.py --iterations 300 --overwrite
uv run python milestone3_experiments/03_evaluate_latent_policies.py --overwrite
uv run python milestone3_experiments/04_quality_diversity_frontier.py --overwrite
uv run python milestone3_experiments/05_policy_robustness.py --overwrite
uv run python milestone3_experiments/06_make_experiment_figures.py
```

Then compile the v2 report:

```bash
cd Stats305C_Project_AngikarGhosal_KaitlynWang
pdflatex -interaction=nonstopmode week6_v2.tex
pdflatex -interaction=nonstopmode week6_v2.tex
```

## Main Experimental Result

The two-factor EM model discovers:

- factor 1: ARC/PIQA/SciQ benchmark axis;
- factor 2: DCLM/FineWeb-Edu/LAMBADA axis.

The top-20% two-factor posterior policy reaches held-out composite quality about
0.680, compared with 0.405 for global DCLM and 0.419 for the previous
hierarchical topic-floor policy. A 20% within-topic floor preserves topic
entropy around 0.964 while still reaching quality about 0.638.
