#!/bin/bash
# Generate all 8 paper figures.
set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
.venv/bin/python3 milestone4/m4/scripts/10_make_figures.py
.venv/bin/python3 milestone4/m4/scripts/12_wandb_export.py
echo "Figures in milestone4/m4/data/figures/"
echo "EDA figures in milestone4/m4/data/figures_eda/"
echo "wandb dashboards:"
echo "  https://wandb.ai/kaitwang-stanford-university/m4-diagnostic"
echo "  https://wandb.ai/kaitwang-stanford-university/m4-validation"
