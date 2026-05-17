#!/bin/bash
# Launch diagnostic runs 100-799 (assumes 0-99 already launched/done).
sbatch --array=100-799%35 milestone4/m4/scripts/slurm_diagnostic.sh
