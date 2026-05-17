#!/bin/bash
# Launch remaining diagnostic runs 100-799 after pilot looks healthy.
# Optionally inspect pilot results first.

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

# Quick pilot health check
N_DONE=$(ls milestone4/m4/data/diagnostic_results/run_*.json 2>/dev/null | wc -l)
echo "Pilot runs with results: $N_DONE / 100"

if [ "$N_DONE" -lt 50 ]; then
  echo "Pilot has fewer than 50 results; you may want to wait longer or investigate."
fi

JOB=$(sbatch --array=100-799%35 milestone4/m4/scripts/slurm_diagnostic.sh | awk '{print $4}')
echo "Full array job id: $JOB"
echo "Monitor with: squeue -u kaitwang --name=m4_diag"
echo
echo "When everything finishes (~6-10 hr):"
echo "  bash milestone4/m4/scripts/RUN_PHASE_2_3.sh"
