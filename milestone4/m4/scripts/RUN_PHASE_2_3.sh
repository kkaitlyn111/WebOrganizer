#!/bin/bash
# Run Phase 2 (targets) + Phase 3 (EBM) once diagnostic runs are done.

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

N_DONE=$(ls milestone4/m4/data/diagnostic_results/run_*.json 2>/dev/null | wc -l)
echo "Diagnostic runs with results: $N_DONE / 800"

if [ "$N_DONE" -lt 200 ]; then
  echo "Too few diagnostic results for a robust target. Wait or investigate."
  exit 1
fi

echo "[1/2] Building document-level quality targets ..."
.venv/bin/python3 milestone4/m4/scripts/05_build_targets.py

echo "[2/2] Training Lasso + EBM + EBM-no-PC ..."
.venv/bin/python3 milestone4/m4/scripts/06_train_ebm.py

echo
echo "DONE phases 2-3. Next:"
echo "  bash milestone4/m4/scripts/RUN_PHASE_4.sh"
