#!/bin/bash
# Run after entities catchup finishes (100/100).
# 1. Builds full master.parquet (100 shards)
# 2. Generates 800 diagnostic run configs
# 3. Launches the pilot 100 Slurm array

set -euo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer

ENT_N=$(ls milestone2/data/entities | wc -l)
if [ "$ENT_N" -lt 100 ]; then
  echo "ERROR: entities not done yet ($ENT_N/100). Wait for catchup."
  exit 1
fi

echo "[1/4] Building master.parquet ..."
.venv/bin/python3 milestone4/m4/scripts/03_build_master.py --overwrite

echo "[2/4] Feature EDA (histograms, correlations, topic/format, Zipf, stats) ..."
.venv/bin/python3 milestone4/m4/scripts/11_feature_eda.py

echo "[3/4] Generating 800 diagnostic configs ..."
.venv/bin/python3 milestone4/m4/scripts/04_generate_configs.py \
    --n-type1 200 --n-type2 300 --n-type3 200 --n-type5 50

echo "[4/4] Launching pilot Slurm array (run_ids 0-99, %35 concurrent) ..."
JOB=$(sbatch --array=0-99%35 milestone4/m4/scripts/slurm_diagnostic.sh | awk '{print $4}')
echo "Pilot array job id: $JOB"
echo "Monitor with: squeue -u kaitwang --name=m4_diag"
echo
echo "When pilot finishes (~1-2 hr):"
echo "  bash milestone4/m4/scripts/RUN_PHASE_1_FULL.sh"
