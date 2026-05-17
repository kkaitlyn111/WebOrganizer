#!/bin/bash
# Unsupervised overnight pipeline: submit diagnostic array, wait, then
# run Phases 2 -> 3 -> 4 -> 6 (figures). Logs everything to overnight.log.
#
# Usage:
#   cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
#   nohup bash milestone4/m4/scripts/run_overnight.sh > milestone4/m4/logs/overnight.log 2>&1 &
#   disown   # so it survives logout
#
# Monitor:
#   tail -f milestone4/m4/logs/overnight.log
#   squeue -u kaitwang
#   wandb.ai/kaitwang-stanford-university/m4-diagnostic   (phone-friendly)
set -uo pipefail
cd /juice5b/scr5b/kaitwang/stats305c/WebOrganizer
mkdir -p milestone4/m4/logs

ts() { date +"[%F %T]"; }
log() { echo "$(ts) $*"; }

TRIMMED=milestone4/m4/data/diagnostic_run_configs_trimmed.json
if [ ! -s "$TRIMMED" ]; then
  log "ERROR: $TRIMMED is missing or empty. Run 04b_trim_configs.py first."
  exit 1
fi
N=$(.venv/bin/python3 -c "import json; print(len(json.load(open('$TRIMMED'))))")
# Reuse existing array if one is queued/running, else submit a new one.
EXISTING=$(squeue -u kaitwang -h -n m4_diag -o "%i" | grep -oE "^[0-9]+" | head -1)
if [ -n "$EXISTING" ]; then
  JOB="$EXISTING"
  log "Phase 1: reusing existing array $JOB ($N configs)"
else
  log "Phase 1: launching diagnostic array for $N configs"
  JOB=$(sbatch --array=0-$((N-1))%35 milestone4/m4/scripts/slurm_diagnostic.sh \
         | awk '{print $4}')
  log "Phase 1 array job id: $JOB"
fi

# Wait for the entire array to finish. We poll squeue every 60s.
log "Waiting for array $JOB to complete..."
while squeue -j "$JOB" -h 2>/dev/null | grep -q "."; do
  REMAINING=$(squeue -j "$JOB" -h 2>/dev/null | wc -l)
  log "  $REMAINING tasks still queued/running"
  sleep 120
done
log "Phase 1: array done"

DONE_N=$(ls milestone4/m4/data/diagnostic_results/run_*.json 2>/dev/null | wc -l)
log "Phase 1: $DONE_N results files on disk"

log "Phase 2: build doc-level quality targets"
.venv/bin/python3 milestone4/m4/scripts/05_build_targets.py \
    --configs milestone4/m4/data/diagnostic_run_configs_trimmed.json \
   || { log "Phase 2 FAILED"; exit 2; }

log "Phase 3: train Lasso + EBM + EBM-no-PC"
.venv/bin/python3 milestone4/m4/scripts/06_train_ebm.py \
   || { log "Phase 3 FAILED"; exit 3; }

log "Phase 4 prep: fresh shards + scoring"
.venv/bin/python3 milestone4/m4/scripts/09_validation_prepare.py \
   || { log "Phase 4 prep FAILED"; exit 4; }

log "Phase 4: launching 10 validation runs"
VJOB=$(sbatch --array=0-9 milestone4/m4/scripts/slurm_validation.sh 2>/dev/null \
        | awk '{print $4}')
if [ -n "$VJOB" ]; then
  log "Phase 4 array job id: $VJOB"
  while squeue -j "$VJOB" -h 2>/dev/null | grep -q "."; do
    sleep 60
  done
  log "Phase 4: array done"
else
  log "WARN: slurm_validation.sh not present; running validations serially"
  for v in $(seq 0 9); do
    .venv/bin/python3 milestone4/m4/scripts/09_run_validation.py --v-id "$v" || true
  done
fi

log "Phase 6: figures + wandb export"
bash milestone4/m4/scripts/RUN_PHASE_6_FIGURES.sh || log "Phase 6 had issues"

log "ALL PHASES DONE. See milestone4/m4/data/figures/ + figures_eda/"
log "wandb: https://wandb.ai/kaitwang-stanford-university/m4-diagnostic"
