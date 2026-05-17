#!/bin/bash
# Launch the pilot 100 diagnostic runs as a job array.
# After they complete and look healthy, launch the remaining 700 with launch_full.sh.
sbatch --array=0-99%35 milestone4/m4/scripts/slurm_diagnostic.sh
