#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=conv-ai-rtx-diag
#SBATCH --output=results/pace_setup/rtx-diag-%j.out
#SBATCH --error=results/pace_setup/rtx-diag-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source results/pace_setup/activate_rtx.sh
export XDG_CACHE_HOME="$HOME/scratch/.cache/rtx_6000"
.venv/bin/python results/pace_setup/diagnose_rtx.py
