#!/bin/bash
#SBATCH --job-name=noise-floor
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# GPU correctness check: does no_draft reproduce the direct greedy reference?
# Submit from the repo root:  mkdir -p slurm/logs && sbatch scripts/noise_floor.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
POLICIES="${POLICIES:-no_draft}"
DTYPE="${DTYPE:-bfloat16}"
LOADS="${LOADS:-light,heavy}"
TARGET_TURNS="${TARGET_TURNS:-16}"
WARMUP_TURNS="${WARMUP_TURNS:-4}"
OUT_DIR="${OUT_DIR:-results/noise_floor/run-${SLURM_JOB_ID}-${DTYPE}}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${SCRATCH_ROOT}/.cache/uv"
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache/xdg"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  dtype: ${DTYPE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

set +e
uv run --no-sync python scripts/run_experiment.py \
  --mode vllm \
  --model_name microsoft/Phi-4-mini-instruct \
  --dtype "${DTYPE}" \
  --split dev \
  --policies "${POLICIES}" \
  --loads "${LOADS}" \
  --seeds 1 \
  --target_turns "${TARGET_TURNS}" \
  --warmup_turns "${WARMUP_TURNS}" \
  --no_timeline \
  --output_dir "${OUT_DIR}"
STATUS=$?
set -e

echo
echo "Runner exit code: ${STATUS} (0 = validated, nonzero = mismatches or crash)"
if [ -f "${OUT_DIR}/validation_vllm_dev.json" ]; then
  uv run --no-sync python - "${OUT_DIR}/validation_vllm_dev.json" <<'EOF'
import json, sys
from collections import Counter
v = json.load(open(sys.argv[1]))
print("Reference status:", v.get("greedy_reference_status"))
print("Turns checked:", v.get("reference_checked_turns"))
bad = Counter((m["load"], m["policy"]) for m in v.get("reference_mismatches", []))
print("Mismatches by (load, policy):", dict(bad) or "none")
print("Policies disagreeing with no_draft:", len(v.get("output_mismatches", [])))
print("Warmup / accounting errors:", len(v.get("warmup_mismatches", [])), "/", len(v.get("accounting_errors", [])))
EOF
fi
echo "Artifacts in: ${OUT_DIR}"