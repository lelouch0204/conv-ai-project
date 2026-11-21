#!/bin/bash
#SBATCH --job-name=early-draft-pilot
#SBATCH --partition=ice-gpu
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:25:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_DIR}"
if [[ ! -f pyproject.toml || ! -f scripts/run_experiment.py ]]; then
  echo "Submit from the repository root or set PROJECT_DIR to the checkout." >&2
  exit 2
fi

GPU_TAG="${GPU_TAG:-rtx_6000}"
DTYPE="${DTYPE:-float16}"
MODEL_NAME="${MODEL_NAME:-microsoft/Phi-4-mini-instruct}"
MODEL_ARGS=(--model_name "${MODEL_NAME}")
if [[ -n "${MODEL_REVISION:-}" ]]; then
  MODEL_ARGS+=(--model_revision "${MODEL_REVISION}")
fi
LOADS="${LOADS:-light,intermediate,heavy}"
TARGET_TURNS="${TARGET_TURNS:-16}"
WARMUP_TURNS="${WARMUP_TURNS:-4}"
POLICIES="${POLICIES:-no_draft,fixed_interval,load_reuse_adaptive,reuse_only}"
SEEDS="${SEEDS:-1}"
OUT_DIR="${OUT_DIR:-results/${GPU_TAG}/run-${SLURM_JOB_ID:-local}}"

mkdir -p results "${OUT_DIR}"

PACE_CACHE_ROOT="${PACE_CACHE_ROOT:-${HOME}/scratch/.cache}"
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PACE_CACHE_ROOT}/${GPU_TAG}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PACE_CACHE_ROOT}/uv}"
export HF_HOME="${HF_HOME:-${PACE_CACHE_ROOT}/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${HF_HOME}"

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --no-sync python scripts/run_experiment.py \
  --mode vllm \
  "${MODEL_ARGS[@]}" \
  --dtype "${DTYPE}" \
  --split dev \
  --policies "${POLICIES}" \
  --loads "${LOADS}" \
  --seeds "${SEEDS}" \
  --target_turns "${TARGET_TURNS}" \
  --warmup_turns "${WARMUP_TURNS}" \
  --output_dir "${OUT_DIR}"

uv run --no-sync python scripts/visualize_run.py "${OUT_DIR}/turns_vllm_dev.jsonl" --ascii
