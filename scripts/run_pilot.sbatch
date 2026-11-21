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

cd /home/hice1/nparikh44/scratch/conv-ai-project

GPU_TAG="${GPU_TAG:-rtx_6000}"
DTYPE="${DTYPE:-float16}"
LOADS="${LOADS:-light,intermediate,heavy}"
TARGET_TURNS="${TARGET_TURNS:-16}"
WARMUP_TURNS="${WARMUP_TURNS:-4}"
OUT_DIR="${OUT_DIR:-results/${GPU_TAG}}"

mkdir -p results "${OUT_DIR}"

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="/home/hice1/nparikh44/scratch/.cache/${GPU_TAG}"
export UV_CACHE_DIR=/home/hice1/nparikh44/scratch/.cache/uv
export HF_HOME=/home/hice1/nparikh44/scratch/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --no-sync python scripts/run_experiment.py \
  --mode vllm \
  --model_name microsoft/Phi-4-mini-instruct \
  --dtype "${DTYPE}" \
  --split dev \
  --policies no_draft,fixed_interval \
  --loads "${LOADS}" \
  --seeds 1 \
  --target_turns "${TARGET_TURNS}" \
  --warmup_turns "${WARMUP_TURNS}" \
  --output_dir "${OUT_DIR}"

uv run --no-sync python scripts/visualize_run.py "${OUT_DIR}/turns_vllm_dev.jsonl" --ascii
