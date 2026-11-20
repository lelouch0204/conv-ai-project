#!/bin/bash
#SBATCH --job-name=early-draft-smoke
#SBATCH --partition=coe-gpu
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

cd /home/hice1/nparikh44/scratch/conv-ai-project
mkdir -p results

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME=/home/hice1/nparikh44/scratch/.cache
export UV_CACHE_DIR=/home/hice1/nparikh44/scratch/.cache/uv
export HF_HOME=/home/hice1/nparikh44/scratch/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

uv run --no-sync python scripts/run_experiment.py \
  --mode vllm \
  --model_name microsoft/Phi-4-mini-instruct \
  --split dev \
  --policies no_draft,fixed_interval \
  --loads light \
  --seeds 1 \
  --target_turns 8 \
  --warmup_turns 2 \
  --output_dir results
