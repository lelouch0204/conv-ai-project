#!/bin/bash
#SBATCH --job-name=early-draft-pilot
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00

set -euo pipefail

cd /home/hice1/nparikh44/scratch/conv-ai-project
mkdir -p results

# Sync dependencies first if not already synced (uv sync)
export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/run_experiment.py \
  --mode vllm \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --split dev \
  --target_turns 48 \
  --seeds 1,2,3 \
  --output_dir results
