#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=conv-ai-rtx-probe
#SBATCH --output=results/pace_setup/rtx-probe-%j.out
#SBATCH --error=results/pace_setup/rtx-probe-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.12.5
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
.venv/bin/python - <<'PY'
import torch
print('torch',torch.__version__,'CUDA',torch.version.cuda,flush=True)
print('compiled arches',torch.cuda.get_arch_list(),flush=True)
print('device',torch.cuda.get_device_name(),'capability',torch.cuda.get_device_capability(),flush=True)
a=torch.ones((32,32),device='cuda',dtype=torch.float16)
print('float16 matmul',(a@a)[0,0].item(),flush=True)
PY
