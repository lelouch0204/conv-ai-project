# Portable activation for the collected PACE-ICE RTX 6000 float32 configuration.
module load python/3.12.5
export PATH="$HOME/scratch/.tools/conv-ai-uv/bin:$PATH"
export UV_CACHE_DIR="$HOME/scratch/.cache/uv"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_BATCH_INVARIANT=0
export MODEL_REVISION=cfbefacb99257ffa30c83adab238a50856ac3083
export GPU_TAG=rtx_6000_float32
export XDG_CACHE_HOME="$HOME/scratch/.cache/rtx_6000_float32"
export DTYPE=float32
