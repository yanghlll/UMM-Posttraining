#!/bin/bash
# Spy-Bagel FSDP Training Script (SHARD_GRAD_OP / ZeRO-2)
# Usage: bash scripts/train_spy_fsdp.sh [CONFIG_NAME]
#   CONFIG_NAME: config variant (default: spy_game_bagel_pickscore)
#   For debug: bash scripts/train_spy_fsdp.sh spy_game_bagel_debug

set -euo pipefail

# Activate spy_bagel conda environment (PyTorch 2.9.1)
eval "$(conda shell.bash hook)"
conda activate spy_bagel

CONFIG_NAME=${1:-spy_game_bagel_pickscore}

# Ensure all output goes under /adialab/usr/shadabk/
export HF_HOME=/adialab/usr/shadabk/.cache/huggingface
export WANDB_DIR=/adialab/usr/shadabk/MedUMM/flow_grpo/logs
export WANDB_CACHE_DIR=/adialab/usr/shadabk/.cache/wandb
export TMPDIR=/adialab/usr/shadabk/.tmp
export TRITON_CACHE_DIR=/adialab/usr/shadabk/MedUMM/.triton
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR"

cd /adialab/usr/shadabk/MedUMM/flow_grpo
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/flow_grpo:${PYTHONPATH:-}"

echo "=== Spy-Bagel FSDP Training ==="
echo "  Config: ${CONFIG_NAME}"
echo "  FSDP: SHARD_GRAD_OP (4 GPUs)"
echo "  Accelerate config: scripts/accelerate_configs/fsdp_4gpu.yaml"
echo ""

accelerate launch \
    --config_file scripts/accelerate_configs/fsdp_4gpu.yaml \
    scripts/train_bagel_spy.py \
    --config config/grpo.py:${CONFIG_NAME}
