#!/bin/bash
# Spy-Bagel OCR Training Script
# Usage: bash scripts/train_spy_ocr.sh [NUM_GPUS]
#   NUM_GPUS: number of GPUs to use (default: 4)

set -euo pipefail

NUM_GPUS=${1:-4}

# Ensure all output goes under /adialab/usr/shadabk/
export HF_HOME=/adialab/usr/shadabk/.cache/huggingface
export WANDB_DIR=/adialab/usr/shadabk/MedUMM/flow_grpo/logs
export WANDB_CACHE_DIR=/adialab/usr/shadabk/.cache/wandb
export TMPDIR=/adialab/usr/shadabk/.tmp
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

cd /adialab/usr/shadabk/MedUMM/flow_grpo

echo "=== Spy-Bagel OCR Training ==="
echo "  GPUs: ${NUM_GPUS}"
echo "  Config: spy_game_bagel_ocr"
echo "  Dataset: /adialab/usr/shadabk/MedUMM/flow_grpo/dataset/ocr"
echo "  Logs: /adialab/usr/shadabk/MedUMM/flow_grpo/logs/spy_game/bagel_ocr"
echo ""

accelerate launch \
    --num_processes ${NUM_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    scripts/train_bagel_spy.py \
    --config config/grpo.py:spy_game_bagel_ocr
