#!/bin/bash
# ──────────────────────────────────────────────────────
# Evaluate Bagel OCR Accuracy (8 GPU, single node)
# ──────────────────────────────────────────────────────
#
# Usage:
#   cd /adialab/usr/shadabk/MedUMM/flow_grpo
#   bash scripts/eval_bagel_ocr.sh
#
# To evaluate a LoRA checkpoint, add --lora_path:
#   Add to the python args below: --lora_path /path/to/lora/checkpoint
#
# To run on fewer GPUs:
#   Change --num_processes below (e.g., 1 for single GPU debug)
# ──────────────────────────────────────────────────────

set -e

NUM_GPUS=${NUM_GPUS:-8}
MODEL_PATH=${MODEL_PATH:-"ByteDance-Seed/BAGEL-7B-MoT"}
DATASET=${DATASET:-"dataset/ocr"}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_output/ocr_bagel"}
NUM_STEPS=${NUM_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.0}
RESOLUTION=${RESOLUTION:-512}

accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --main_process_port 29501 \
    scripts/eval_bagel_ocr.py \
    --model_path ${MODEL_PATH} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --num_steps ${NUM_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --resolution ${RESOLUTION} \
    --save_images
