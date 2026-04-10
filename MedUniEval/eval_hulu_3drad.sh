#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="4"

DATASET_JSON_PATH="./data/3drad/3drad_task1.json"
OUTPUT_PATH="./output/3drad"

EVAL_DATASETS="3DRad"

MODEL_NAME="Hulumed_qwen2"
MODEL_PATH="/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/Hulu-Med-7B"

USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$TOTAL_GPUS
  
#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1
RAD3D_NUM_SLICES=64

# Eval LLM setting
MAX_NEW_TOKENS=16384
MAX_IMAGE_NUM=600
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0

if [ ! -f "$DATASET_JSON_PATH" ]; then
    echo "3D-RAD dataset json not found: $DATASET_JSON_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --dataset_json_path "$DATASET_JSON_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --rad3d_num_slices "$RAD3D_NUM_SLICES" \
    --use_vllm "$USE_VLLM" \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge False \
    --test_times "$TEST_TIMES"  \
