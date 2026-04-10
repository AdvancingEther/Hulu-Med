#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/deeplearning/anaconda3/envs/hulu/bin/python"

export CUDA_VISIBLE_DEVICES="3"

DATASET_JSON_PATH="$SCRIPT_DIR/data/3drad/3drad_task1.json"
OUTPUT_PATH="$SCRIPT_DIR/output/3drad_hf_metrics"

EVAL_DATASETS="3DRad"

MODEL_NAME="Hulumed_qwen2"
MODEL_PATH="/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/Hulu-Med-7B"

USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$TOTAL_GPUS

SEED="42"
REASONING="False"
TEST_TIMES="1"
RAD3D_NUM_SLICES="64"

MAX_NEW_TOKENS="64"
MAX_IMAGE_NUM="600"
TEMPERATURE="0"
TOP_P="0.95"
REPETITION_PENALTY="1.0"

RESULTS_DIR="$OUTPUT_PATH/$EVAL_DATASETS"
RESULTS_PATH="$RESULTS_DIR/results.json"

if [ ! -f "$DATASET_JSON_PATH" ]; then
    echo "3D-RAD dataset json not found: $DATASET_JSON_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

echo "Stage 1/2: inference"
if [ "$CHUNKS" -eq 1 ]; then
    (
        cd "$SCRIPT_DIR"
        "$PYTHON_BIN" scripts/infer_only.py \
            --eval_datasets "$EVAL_DATASETS" \
            --dataset_json_path "$DATASET_JSON_PATH" \
            --output_path "$OUTPUT_PATH" \
            --model_name "$MODEL_NAME" \
            --model_path "$MODEL_PATH" \
            --seed "$SEED" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_image_num "$MAX_IMAGE_NUM" \
            --rad3d_num_slices "$RAD3D_NUM_SLICES" \
            --use_vllm "$USE_VLLM" \
            --reasoning "$REASONING" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --repetition_penalty "$REPETITION_PENALTY" \
            --test_times "$TEST_TIMES" \
            --num_chunks 1 \
            --chunk_idx 0
    )
else
    for idx in "${!GPULIST[@]}"; do
        (
            cd "$SCRIPT_DIR"
            CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" \
            "$PYTHON_BIN" scripts/infer_only.py \
                --eval_datasets "$EVAL_DATASETS" \
                --dataset_json_path "$DATASET_JSON_PATH" \
                --output_path "$OUTPUT_PATH" \
                --model_name "$MODEL_NAME" \
                --model_path "$MODEL_PATH" \
                --seed "$SEED" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --max_image_num "$MAX_IMAGE_NUM" \
                --rad3d_num_slices "$RAD3D_NUM_SLICES" \
                --use_vllm "$USE_VLLM" \
                --reasoning "$REASONING" \
                --temperature "$TEMPERATURE" \
                --top_p "$TOP_P" \
                --repetition_penalty "$REPETITION_PENALTY" \
                --test_times "$TEST_TIMES" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$idx"
        ) &
    done
    wait
fi

if [ ! -f "$RESULTS_PATH" ]; then
    echo "results.json not found after inference: $RESULTS_PATH" >&2
    exit 1
fi

echo "Stage 2/2: HF metrics"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:15732}"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"

(
    cd "$SCRIPT_DIR"
    "$PYTHON_BIN" scripts/eval_3drad_hf_metrics.py \
        --results_path "$RESULTS_PATH" \
        --output_dir "$RESULTS_DIR"
)
