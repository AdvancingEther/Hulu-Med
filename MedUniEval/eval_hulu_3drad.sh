#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/deeplearning/anaconda3/envs/hulu/bin/python"

export CUDA_VISIBLE_DEVICES="3"

DATASET_JSON_PATH="$SCRIPT_DIR/data/3drad/3drad_task1.json"

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

MAX_NEW_TOKENS="64"
MAX_IMAGE_NUM="600"
TEMPERATURE="0"
TOP_P="0.95"
REPETITION_PENALTY="1.0"

if [ ! -f "$DATASET_JSON_PATH" ]; then
    echo "3D-RAD dataset json not found: $DATASET_JSON_PATH" >&2
    exit 1
fi

PROXY_URL="${PROXY_URL:-http://127.0.0.1:15732}"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"

SLICE_SETTINGS=("32" "64" "96")
TOKEN_COMPRESSION_SETTINGS=("true" "false")

run_experiment() {
    local rad3d_num_slices="$1"
    local use_token_compression="$2"
    local output_tag

    if [ "$use_token_compression" = "true" ]; then
        output_tag="compressl1"
    else
        output_tag="ori"
    fi

    local output_path="$SCRIPT_DIR/output/3drad_hf_metrics_slice_${rad3d_num_slices}_${output_tag}"
    local results_dir="$output_path/$EVAL_DATASETS"
    local results_path="$results_dir/results.json"

    mkdir -p "$output_path"

    echo "========================================"
    echo "Running slice=${rad3d_num_slices}, use_token_compression=${use_token_compression}"
    echo "Output: $output_path"
    echo "========================================"

    echo "Stage 1/2: inference"
    if [ "$CHUNKS" -eq 1 ]; then
        (
            cd "$SCRIPT_DIR"
            "$PYTHON_BIN" scripts/infer_only.py \
                --eval_datasets "$EVAL_DATASETS" \
                --dataset_json_path "$DATASET_JSON_PATH" \
                --output_path "$output_path" \
                --model_name "$MODEL_NAME" \
                --model_path "$MODEL_PATH" \
                --seed "$SEED" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --max_image_num "$MAX_IMAGE_NUM" \
                --rad3d_num_slices "$rad3d_num_slices" \
                --use_token_compression "$use_token_compression" \
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
                    --output_path "$output_path" \
                    --model_name "$MODEL_NAME" \
                    --model_path "$MODEL_PATH" \
                    --seed "$SEED" \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --max_image_num "$MAX_IMAGE_NUM" \
                    --rad3d_num_slices "$rad3d_num_slices" \
                    --use_token_compression "$use_token_compression" \
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

    if [ ! -f "$results_path" ]; then
        echo "results.json not found after inference: $results_path" >&2
        exit 1
    fi

    echo "Stage 2/2: HF metrics"
    (
        cd "$SCRIPT_DIR"
        "$PYTHON_BIN" scripts/eval_3drad_hf_metrics.py \
            --results_path "$results_path" \
            --output_dir "$results_dir"
    )
}

for rad3d_num_slices in "${SLICE_SETTINGS[@]}"; do
    for use_token_compression in "${TOKEN_COMPRESSION_SETTINGS[@]}"; do
        run_experiment "$rad3d_num_slices" "$use_token_compression"
    done
done
