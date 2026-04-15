#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/deeplearning/anaconda3/envs/hulu/bin/python"

export CUDA_VISIBLE_DEVICES="3"

DATASET_JSON_PATH="$SCRIPT_DIR/data/3drad/3drad_task1.json"
EVAL_DATASETS="3DRad"

MODEL_NAME="Hulumed_qwen2_visionzip"
MODEL_PATH="/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/Hulu-Med-7B"

USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$TOTAL_GPUS

SEED="42"
REASONING="False"
TEST_TIMES="1"
VISION_ZIP_ENABLE="true"

MAX_NEW_TOKENS="64"
MAX_IMAGE_NUM="600"
TEMPERATURE="0"
TOP_P="0.95"
REPETITION_PENALTY="1.0"
FIXED_RESOLUTION="${FIXED_RESOLUTION:-336}"

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
VZIP_TAGS=("11" "33" "50" "70")
VZIP_DOMAIN_RATIOS=("0.10" "0.30" "0.45" "0.65")
VZIP_CONTEXTUAL_RATIOS=("0.01" "0.03" "0.05" "0.05")

run_experiment() {
    local rad3d_num_slices="$1"
    local vzip_tag="$2"
    local vision_zip_domain_kept_ratio="$3"
    local vision_zip_contextual_kept_ratio="$4"
    local -a fixed_resolution_args=()

    local output_tag="vzip_${vzip_tag}"
    if [ -n "$FIXED_RESOLUTION" ]; then
        output_tag="${output_tag}_fixres${FIXED_RESOLUTION}"
        fixed_resolution_args=(--fixed_resolution "$FIXED_RESOLUTION")
    fi

    local output_path="$SCRIPT_DIR/output/3drad_hf_metrics_slice_${rad3d_num_slices}_${output_tag}"
    local results_dir="$output_path/$EVAL_DATASETS"
    local results_path="$results_dir/results.json"

    mkdir -p "$output_path"

    echo "========================================"
    echo "Running slice=${rad3d_num_slices}, vzip=${vzip_tag} (domain=${vision_zip_domain_kept_ratio}, contextual=${vision_zip_contextual_kept_ratio})"
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
                "${fixed_resolution_args[@]}" \
                --vision_zip_enable "$VISION_ZIP_ENABLE" \
                --vision_zip_domain_kept_ratio "$vision_zip_domain_kept_ratio" \
                --vision_zip_contextual_kept_ratio "$vision_zip_contextual_kept_ratio" \
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
                    "${fixed_resolution_args[@]}" \
                    --vision_zip_enable "$VISION_ZIP_ENABLE" \
                    --vision_zip_domain_kept_ratio "$vision_zip_domain_kept_ratio" \
                    --vision_zip_contextual_kept_ratio "$vision_zip_contextual_kept_ratio" \
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
    for idx in "${!VZIP_TAGS[@]}"; do
        run_experiment \
            "$rad3d_num_slices" \
            "${VZIP_TAGS[$idx]}" \
            "${VZIP_DOMAIN_RATIOS[$idx]}" \
            "${VZIP_CONTEXTUAL_RATIOS[$idx]}"
    done
done
