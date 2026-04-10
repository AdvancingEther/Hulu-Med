import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

from LLMs import init_llm


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_eval_datasets(datasets_str):
    return datasets_str.split(",")


def save_json(filename, payload):
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def run_inference_only(benchmark):
    model = benchmark.model
    output_path = benchmark.output_path
    num_chunks = benchmark.num_chunks
    chunk_idx = benchmark.chunk_idx

    os.makedirs(output_path, exist_ok=True)

    if num_chunks == 1:
        results_path = os.path.join(output_path, "results.json")
        out_samples = benchmark.run(benchmark.samples, model)
        save_json(results_path, out_samples)
        return results_path

    if num_chunks > 1:
        if chunk_idx == 0:
            import glob

            old_results = glob.glob(os.path.join(output_path, "results_*.json"))
            for result_path in old_results:
                try:
                    os.remove(result_path)
                    print(f"Removed old file: {result_path}")
                except OSError:
                    pass

            final_results_path = os.path.join(output_path, "results.json")
            if os.path.exists(final_results_path):
                os.remove(final_results_path)
                print(f"Removed old file: {final_results_path}")

        chunk_results_path = os.path.join(output_path, f"results_{chunk_idx}.json")
        out_samples = benchmark.run(benchmark.samples, model)
        save_json(chunk_results_path, out_samples)

        result_filenames = [
            name
            for name in os.listdir(output_path)
            if name.startswith("results_") and name.endswith(".json")
        ]
        if len(result_filenames) != num_chunks:
            print(
                f"Saved chunk results to {chunk_results_path}. "
                f"Waiting for the other {num_chunks - len(result_filenames)} chunk(s)."
            )
            return chunk_results_path

        total_results = []
        for result_name in sorted(result_filenames):
            result_path = os.path.join(output_path, result_name)
            print(f"Merging {result_path}")
            with open(result_path, "r", encoding="utf-8") as handle:
                total_results.extend(json.load(handle))

        final_results_path = os.path.join(output_path, "results.json")
        save_json(final_results_path, total_results)
        return final_results_path

    raise ValueError("num_chunks must be greater than 0")


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_datasets", type=parse_eval_datasets, default=["3DRad"])
    parser.add_argument("--datasets_path", type=str, default="benchmarks")
    parser.add_argument("--dataset_json_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="eval_results/inference_only")
    parser.add_argument("--model_name", type=str, default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=str, default="1")
    parser.add_argument("--use_vllm", type=str, default="True")
    parser.add_argument("--reasoning", type=str, default="False")

    parser.add_argument("--num_chunks", type=str, default="1")
    parser.add_argument("--chunk_idx", type=str, default="0")

    parser.add_argument("--max_image_num", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--rad3d_num_slices", type=int, default=64)

    parser.add_argument("--test_times", type=int, default=1)

    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["REASONING"] = args.reasoning
    os.environ["use_vllm"] = args.use_vllm
    os.environ["max_image_num"] = str(args.max_image_num)
    os.environ["RAD3D_NUM_SLICES"] = str(args.rad3d_num_slices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.use_vllm == "True":
        os.environ["tensor_parallel_size"] = args.tensor_parallel_size
        if int(args.tensor_parallel_size) > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        torch.multiprocessing.set_start_method("spawn")
        os.environ["num_chunks"] = args.num_chunks
        os.environ["chunk_idx"] = args.chunk_idx

    os.makedirs(args.output_path, exist_ok=True)

    print("initializing LLM...")
    model = init_llm(args)

    from benchmarks import prepare_benchmark

    for eval_dataset in args.eval_datasets:
        set_seed(args.seed)
        print(f"running inference on {eval_dataset}...")

        eval_dataset_path = args.dataset_json_path if args.dataset_json_path else os.path.join(args.datasets_path, eval_dataset)
        eval_output_path = os.path.join(args.output_path, eval_dataset)
        os.makedirs(eval_output_path, exist_ok=True)
        benchmark = prepare_benchmark(model, eval_dataset, eval_dataset_path, eval_output_path)
        benchmark.load_data()
        results_path = run_inference_only(benchmark)
        print(f"inference results saved to {results_path}")


if __name__ == "__main__":
    main()
