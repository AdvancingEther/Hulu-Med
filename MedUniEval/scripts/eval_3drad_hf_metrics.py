import argparse
import importlib.util
import json
import os


def load_eval_module(module_path):
    spec = importlib.util.spec_from_file_location("eval_3d_hf_module", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    spec.loader.exec_module(module)
    return module


def save_json(filename, payload):
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Path to the 3D-RAD results.json file.")
    parser.add_argument("--output_dir", default=None, help="Directory to store HF metric outputs. Defaults to the results directory.")
    parser.add_argument(
        "--eval_module_path",
        default=None,
        help="Optional path to eval_3d_hf.py. Defaults to MedUniEval/utils/eval_3d_hf.py next to this script.",
    )
    args = parser.parse_args()

    results_path = os.path.abspath(args.results_path)
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)

    eval_module_path = args.eval_module_path
    if eval_module_path is None:
        eval_module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "eval_3d_hf.py",
        )
    eval_module_path = os.path.abspath(eval_module_path)

    with open(results_path, "r", encoding="utf-8") as handle:
        out_samples = json.load(handle)

    eval_module = load_eval_module(eval_module_path)

    print("\n" + "=" * 80)
    print("Starting 3D-RAD HF Evaluation")
    print("=" * 80)

    results_text, metrics, wrong_answers = eval_module.evaluate_3drad(out_samples)
    print(results_text)

    metrics_path = os.path.join(output_dir, "hf_metrics.json")
    wrong_answers_path = os.path.join(output_dir, "hf_wrong_answers.json")

    save_json(metrics_path, metrics)
    save_json(wrong_answers_path, wrong_answers)

    print(f"\nHF metrics saved to: {metrics_path}")
    print(f"HF wrong answers saved to: {wrong_answers_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
