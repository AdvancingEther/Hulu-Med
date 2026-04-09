import argparse
import csv
import json
import os
from collections import Counter


TASK_TYPE_MAPPING = {
    "Task1_Image_Observation": ("Organ_Identification", "Observation"),
    "Task2_Anomaly_Detection": ("Abnormality_Detection", "Observation"),
    "Task3_Medical_Computation": ("Medical_Computation", "Measurement"),
    "Task4_Existence_Detection": ("Abnormality_Detection", "Presence"),
    "Task5_Static_Temporal_Diagnosis": ("Abnormality_Detection", "Static_Temporal_Status"),
    "Task6_Longitudinal_Temporal_Diagnosis": ("Abnormality_Detection", "Longitudinal_Temporal_Status"),
}

SUBTASK_MAPPING = {
    ("Task1_Image_Observation", "Anatomical_observation"): ("Organ_Identification", "Organ_Name"),
    ("Task1_Image_Observation", "Pathological_observation"): ("Abnormality_Detection", "Presence"),
    ("Task2_Anomaly_Detection", "Abnormality_feature"): ("Abnormality_Detection", "Severity"),
    ("Task2_Anomaly_Detection", "Abnormality_position"): ("Spatial_Relationship", "Location"),
    ("Task2_Anomaly_Detection", "Abnormality_type"): ("Abnormality_Detection", "Presence"),
    ("Task2_Anomaly_Detection", "Diagnosis"): ("Abnormality_Detection", "Presence"),
    ("Task3_Medical_Computation", "Diameter"): ("Medical_Computation", "Distance"),
    ("Task3_Medical_Computation", "Size"): ("Medical_Computation", "Distance"),
    ("Task3_Medical_Computation", "Thickness"): ("Medical_Computation", "Thickness"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D-RAD dataset json that points to cached .npy volumes."
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="/home/deeplearning/data/data2/wzc/VolInterp/methods/3D-RAD/3DRAD/test",
        help="Root directory of the original 3D-RAD test CSV files.",
    )
    parser.add_argument(
        "--ct_rate_root",
        type=str,
        default="/home/deeplearning/data/data2/lpt/project_2/datasets/CT-RATE/dataset/valid_fixed",
        help="Root directory of CT-RATE NIfTI files used to resolve VolumeName.",
    )
    parser.add_argument(
        "--npy_cache_root",
        type=str,
        default="/home/deeplearning/data/data2/wzc/VolInterp/Data/3D-RAD/Rad3d_npy",
        help="Root directory of cached .npy volumes.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/deeplearning/data/data2/wzc/VolInterp/methods/prune-proj/MedUniEval/data/3drad/3drad_test.json",
        help="Output dataset json path.",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Optional comma-separated task folder names to keep.",
    )
    parser.add_argument(
        "--nii_axis",
        type=int,
        default=2,
        help="Metadata field kept for compatibility.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error when a mapped .npy file does not exist.",
    )
    return parser.parse_args()


def map_volume_to_ct_rate_path(volume_name, ct_rate_root):
    if not volume_name.endswith(".nii.gz"):
        raise ValueError(f"Unexpected VolumeName format: {volume_name}")

    stem = volume_name[:-7]
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "test":
        raise ValueError(f"Unexpected VolumeName format: {volume_name}")

    case_id, series_letter, image_idx = parts[1], parts[2], parts[3]
    valid_case = f"valid_{case_id}"
    valid_series = f"{valid_case}_{series_letter}"
    return os.path.join(ct_rate_root, valid_case, valid_series, f"{valid_series}_{image_idx}.nii.gz")


def map_npy_cache_path(nii_path, ct_rate_root, npy_cache_root):
    rel_path = os.path.relpath(nii_path, ct_rate_root)
    rel_path = rel_path.replace("valid_", "test_")
    if rel_path.endswith(".nii.gz"):
        rel_path = rel_path[:-7] + ".npy"
    elif rel_path.endswith(".nii"):
        rel_path = rel_path[:-4] + ".npy"
    else:
        rel_path = os.path.splitext(rel_path)[0] + ".npy"
    return os.path.join(npy_cache_root, rel_path)


def normalize_question_type(value):
    value = (value or "").strip().lower()
    return "CLOSED" if value in {"close", "closed"} else "OPEN"


def build_choices_text(row):
    choices = []
    for label in ["A", "B", "C", "D"]:
        choice_value = (row.get(f"Choice {label}") or "").strip()
        if choice_value and choice_value != "-":
            choices.append(f"{label}. {choice_value}")
    return "\n".join(choices)


def build_question_text(row):
    question = (row.get("Question") or "").strip()
    choices_text = build_choices_text(row)
    if choices_text:
        return "<video>\n" + question + "\n" + choices_text
    return "<video>\n" + question


def map_type_and_subtype(task_name, subtask_name):
    if (task_name, subtask_name) in SUBTASK_MAPPING:
        return SUBTASK_MAPPING[(task_name, subtask_name)]
    if task_name in TASK_TYPE_MAPPING:
        return TASK_TYPE_MAPPING[task_name]
    return "Abnormality_Detection", "Unknown"


def iter_test_csvs(test_root):
    for task_name in sorted(os.listdir(test_root)):
        task_dir = os.path.join(test_root, task_name)
        if not os.path.isdir(task_dir):
            continue
        for filename in sorted(os.listdir(task_dir)):
            if filename.endswith(".csv"):
                yield task_name, filename[:-4], os.path.join(task_dir, filename)


def main():
    args = parse_args()
    selected_tasks = {item.strip() for item in args.task_names.split(",") if item.strip()}

    samples = []
    type_counter = Counter()
    task_counter = Counter()
    missing_npy = []

    for task_name, subtask_name, csv_path in iter_test_csvs(args.test_root):
        if selected_tasks and task_name not in selected_tasks:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                volume_name = (row.get("VolumeName") or "").strip()
                if not volume_name:
                    continue

                nii_path = map_volume_to_ct_rate_path(volume_name, args.ct_rate_root)
                npy_path = map_npy_cache_path(nii_path, args.ct_rate_root, args.npy_cache_root)
                npy_exists = os.path.exists(npy_path)
                if not npy_exists:
                    missing_npy.append((csv_path, row_idx + 2, npy_path))
                    if args.strict:
                        raise FileNotFoundError(f"Missing cached npy: {npy_path} ({csv_path}:{row_idx + 2})")

                mapped_type, mapped_subtype = map_type_and_subtype(task_name, subtask_name)
                question_type = normalize_question_type(row.get("QuestionType"))
                answer_choice = (row.get("AnswerChoice") or "").strip()
                answer = (row.get("Answer") or "").strip()
                gold_answer = answer_choice if question_type == "CLOSED" and answer_choice else answer

                sample = {
                    "id": f"{task_name}/{subtask_name}/{row_idx}",
                    "type": mapped_type,
                    "sub-type": mapped_subtype,
                    "Question_Type": question_type,
                    "npy_path": npy_path,
                    "conversations": [
                        {"from": "human", "value": build_question_text(row)},
                        {"from": "gpt", "value": gold_answer},
                    ],
                }
                samples.append(sample)
                type_counter[mapped_type] += 1
                task_counter[task_name] += 1

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples to: {args.output_json}")
    print("Counts by type:")
    for key, value in sorted(type_counter.items()):
        print(f"  {key}: {value}")
    print("Counts by task:")
    for key, value in sorted(task_counter.items()):
        print(f"  {key}: {value}")
    if missing_npy:
        print(f"Warning: {len(missing_npy)} samples point to missing npy files.")
        for csv_path, row_num, npy_path in missing_npy[:10]:
            print(f"  missing: {npy_path} ({csv_path}:{row_num})")


if __name__ == "__main__":
    main()
