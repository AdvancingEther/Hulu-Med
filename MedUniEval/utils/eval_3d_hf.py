import collections
import re
import warnings

import evaluate
from tabulate import tabulate
from tqdm import tqdm

warnings.simplefilter("ignore")


QUESTION_TYPE_MAPPING = {
    1: "Plane",
    2: "Phase",
    3: "Organ",
    4: "Abnormality",
    5: "Location",
}

THREERAD_TYPE_MAPPING = {
    "Medical_Computation": "Medical Computation",
    "Spatial_Relationship": "Spatial Relationship",
    "Abnormality_Detection": "Abnormality Detection",
    "Organ_Identification": "Organ Identification",
    "Image_Quality": "Image Quality",
}

THREERAD_TASK_SUBTASKS = {
    "Task1_Image_Observation": [
        "Anatomical_observation",
        "Pathological_observation",
    ],
    "Task2_Anomaly_Detection": [
        "Abnormality_feature",
        "Abnormality_position",
        "Abnormality_type",
        "Diagnosis",
    ],
    "Task3_Medical_Computation": [
        "Diameter",
        "Size",
        "Thickness",
    ],
    "Task4_Existence_Detection": [
        "Arterial wall calcification",
        "Atelectasis",
        "Bronchiectasis",
        "Cardiomegaly",
        "Consolidation",
        "Coronary artery wall calcification",
        "Emphysema",
        "Hiatal hernia",
        "Interlobular septal thickening",
        "Lung nodule",
        "Lung opacity",
        "Lymphadenopathy",
        "Medical material",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Pericardial effusion",
        "Pleural effusion",
        "Pulmonary fibrotic sequela",
    ],
    "Task5_Static_Temporal_Diagnosis": ["b", "c", "d", "e", "f", "g", "h"],
    "Task6_Longitudinal_Temporal_Diagnosis": ["b", "c", "d", "e", "f", "g", "h"],
}


_HF_METRIC_CACHE = {}


def _get_hf_metric(name: str):
    if name not in _HF_METRIC_CACHE:
        _HF_METRIC_CACHE[name] = evaluate.load(name)
    return _HF_METRIC_CACHE[name]


def extract_choice_letter(text):
    """
    Support choices like:
      a
      A
      a.
      b)
      C:
    and allow range a-h.
    """
    text = str(text).lower().strip()
    match = re.match(r"^([a-h])[.\):]?\s*", text)
    if match:
        return match.group(1)
    if text in list("abcdefgh"):
        return text
    return text


def _normalize_open_text(text):
    """
    Keep preprocessing light to stay close to official HF-evaluate style.
    """
    if text is None:
        return ""
    return str(text).strip()


def _compute_open_metrics_hf(preds, refs):
    """
    Corpus-level HF evaluate metrics.
    Returns values in [0, 1].
    """
    if len(preds) == 0:
        return {
            "bleu": 0.0,       # BLEU-1
            "rouge1": 0.0,
            "meteor": 0.0,
            "bert_f1": 0.0,
        }

    preds = [_normalize_open_text(x) for x in preds]
    refs = [_normalize_open_text(x) for x in refs]

    bleu_metric = _get_hf_metric("bleu")
    rouge_metric = _get_hf_metric("rouge")
    meteor_metric = _get_hf_metric("meteor")
    bertscore_metric = _get_hf_metric("bertscore")

    bleu_result = bleu_metric.compute(
        predictions=preds,
        references=[[r] for r in refs],
        max_order=1,
    )

    rouge_result = rouge_metric.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rouge1"],
    )

    meteor_result = meteor_metric.compute(
        predictions=preds,
        references=refs,
    )

    bert_result = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type="roberta-large",
    )
    bert_f1 = (
        sum(bert_result["f1"]) / len(bert_result["f1"])
        if len(bert_result["f1"]) > 0
        else 0.0
    )

    return {
        "bleu": float(bleu_result.get("bleu", 0.0)),
        "rouge1": float(rouge_result.get("rouge1", 0.0)),
        "meteor": float(meteor_result.get("meteor", 0.0)),
        "bert_f1": float(bert_f1),
    }


def _format_open_metrics_to_table(open_metrics, has_open_questions=True):
    def fmt_metric(key):
        if not has_open_questions:
            return "N/A"
        return f"{open_metrics.get(key, 0.0) * 100:.4f}"

    return [
        ["BLEU-1", fmt_metric("bleu")],
        ["ROUGE-1", fmt_metric("rouge1")],
        ["METEOR", fmt_metric("meteor")],
        ["BERTScore F1", fmt_metric("bert_f1")],
    ]


def _safe_mean(values):
    return sum(values) / len(values) if len(values) > 0 else 0.0


def _parse_3drad_task_info(pred_item):
    task_name = pred_item.get("task_name")
    subtask_name = pred_item.get("subtask_name")
    if task_name and subtask_name:
        return str(task_name).strip(), str(subtask_name).strip()

    sample_id = str(pred_item.get("id", "")).strip()
    parts = sample_id.split("/")
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _evaluate_core_hf(out_samples, desc, category_key, category_name_mapping, calculate_overall=True):
    """
    Keep the same outer interface as the old eval code:
        return results_text, metrics_result, wrong_answers_by_type

    Open questions:
        - bleu   (BLEU-1)
        - rouge1
        - meteor
        - bert_f1

    Closed questions:
        - accuracy
    """
    open_preds_by_type = collections.defaultdict(list)
    open_refs_by_type = collections.defaultdict(list)

    total_open_count_by_type = collections.defaultdict(int)
    closed_questions_count_by_type = collections.defaultdict(int)
    closed_questions_correct_by_type = collections.defaultdict(int)
    wrong_answers_by_type = collections.defaultdict(list)

    for pred_item in tqdm(out_samples, desc=desc):
        try:
            gt_value = pred_item["conversations"][1]["value"]
            question = pred_item["conversations"][0]["value"]
        except (KeyError, IndexError, TypeError):
            print(f"Warning: Invalid sample format, skipping sample: {pred_item}")
            continue

        pred_value = pred_item.get("response", "")
        category = pred_item.get(category_key)
        openclose_type = str(pred_item.get("Question_Type", "OPEN")).upper()

        if category is None:
            continue

        if openclose_type == "OPEN":
            total_open_count_by_type[category] += 1
            open_preds_by_type[category].append(pred_value)
            open_refs_by_type[category].append(gt_value)

        elif openclose_type in ["CLOSED", "CLOSE"]:
            closed_questions_count_by_type[category] += 1

            answer_letter = extract_choice_letter(gt_value)
            response_letter = extract_choice_letter(pred_value)
            is_correct = answer_letter == response_letter

            if is_correct:
                closed_questions_correct_by_type[category] += 1
            else:
                wrong_answer_log = {
                    "question": question,
                    "correct_answer": gt_value,
                    "predicted_answer": pred_value,
                }
                if "sub-type" in pred_item:
                    wrong_answer_log["sub_type"] = pred_item.get("sub-type")
                wrong_answers_by_type[category].append(wrong_answer_log)

    results_tables = []
    metrics_result = {"by_category": {}}

    all_categories = sorted(
        list(set(open_preds_by_type.keys()) | set(closed_questions_count_by_type.keys())),
        key=str,
    )

    category_open_metrics_dict = {}
    category_closed_accs = []

    for cat_key in all_categories:
        category_name = category_name_mapping.get(cat_key, f"Unknown Type {cat_key}")

        preds = open_preds_by_type.get(cat_key, [])
        refs = open_refs_by_type.get(cat_key, [])
        open_count = total_open_count_by_type.get(cat_key, 0)

        open_metrics = _compute_open_metrics_hf(preds, refs)
        category_open_metrics_dict[cat_key] = open_metrics

        closed_count = closed_questions_count_by_type.get(cat_key, 0)
        closed_correct = closed_questions_correct_by_type.get(cat_key, 0)
        closed_acc = (closed_correct / closed_count) if closed_count > 0 else 0.0
        if closed_count > 0:
            category_closed_accs.append(closed_acc)

        combined_table_data = _format_open_metrics_to_table(
            open_metrics,
            has_open_questions=(open_count > 0),
        ) + [
            ["Closed Question Accuracy", f"{closed_acc * 100:.4f}" if closed_count > 0 else "N/A"],
            ["Open Questions", open_count],
            ["Closed Questions", closed_count],
            ["Total Samples", open_count + closed_count],
        ]

        results_tables.extend([
            f"\n{'=' * 60}",
            f"Category: {category_name}",
            "=" * 60,
            tabulate(combined_table_data, headers=["Metric", "Performance (%)"], tablefmt="grid"),
        ])

        metrics_result["by_category"][category_name] = {
            "open": {
                "bleu": open_metrics["bleu"] * 100,
                "rouge1": open_metrics["rouge1"] * 100,
                "meteor": open_metrics["meteor"] * 100,
                "bert_f1": open_metrics["bert_f1"] * 100,
            },
            "closed": {
                "accuracy": closed_acc * 100,
                "total": closed_count,
                "correct": closed_correct,
            },
            "open_count": open_count,
            "closed_count": closed_count,
            "total_samples": open_count + closed_count,
        }

    if calculate_overall and all_categories:
        overall_open_avg = {}
        for metric_name in ["bleu", "rouge1", "meteor", "bert_f1"]:
            vals = [category_open_metrics_dict[c][metric_name] for c in category_open_metrics_dict]
            overall_open_avg[metric_name] = sum(vals) / len(vals) if len(vals) > 0 else 0.0

        overall_closed_acc_avg = (
            sum(category_closed_accs) / len(category_closed_accs)
            if len(category_closed_accs) > 0
            else 0.0
        )

        total_open_samples = sum(total_open_count_by_type.values())
        total_closed_samples = sum(closed_questions_count_by_type.values())
        total_closed_correct = sum(closed_questions_correct_by_type.values())

        overall_table_data = _format_open_metrics_to_table(
            overall_open_avg,
            has_open_questions=(total_open_samples > 0),
        )
        overall_table_data.extend([
            ["Closed Question Accuracy (Category Avg)", f"{overall_closed_acc_avg * 100:.4f}"],
            ["Open Questions", total_open_samples],
            ["Closed Questions", total_closed_samples],
            ["Total Samples", total_open_samples + total_closed_samples],
            ["Total Categories", len(all_categories)],
        ])

        results_tables.extend([
            f"\n{'=' * 60}",
            "Overall Performance Summary (Category Average)",
            "=" * 60,
            tabulate(overall_table_data, headers=["Metric", "Performance (%)"], tablefmt="grid"),
        ])

        metrics_result["overall"] = {
            "open": {
                "bleu": overall_open_avg["bleu"] * 100,
                "rouge1": overall_open_avg["rouge1"] * 100,
                "meteor": overall_open_avg["meteor"] * 100,
                "bert_f1": overall_open_avg["bert_f1"] * 100,
            },
            "closed": {
                "accuracy_category_avg": overall_closed_acc_avg * 100,
                "total": total_closed_samples,
                "correct": total_closed_correct,
            },
            "total_samples": total_open_samples + total_closed_samples,
            "num_categories": len(all_categories),
        }

    return "\n".join(results_tables), metrics_result, wrong_answers_by_type


def _evaluate_3drad_hf_by_task(out_samples):
    open_preds_by_task = collections.defaultdict(lambda: collections.defaultdict(list))
    open_refs_by_task = collections.defaultdict(lambda: collections.defaultdict(list))

    total_open_count_by_task = collections.defaultdict(lambda: collections.defaultdict(int))
    closed_questions_count_by_task = collections.defaultdict(lambda: collections.defaultdict(int))
    closed_questions_correct_by_task = collections.defaultdict(lambda: collections.defaultdict(int))
    wrong_answers_by_task = collections.defaultdict(lambda: collections.defaultdict(list))

    for pred_item in tqdm(out_samples, desc="Evaluating 3D-RAD predictions"):
        try:
            gt_value = pred_item["conversations"][1]["value"]
            question = pred_item["conversations"][0]["value"]
        except (KeyError, IndexError, TypeError):
            print(f"Warning: Invalid sample format, skipping sample: {pred_item}")
            continue

        task_name, subtask_name = _parse_3drad_task_info(pred_item)
        if task_name is None or subtask_name is None:
            print(f"Warning: Invalid 3D-RAD id format, skipping sample: {pred_item.get('id')}")
            continue

        pred_value = pred_item.get("response", "")
        openclose_type = str(pred_item.get("Question_Type", "OPEN")).upper()

        if openclose_type == "OPEN":
            total_open_count_by_task[task_name][subtask_name] += 1
            open_preds_by_task[task_name][subtask_name].append(pred_value)
            open_refs_by_task[task_name][subtask_name].append(gt_value)

        elif openclose_type in ["CLOSED", "CLOSE"]:
            closed_questions_count_by_task[task_name][subtask_name] += 1

            answer_letter = extract_choice_letter(gt_value)
            response_letter = extract_choice_letter(pred_value)
            is_correct = answer_letter == response_letter

            if is_correct:
                closed_questions_correct_by_task[task_name][subtask_name] += 1
            else:
                wrong_answer_log = {
                    "question": question,
                    "correct_answer": gt_value,
                    "predicted_answer": pred_value,
                }
                if "sub-type" in pred_item:
                    wrong_answer_log["sub_type"] = pred_item.get("sub-type")
                wrong_answers_by_task[task_name][subtask_name].append(wrong_answer_log)

    results_tables = []
    metrics_result = {"by_task": {}}

    all_tasks = sorted(
        list(set(open_preds_by_task.keys()) | set(closed_questions_count_by_task.keys())),
        key=str,
    )

    unknown_tasks = [task_name for task_name in all_tasks if task_name not in THREERAD_TASK_SUBTASKS]
    if unknown_tasks:
        raise ValueError(
            "Unexpected 3D-RAD task(s) found in results: "
            + ", ".join(unknown_tasks)
        )

    task_macro_open_metrics = {}
    task_macro_closed_accs = []

    for task_name in all_tasks:
        actual_subtask_names = sorted(
            list(
                set(open_preds_by_task.get(task_name, {}).keys())
                | set(closed_questions_count_by_task.get(task_name, {}).keys())
            ),
            key=str,
        )
        expected_subtask_names = THREERAD_TASK_SUBTASKS[task_name]

        missing_subtasks = [name for name in expected_subtask_names if name not in actual_subtask_names]
        unexpected_subtasks = [name for name in actual_subtask_names if name not in expected_subtask_names]
        if missing_subtasks or unexpected_subtasks:
            mismatch_parts = []
            if missing_subtasks:
                mismatch_parts.append("missing: " + ", ".join(missing_subtasks))
            if unexpected_subtasks:
                mismatch_parts.append("unexpected: " + ", ".join(unexpected_subtasks))
            raise ValueError(
                f"3D-RAD task {task_name} subtask mismatch; " + "; ".join(mismatch_parts)
            )

        metrics_result["by_task"][task_name] = {"subtasks": {}}
        subtask_open_metrics_dict = {}
        subtask_closed_accs = []

        for subtask_name in expected_subtask_names:
            preds = open_preds_by_task.get(task_name, {}).get(subtask_name, [])
            refs = open_refs_by_task.get(task_name, {}).get(subtask_name, [])
            open_count = total_open_count_by_task.get(task_name, {}).get(subtask_name, 0)

            open_metrics = _compute_open_metrics_hf(preds, refs)
            if open_count > 0:
                subtask_open_metrics_dict[subtask_name] = open_metrics

            closed_count = closed_questions_count_by_task.get(task_name, {}).get(subtask_name, 0)
            closed_correct = closed_questions_correct_by_task.get(task_name, {}).get(subtask_name, 0)
            closed_acc = (closed_correct / closed_count) if closed_count > 0 else 0.0
            if closed_count > 0:
                subtask_closed_accs.append(closed_acc)

            metrics_result["by_task"][task_name]["subtasks"][subtask_name] = {
                "open": {
                    "bleu": open_metrics["bleu"] * 100,
                    "rouge1": open_metrics["rouge1"] * 100,
                    "meteor": open_metrics["meteor"] * 100,
                    "bert_f1": open_metrics["bert_f1"] * 100,
                },
                "closed": {
                    "accuracy": closed_acc * 100,
                    "total": closed_count,
                    "correct": closed_correct,
                },
                "open_count": open_count,
                "closed_count": closed_count,
                "total_samples": open_count + closed_count,
            }

        task_open_macro = {}
        for metric_name in ["bleu", "rouge1", "meteor", "bert_f1"]:
            vals = [subtask_open_metrics_dict[name][metric_name] for name in subtask_open_metrics_dict]
            task_open_macro[metric_name] = _safe_mean(vals)

        task_closed_macro = _safe_mean(subtask_closed_accs)
        if len(subtask_closed_accs) > 0:
            task_macro_closed_accs.append(task_closed_macro)
        task_macro_open_metrics[task_name] = task_open_macro

        total_open_samples = sum(total_open_count_by_task.get(task_name, {}).values())
        total_closed_samples = sum(closed_questions_count_by_task.get(task_name, {}).values())
        total_closed_correct = sum(closed_questions_correct_by_task.get(task_name, {}).values())

        macro_table_data = _format_open_metrics_to_table(
            task_open_macro,
            has_open_questions=(len(subtask_open_metrics_dict) > 0),
        )
        macro_table_data.extend([
            [
                "Closed Question Accuracy (Macro Mean)",
                f"{task_closed_macro * 100:.4f}" if len(subtask_closed_accs) > 0 else "N/A",
            ],
            ["Open Questions", total_open_samples],
            ["Closed Questions", total_closed_samples],
            ["Total Samples", total_open_samples + total_closed_samples],
            ["Open Subtasks", len(subtask_open_metrics_dict)],
            ["Closed Subtasks", len(subtask_closed_accs)],
            ["Total Subtasks", len(expected_subtask_names)],
        ])

        results_tables.extend([
            f"\n{'=' * 60}",
            f"Task Macro Mean: {task_name}",
            "=" * 60,
            tabulate(macro_table_data, headers=["Metric", "Performance (%)"], tablefmt="grid"),
        ])

        metrics_result["by_task"][task_name]["macro_mean"] = {
            "open": {
                "bleu": task_open_macro["bleu"] * 100,
                "rouge1": task_open_macro["rouge1"] * 100,
                "meteor": task_open_macro["meteor"] * 100,
                "bert_f1": task_open_macro["bert_f1"] * 100,
            },
            "closed": {
                "accuracy_macro_mean": task_closed_macro * 100,
                "total": total_closed_samples,
                "correct": total_closed_correct,
            },
            "open_subtasks": len(subtask_open_metrics_dict),
            "closed_subtasks": len(subtask_closed_accs),
            "total_subtasks": len(expected_subtask_names),
            "total_samples": total_open_samples + total_closed_samples,
        }

    if len(all_tasks) > 0:
        overall_open_macro = {}
        for metric_name in ["bleu", "rouge1", "meteor", "bert_f1"]:
            vals = [task_macro_open_metrics[task_name][metric_name] for task_name in task_macro_open_metrics]
            overall_open_macro[metric_name] = _safe_mean(vals)

        overall_closed_macro = _safe_mean(task_macro_closed_accs)
        total_samples = 0
        for task_name in all_tasks:
            total_samples += metrics_result["by_task"][task_name]["macro_mean"]["total_samples"]

        metrics_result["overall"] = {
            "macro_mean": {
                "open": {
                    "bleu": overall_open_macro["bleu"] * 100,
                    "rouge1": overall_open_macro["rouge1"] * 100,
                    "meteor": overall_open_macro["meteor"] * 100,
                    "bert_f1": overall_open_macro["bert_f1"] * 100,
                },
                "closed": {
                    "accuracy_task_macro_mean": overall_closed_macro * 100,
                },
                "total_tasks": len(all_tasks),
                "total_samples": total_samples,
            }
        }

    wrong_answers_result = {
        task_name: dict(subtask_payload)
        for task_name, subtask_payload in wrong_answers_by_task.items()
    }

    return "\n".join(results_tables), metrics_result, wrong_answers_result


def evaluate_m3d(out_samples):
    return _evaluate_core_hf(
        out_samples=out_samples,
        desc="Evaluating M3D predictions",
        category_key="type",
        category_name_mapping=QUESTION_TYPE_MAPPING,
        calculate_overall=True,
    )


def evaluate_3drad(out_samples):
    return _evaluate_3drad_hf_by_task(out_samples)
