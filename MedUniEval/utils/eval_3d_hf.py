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


def evaluate_m3d(out_samples):
    return _evaluate_core_hf(
        out_samples=out_samples,
        desc="Evaluating M3D predictions",
        category_key="type",
        category_name_mapping=QUESTION_TYPE_MAPPING,
        calculate_overall=True,
    )


def evaluate_3drad(out_samples):
    return _evaluate_core_hf(
        out_samples=out_samples,
        desc="Evaluating 3D-RAD predictions",
        category_key="type",
        category_name_mapping=THREERAD_TYPE_MAPPING,
        calculate_overall=False,
    )