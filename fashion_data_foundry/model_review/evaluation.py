"""候选评估、拒识阈值选择与模型选型。

评估口径：

* 调参（含拒识阈值）只在 validation 上完成，test 仅在选定模型后跑一次最终检验；
* 预测允许拒识：置信度低于阈值时输出 ``abstain``，同时报告非拒识覆盖率，
  不允许只报一个总体准确率来掩盖稀有类表现；
* 指标按类别给出 precision / recall / f1 / support，并附宏平均、平衡准确率与
  覆盖率，选型依据同时考虑类不平衡、家族规模和覆盖率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

ABSTAIN = "abstain"
TARGET_COVERAGE = 0.8
TARGET_ABSTENTION_RATE = 0.2
MIN_COVERAGE = 0.7
MIN_GAIN_OVER_BASELINE = 0.02
MIN_TEST_LABELED = 2
MIN_TRAIN_PER_CLASS = 2
MIN_VALIDATION_LABELED = 2


@dataclass(slots=True)
class EvaluatedCandidate:
    candidate_id: str
    feature_set: str
    hyperparameters: dict[str, Any]
    validation: dict[str, Any]
    test: dict[str, Any] | None = None
    selection_rationale: list[str] = field(default_factory=list)
    model_bundle: dict[str, Any] = field(default_factory=dict)


def predicted_label(probability: Mapping[str, float]) -> tuple[str, float]:
    if not probability:
        return ABSTAIN, 0.0
    label, confidence = max(probability.items(), key=lambda item: item[1])
    return label, confidence


def apply_abstention(
    probability_rows: Sequence[Mapping[str, float]],
    threshold: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for probabilities in probability_rows:
        label, confidence = predicted_label(probabilities)
        abstained = confidence < threshold
        predictions.append(
            {
                "label": ABSTAIN if abstained else label,
                "abstained": abstained,
                "confidence": round(confidence, 6),
                "probabilities": {key: round(value, 6) for key, value in sorted(probabilities.items())},
            }
        )
    return predictions


def per_class_metrics(
    predictions: Sequence[Mapping[str, Any]],
    actual_labels: Sequence[str],
    classes: Sequence[str],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    correct = 0
    covered = 0
    abstentions = 0
    for label in classes:
        tp = fp = fn = support = 0
        for prediction, actual in zip(predictions, actual_labels):
            if actual == label:
                support += 1
            if prediction["abstained"]:
                if actual == label:
                    fn += 1
                continue
            predicted = prediction["label"]
            if predicted == label and actual == label:
                tp += 1
            elif predicted == label and actual != label:
                fp += 1
            elif actual == label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    for prediction, actual in zip(predictions, actual_labels):
        if prediction["abstained"]:
            abstentions += 1
            continue
        covered += 1
        if prediction["label"] == actual:
            correct += 1
    total = max(len(actual_labels), 1)
    macro_f1 = round(
        sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1), 6
    )
    recalls = [item["recall"] for item in per_class.values() if item["support"]]
    balanced_accuracy = round(sum(recalls) / len(recalls), 6) if recalls else 0.0
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy_on_covered": round(correct / covered, 6) if covered else 0.0,
        "coverage": round(covered / total, 6),
        "abstention_rate": round(abstentions / total, 6),
        "support": len(actual_labels),
    }


def choose_abstention_threshold(
    probability_rows: Sequence[Mapping[str, float]],
    actual_labels: Sequence[str],
    classes: Sequence[str],
    target_abstention: float = TARGET_ABSTENTION_RATE,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[float, dict[str, Any]]:
    """在验证集上挑拒识阈值。

    可行域要求覆盖率不低于 ``min_coverage``；在可行域内优先选择拒识率最接近
    目标值（默认 20%）的阈值，拒识率差距相同时再比较宏 F1 与平衡准确率。
    这样阈值不会因为高覆盖下宏 F1 略高就退化成 0，低置信度的稀有类预测会
    真正进入拒识，而不是被一个总体指标掩盖。
    """
    confidences = sorted(
        max(row.values()) for row in probability_rows if row
    )
    if not confidences:
        return 1.0, per_class_metrics([], actual_labels, classes)
    quantile_index = min(
        len(confidences) - 1,
        max(0, round(target_abstention * len(confidences))),
    )
    anchor = confidences[quantile_index]
    candidate_thresholds = sorted(
        {0.0, anchor, *(value + 1e-12 for value in confidences)}
    )
    best_threshold = 0.0
    best_metrics = per_class_metrics(
        apply_abstention(probability_rows, 0.0), actual_labels, classes
    )
    best_score: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds:
        predictions = apply_abstention(probability_rows, threshold)
        metrics = per_class_metrics(predictions, actual_labels, classes)
        if metrics["coverage"] < min_coverage:
            continue
        score = (
            -abs(metrics["abstention_rate"] - target_abstention),
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return round(best_threshold, 6), best_metrics


def evaluate_candidate(
    candidate_id: str,
    feature_set: str,
    hyperparameters: Mapping[str, Any],
    model: Any,
    classes: Sequence[str],
    validation_rows: Sequence[Sequence[float]],
    validation_labels: Sequence[str],
) -> EvaluatedCandidate:
    validation_probabilities = model.predict_proba(validation_rows)
    threshold, threshold_metrics = choose_abstention_threshold(
        validation_probabilities, validation_labels, classes
    )
    validation_predictions = apply_abstention(validation_probabilities, threshold)
    validation_metrics = per_class_metrics(
        validation_predictions, validation_labels, classes
    )
    validation_metrics["abstention_threshold"] = threshold
    validation_metrics["threshold_search_target_abstention"] = TARGET_ABSTENTION_RATE
    validation_metrics["threshold_search_min_coverage"] = MIN_COVERAGE
    validation_metrics["confidence_distribution"] = _confidence_distribution(
        validation_probabilities
    )
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        feature_set=feature_set,
        hyperparameters=dict(hyperparameters),
        validation=validation_metrics,
        model_bundle=model.to_bundle(),
    )


def final_test(
    candidate: EvaluatedCandidate,
    model: Any,
    classes: Sequence[str],
    test_rows: Sequence[Sequence[float]],
    test_labels: Sequence[str],
) -> dict[str, Any]:
    threshold = candidate.validation["abstention_threshold"]
    predictions = apply_abstention(model.predict_proba(test_rows), threshold)
    metrics = per_class_metrics(predictions, test_labels, classes)
    metrics["abstention_threshold"] = threshold
    candidate.test = metrics
    return metrics


def select_candidate(
    candidates: Sequence[EvaluatedCandidate],
    class_support: Mapping[str, int],
    family_profile: Mapping[str, Any],
) -> tuple[str | None, list[str]]:
    """选型：必须先跑赢朴素基线，再在宏 F1 / 平衡准确率 / 覆盖率之间权衡。"""
    if not candidates:
        return None, ["没有可训练候选"]
    baseline = next(
        (item for item in candidates if item.candidate_id == "majority_prior_baseline"),
        None,
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.validation["macro_f1"],
            item.validation["balanced_accuracy"],
            item.validation["coverage"],
        ),
        reverse=True,
    )
    leader = ranked[0]
    rationale: list[str] = []
    rare_classes = sorted(
        label for label, support in class_support.items()
        if support <= max(2, min(class_support.values()) * 1.5)
    ) or [min(class_support, key=class_support.get)]
    for candidate in candidates:
        macro = candidate.validation["macro_f1"]
        balanced = candidate.validation["balanced_accuracy"]
        coverage = candidate.validation["coverage"]
        rare_f1 = min(
            (candidate.validation["per_class"][label]["f1"] for label in rare_classes
             if label in candidate.validation["per_class"]),
            default=0.0,
        )
        candidate.selection_rationale = [
            f"validation macro_f1={macro}, balanced_accuracy={balanced}, coverage={coverage}",
            f"稀有类别 {rare_classes} 的最低 f1={rare_f1}，类别支持度 {dict(sorted(class_support.items()))}",
            f"家族规模最小={family_profile.get('smallest_family_size')}、最大={family_profile.get('largest_family_size')}，"
            "划分以 style_family 为单位，指标反映跨家族泛化而非变体记忆",
        ]
    if baseline is not None and leader.candidate_id != baseline.candidate_id:
        gain = leader.validation["macro_f1"] - baseline.validation["macro_f1"]
        rationale.append(
            f"领先候选 {leader.candidate_id} 的 macro_f1 较朴素基线高 {round(gain, 6)}"
        )
        if gain < MIN_GAIN_OVER_BASELINE:
            rationale.append(
                f"领先优势低于 {MIN_GAIN_OVER_BASELINE}，复杂模型不足以证明换季泛化收益，保留朴素基线"
            )
            return baseline.candidate_id, rationale
    if baseline is not None and leader.candidate_id == baseline.candidate_id:
        rationale.append("朴素先验基线在验证集上不落后于特征模型，按可解释与治理成本优先保留基线")
    else:
        rationale.append(
            f"选择 {leader.candidate_id}：宏 F1 与平衡准确率领先，且在最低覆盖率 "
            f"{MIN_COVERAGE:.0%} 约束下保留约 {TARGET_ABSTENTION_RATE:.0%} 的拒识空间"
        )
    return leader.candidate_id, rationale


def _confidence_distribution(
    probability_rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    confidences = [max(row.values()) for row in probability_rows if row]
    if not confidences:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    ordered = sorted(confidences)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "min": round(ordered[0], 6),
        "median": round(median, 6),
        "max": round(ordered[-1], 6),
    }


def sufficiency_blockers(
    train_labels: Sequence[str],
    validation_labels: Sequence[str],
    test_labels: Sequence[str],
    classes: Sequence[str],
) -> list[str]:
    """数据不足时给出去向明确的阻断原因，与正常发布状态区分。"""
    blockers: list[str] = []
    if len(set(train_labels)) < 2:
        blockers.append("训练集硬标签类别少于 2 个，无法训练多分类器")
    train_support = {label: train_labels.count(label) for label in classes}
    for label, support in sorted(train_support.items()):
        if support < MIN_TRAIN_PER_CLASS:
            blockers.append(f"类别 {label} 训练样本仅 {support} 条（要求 >= {MIN_TRAIN_PER_CLASS}）")
    if len(validation_labels) < MIN_VALIDATION_LABELED:
        blockers.append(
            f"验证集带硬标签样本仅 {len(validation_labels)} 条（要求 >= {MIN_VALIDATION_LABELED}）"
        )
    if len(test_labels) < MIN_TEST_LABELED:
        blockers.append(f"测试集带硬标签样本仅 {len(test_labels)} 条（要求 >= {MIN_TEST_LABELED}）")
    return blockers
