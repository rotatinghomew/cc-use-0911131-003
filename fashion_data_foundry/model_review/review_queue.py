"""人工复核队列排序。

排序信号联合三类信息：

1. 模型不确定性（低置信度、拒识、标注分歧或标签缺失的固有不确定性）；
2. 稀有类别风险（训练支持度低的类别预测，错分代价更高）；
3. 款式家族覆盖（在预算内做家族轮询，避免预算全花在同一个大货家族上）。

分配采用两段式：先计算每条商品的内在优先级，再在家族轮询约束下依次入选——
每一轮先从尚未覆盖的家族取其最高分条目，之后允许回到高分家族补位，因此
分歧 / 拒识条目不会为了“看上去多样”被排到预算之外。

已确认商品与已撤回资产不允许再次入队；预算是硬上限，宁可少推也不超额。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from fashion_data_foundry.model_review.evaluation import ABSTAIN
from fashion_data_foundry.model_review.preparation import PreparedAssets

UNCERTAINTY_WEIGHT = 0.6
RARE_CLASS_WEIGHT = 0.25
FAMILY_COVERAGE_WEIGHT = 0.15
LOW_CONFIDENCE_LINE = 0.55
RARE_SUPPORT_RATIO = 0.25
# 标注侧不确定性（分歧 / 缺失）是复核的最高层级，必须先于模型拒识入队。
LABEL_UNCERTAINTY_LINE = 0.85
# 标注分歧 / 缺失 / 拒识条目会被提到该不确定性线以上，进入高优先级复核池。
HIGH_PRIORITY_LINE = 0.8


def build_review_queue(
    prepared: PreparedAssets,
    predictions: Sequence[Mapping[str, Any]],
    review_budget: int,
    confirmed_item_ids: Sequence[str],
    class_support: Mapping[str, int],
) -> list[dict[str, Any]]:
    if review_budget < 0:
        raise ValueError("review_budget 不能为负数")
    confirmed = set(confirmed_item_ids)
    withdrawn = set(prepared.withdrawn_item_ids)
    prediction_by_item = {item["item_id"]: item for item in predictions}
    max_support = max(class_support.values()) if class_support else 1

    eligible: list[dict[str, Any]] = []
    for item_id, state in prepared.label_states.items():
        if item_id in confirmed or item_id in withdrawn:
            continue
        prediction = prediction_by_item.get(item_id)
        if prediction is None:
            continue
        components, reasons = _score_item(
            state, prediction, class_support, max_support
        )
        eligible.append(
            {
                "item_id": item_id,
                "family": state.family,
                "intrinsic_score": round(
                    UNCERTAINTY_WEIGHT * components["uncertainty"]
                    + RARE_CLASS_WEIGHT * components["rarity"],
                    6,
                ),
                "score_components": components,
                "reasons": reasons,
                "predicted_label": prediction["label"],
                "confidence": prediction["confidence"],
            }
        )

    return _allocate_budget(eligible, review_budget)


def _score_item(
    state: Any,
    prediction: Mapping[str, Any],
    class_support: Mapping[str, int],
    max_support: int,
) -> tuple[dict[str, float], list[str]]:
    confidence = float(prediction["confidence"])
    uncertainty = 1.0 - confidence
    reasons: list[str] = []
    if prediction.get("abstained") or prediction["label"] == ABSTAIN:
        uncertainty = max(uncertainty, 0.95)
        reasons.append("model_abstained")
    if state.uncertain:
        uncertainty = max(uncertainty, 0.9)
        reasons.append(state.uncertainty_reason or "label_uncertainty")
        if state.votes:
            reasons.append("vote_distribution=" + _compact_votes(state.votes))
    if confidence < LOW_CONFIDENCE_LINE and "low_confidence" not in reasons:
        reasons.append("low_confidence")

    label = prediction["label"]
    support = class_support.get(label, 0)
    rarity = 1.0 - (support / max_support) if label != ABSTAIN else 1.0
    if label == ABSTAIN:
        reasons.append("rare_class:unknown_due_to_abstain")
    elif support and support <= max(2, max_support * RARE_SUPPORT_RATIO):
        reasons.append(f"rare_class:{label}(support={support})")

    components = {
        "uncertainty": round(uncertainty, 6),
        "rarity": round(rarity, 6),
        "family_coverage": 0.0,
    }
    return components, reasons


def _allocate_budget(
    eligible: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """三级预算分配。

    第一级是标注侧不确定性（标注者分歧 / 标签缺失）：这是真值层面的疑点，
    优先级高于任何模型信号，按内在分数先入队；第二级是模型侧高风险条目
    （拒识 / 低置信度 / 稀有类）；第三级再用家族轮询补齐剩余预算以保证
    款式家族覆盖。任何一级都不能把预算超掉，总长度永远不超过预算。
    """
    label_tier = [
        entry for entry in eligible
        if entry["score_components"]["uncertainty"] >= LABEL_UNCERTAINTY_LINE
        and any(
            reason in ("annotator_disagreement", "missing_label")
            for reason in entry["reasons"]
        )
    ]
    model_tier = [
        entry for entry in eligible
        if entry not in label_tier
        and entry["score_components"]["uncertainty"] >= HIGH_PRIORITY_LINE
    ]
    regular = [
        entry for entry in eligible
        if entry not in label_tier and entry not in model_tier
    ]
    for tier in (label_tier, model_tier):
        tier.sort(key=lambda item: (-item["intrinsic_score"], item["item_id"]))

    by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in regular:
        by_family.setdefault(entry["family"], []).append(entry)
    for entries in by_family.values():
        entries.sort(key=lambda item: (-item["intrinsic_score"], item["item_id"]))

    queue: list[dict[str, Any]] = []
    queued: set[str] = set()
    family_round: dict[str, int] = {}

    for tier_name, tier in (
        ("label_uncertainty", label_tier),
        ("model_risk", model_tier),
    ):
        for entry in tier:
            if len(queue) >= budget:
                break
            seen = family_round.get(entry["family"], 0)
            entry["score_components"]["family_coverage"] = round(1.0 / (1 + seen), 6)
            entry["priority_score"] = _final_score(entry["score_components"])
            entry["reasons"].append(f"family_coverage_round:{seen + 1}")
            entry["reasons"].append(f"priority_tier:{tier_name}")
            queued.add(entry["item_id"])
            family_round[entry["family"]] = seen + 1
            queue.append(entry)
        if len(queue) >= budget:
            break

    while len(queue) < budget:
        progressed = False
        for family in sorted(by_family):
            if len(queue) >= budget:
                break
            entries = by_family[family]
            while entries and entries[0]["item_id"] in queued:
                entries.pop(0)
            if not entries:
                continue
            round_index = family_round.get(family, 0)
            if round_index >= 1 and not _all_families_started(by_family, family_round):
                continue
            entry = entries.pop(0)
            seen = family_round.get(family, 0)
            entry["score_components"]["family_coverage"] = round(1.0 / (1 + seen), 6)
            entry["priority_score"] = _final_score(entry["score_components"])
            entry["reasons"].append(f"family_coverage_round:{seen + 1}")
            entry["reasons"].append("priority_tier:family_coverage")
            queued.add(entry["item_id"])
            family_round[family] = seen + 1
            queue.append(entry)
            progressed = True
        if not progressed:
            break

    queue.sort(key=lambda item: (-item["priority_score"], item["family"], item["item_id"]))
    return [
        {
            "item_id": entry["item_id"],
            "family": entry["family"],
            "reason": entry["reasons"],
            "priority_score": entry["priority_score"],
            "score_components": entry["score_components"],
            "predicted_label": entry["predicted_label"],
            "confidence": entry["confidence"],
        }
        for entry in queue
    ]


def _final_score(components: Mapping[str, float]) -> float:
    return round(
        UNCERTAINTY_WEIGHT * components["uncertainty"]
        + RARE_CLASS_WEIGHT * components["rarity"]
        + FAMILY_COVERAGE_WEIGHT * components["family_coverage"],
        6,
    )


def _all_families_started(
    by_family: Mapping[str, list[dict[str, Any]]],
    family_round: Mapping[str, int],
) -> bool:
    return all(family_round.get(family, 0) >= 1 for family in by_family)


def _compact_votes(votes: Mapping[str, int]) -> str:
    return ",".join(f"{label}:{count}" for label, count in sorted(votes.items()))
