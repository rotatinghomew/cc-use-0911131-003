"""本地资产的标签整理与款式家族不相交数据划分。

治理约束（与 fixtures/analysis/说明.md 对齐）：

* ``final_cutoff`` 之后才可用的标注（``known_at`` 更晚，包括修订标签）一律不参与
  选型、训练和最终检验；
* 被撤回资产（``withdrawn_at`` 不晚于截点）退出建模，也不会进入复核队列；
* 标注者意见不一致或标签缺失时保留不确定性，只记录投票分布，不把多数意见直接当真值；
* 训练 / 调参 / 最终检验三个集合按 ``style_family`` 划分，家族之间互不共享，避免
  同一款的不同颜色变体同时出现在训练集和验证集。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
DEFAULT_RATIOS = (0.6, 0.2, 0.2)
LABEL_VERSION_PREFIX = "label-set"


class InputError(ValueError):
    """输入资产不满足建模前置条件。"""


def parse_iso_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise InputError(f"日期字段必须是 ISO 8601 字符串，收到 {type(value).__name__}")
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise InputError(f"无法解析日期 {value!r}") from exc


@dataclass(frozen=True, slots=True)
class LabelState:
    """单个商品在截点内的标注共识状态。

    ``hard_label`` 仅在所有有效标注完全一致时给出；分歧或缺失时为 ``None``，
    投票分布原样保留在 ``votes`` 中，供复核队列识别不确定性。
    """

    item_id: str
    family: str
    hard_label: str | None
    votes: dict[str, int]
    annotator_ids: tuple[str, ...]
    uncertain: bool
    uncertainty_reason: str | None
    eligible_label_records: int
    dropped_label_records: int


@dataclass(frozen=True, slots=True)
class PreparedAssets:
    cutoff: str
    label_version: str
    items: dict[str, Mapping[str, Any]]
    families: dict[str, tuple[str, ...]]
    label_states: dict[str, LabelState]
    split_of_family: dict[str, str]
    split_of_item: dict[str, str]
    withdrawn_item_ids: tuple[str, ...]
    excluded_label_records: tuple[Mapping[str, Any], ...]

    def items_in_split(self, split: str) -> list[str]:
        return sorted(
            item_id
            for item_id, assigned in self.split_of_item.items()
            if assigned == split
        )

    def labeled_items_in_split(self, split: str) -> list[str]:
        return [
            item_id
            for item_id in self.items_in_split(split)
            if self.label_states[item_id].hard_label is not None
        ]

    def uncertain_items_in_split(self, split: str) -> list[str]:
        return [
            item_id
            for item_id in self.items_in_split(split)
            if self.label_states[item_id].uncertain
        ]


def _require_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise InputError("输入必须是 JSON 对象")
    return payload


def build_label_states(
    items: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    cutoff: date,
) -> tuple[dict[str, LabelState], list[Mapping[str, Any]]]:
    """按截点整理标注，返回每个商品的共识状态与被剔除的标注记录。"""
    raw_by_item: dict[str, list[Mapping[str, Any]]] = {}
    dropped: list[Mapping[str, Any]] = []
    for record in labels:
        known_at = parse_iso_date(record.get("known_at"))
        if known_at is None or known_at > cutoff:
            dropped.append(dict(record))
            continue
        raw_by_item.setdefault(str(record["item_id"]), []).append(record)

    states: dict[str, LabelState] = {}
    for item in items:
        item_id = str(item["item_id"])
        records = raw_by_item.get(item_id, [])
        # 同一标注者若存在多条截点内记录，仅保留 revision 最大的一条。
        latest_by_annotator: dict[str, Mapping[str, Any]] = {}
        for record in records:
            annotator = str(record["annotator_id"])
            previous = latest_by_annotator.get(annotator)
            if previous is None or int(record.get("revision", 1)) > int(previous.get("revision", 1)):
                latest_by_annotator[annotator] = record
        votes: dict[str, int] = {}
        for record in latest_by_annotator.values():
            value = record.get("label")
            if value in (None, ""):
                continue
            votes[str(value)] = votes.get(str(value), 0) + 1
        annotators = tuple(sorted(latest_by_annotator))
        dropped_count = sum(1 for record in labels if str(record.get("item_id")) == item_id) - len(records)
        if not votes:
            states[item_id] = LabelState(
                item_id=item_id,
                family=str(item["style_family"]),
                hard_label=None,
                votes={},
                annotator_ids=annotators,
                uncertain=True,
                uncertainty_reason="missing_label",
                eligible_label_records=len(records),
                dropped_label_records=dropped_count,
            )
        elif len(votes) == 1:
            states[item_id] = LabelState(
                item_id=item_id,
                family=str(item["style_family"]),
                hard_label=next(iter(votes)),
                votes=votes,
                annotator_ids=annotators,
                uncertain=False,
                uncertainty_reason=None,
                eligible_label_records=len(records),
                dropped_label_records=dropped_count,
            )
        else:
            states[item_id] = LabelState(
                item_id=item_id,
                family=str(item["style_family"]),
                hard_label=None,
                votes=dict(sorted(votes.items())),
                annotator_ids=annotators,
                uncertain=True,
                uncertainty_reason="annotator_disagreement",
                eligible_label_records=len(records),
                dropped_label_records=dropped_count,
            )
    return states, dropped


def _family_rarity(
    families: Mapping[str, Sequence[str]],
    states: Mapping[str, LabelState],
) -> dict[str, dict[str, int]]:
    profile: dict[str, dict[str, int]] = {}
    for family, member_ids in families.items():
        counts: dict[str, int] = {}
        for item_id in member_ids:
            state = states[item_id]
            for label, count in state.votes.items():
                counts[label] = counts.get(label, 0) + count
        profile[family] = counts
    return profile


def _family_dominant_label(
    family: str,
    families: Mapping[str, Sequence[str]],
    states: Mapping[str, LabelState],
    profile: Mapping[str, Mapping[str, int]],
) -> str:
    """家族分层键：以硬共识标签为主；分歧家族单独成层，不强行并入多数类。"""
    hard_labels = {
        states[item_id].hard_label
        for item_id in families[family]
        if states[item_id].hard_label is not None
    }
    if len(hard_labels) == 1:
        return next(iter(hard_labels))
    if profile[family]:
        return "mixed:" + "|".join(sorted(profile[family]))
    return "unlabeled"


def assign_family_splits(
    families: Mapping[str, Sequence[str]],
    states: Mapping[str, LabelState],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    """按家族分层、确定地分配到 train/validation/test。

    划分单位是整个 ``style_family``，因此同一款的颜色变体不会跨集合。
    先按家族共识标签分层，低频类家族数量最少，优先在三个集合间轮转，保证
    train/validation/test 都能观察到稀有类；层内再用带种子的散列打散，
    使同一份资产重复运行得到完全相同的划分且不依赖随机库状态。
    """
    profile = _family_rarity(families, states)
    ordered = sorted(families)
    layer_of_family = {
        family: _family_dominant_label(family, families, states, profile)
        for family in ordered
    }
    layers: dict[str, list[str]] = {}
    for family in ordered:
        layers.setdefault(layer_of_family[family], []).append(family)
    # 家族数最少的层（低频类）先分配，避免被大类挤掉检验覆盖。
    layer_order = sorted(layers, key=lambda name: (len(layers[name]), name))

    targets = {
        split: max(1, round(len(ordered) * ratios[index])) if len(ordered) >= len(SPLITS) else 1
        for index, split in enumerate(SPLITS)
    }
    while sum(targets.values()) > len(ordered):
        targets[SPLIT_TRAIN] -= 1
    while sum(targets.values()) < len(ordered):
        targets[SPLIT_TRAIN] += 1

    assignment: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}
    # 比例分层：先给每个层按比例切分内部分配名额（小层保证三个集合各至少 1 个），
    # 再在层内按确定性散列顺序、依名额轮转。这样大层与小层都覆盖三个集合，
    # 同时全局比例稳定，且同一份资产重复运行得到完全一致的划分。
    for layer_name in layer_order:
        layer_families = sorted(
            layers[layer_name],
            key=lambda family: (_deterministic_jitter(family, seed), family),
        )
        layer_quota = _layer_quota(len(layer_families), ratios)
        cursor = 0
        for family in layer_families:
            split = _take_quota_slot(layer_quota, cursor, counts, targets)
            assignment[family] = split
            counts[split] += 1
            cursor += 1
    return assignment


def _layer_quota(
    layer_size: int, ratios: tuple[float, float, float]
) -> dict[str, int]:
    if layer_size >= len(SPLITS):
        quota = {
            split: max(1, round(layer_size * ratios[index]))
            for index, split in enumerate(SPLITS)
        }
        while sum(quota.values()) > layer_size:
            quota[SPLIT_TRAIN] -= 1
        while sum(quota.values()) < layer_size:
            quota[SPLIT_TRAIN] += 1
        return quota
    quota = {split: 0 for split in SPLITS}
    for index in range(layer_size):
        quota[SPLITS[index]] = 1
    return quota


def _take_quota_slot(
    layer_quota: Mapping[str, int],
    cursor: int,
    counts: Mapping[str, int],
    targets: Mapping[str, int],
) -> str:
    preferred = SPLITS[cursor % len(SPLITS)]
    order = [preferred] + [
        SPLITS[(SPLITS.index(preferred) + offset) % len(SPLITS)]
        for offset in range(1, len(SPLITS))
    ]
    for candidate in order:
        if layer_quota[candidate] > 0 and counts[candidate] < targets[candidate]:
            layer_quota[candidate] -= 1
            return candidate
    # 全局配额已分时，退回层内仍有名额的集合，保证不丢家族。
    for candidate in order:
        if layer_quota[candidate] > 0:
            layer_quota[candidate] -= 1
            return candidate
    layer_quota[preferred] -= 1
    return preferred


def _deterministic_jitter(family: str, seed: int) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def prepare_assets(
    payload: Mapping[str, Any],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> PreparedAssets:
    """校验并整理本地资产，产出可追溯的划分与标签状态。"""
    payload = _require_mapping(payload)
    cutoff = parse_iso_date(payload.get("final_cutoff"))
    if cutoff is None:
        raise InputError("final_cutoff 必填且必须是 ISO 8601 日期")
    seed = int(payload.get("seed", 0))
    items = payload.get("items", [])
    labels = payload.get("labels", [])
    if not isinstance(items, list) or not isinstance(labels, list):
        raise InputError("items 与 labels 必须是数组")

    seen_item_ids: set[str] = set()
    families: dict[str, list[str]] = {}
    item_index: dict[str, Mapping[str, Any]] = {}
    withdrawn: list[str] = []
    for item in items:
        item_id = str(item.get("item_id", ""))
        if not item_id:
            raise InputError("item_id 不能为空")
        if item_id in seen_item_ids:
            raise InputError(f"item_id 重复: {item_id}")
        seen_item_ids.add(item_id)
        family = str(item.get("style_family", ""))
        if not family:
            raise InputError(f"{item_id} 缺少 style_family")
        item_index[item_id] = item
        withdrawn_at = parse_iso_date(item.get("withdrawn_at"))
        if withdrawn_at is not None and withdrawn_at <= cutoff:
            withdrawn.append(item_id)
            continue
        families.setdefault(family, []).append(item_id)

    unknown = {str(record.get("item_id")) for record in labels} - seen_item_ids
    if unknown:
        raise InputError(f"标注引用了不存在的商品: {sorted(unknown)}")

    states, dropped = build_label_states(
        [item for item in items if item["item_id"] not in set(withdrawn)],
        labels,
        cutoff,
    )
    family_splits = assign_family_splits(families, states, seed, ratios)
    item_splits = {
        item_id: family_splits[family]
        for family, member_ids in families.items()
        for item_id in member_ids
    }

    label_version = _label_version(payload, cutoff, labels, dropped)
    return PreparedAssets(
        cutoff=payload["final_cutoff"],
        label_version=label_version,
        items=item_index,
        families={family: tuple(sorted(member_ids)) for family, member_ids in families.items()},
        label_states=states,
        split_of_family=family_splits,
        split_of_item=item_splits,
        withdrawn_item_ids=tuple(sorted(withdrawn)),
        excluded_label_records=tuple(dropped),
    )


def _label_version(
    payload: Mapping[str, Any],
    cutoff: date,
    labels: Sequence[Mapping[str, Any]],
    dropped: Sequence[Mapping[str, Any]],
) -> str:
    from fashion_data_foundry.governance.asset_ledger import fingerprint

    eligible = sorted(
        (
            record
            for record in labels
            if (parse_iso_date(record.get("known_at")) is not None
                and parse_iso_date(record.get("known_at")) <= cutoff)
        ),
        key=lambda record: str(record.get("label_id")),
    )
    return f"{LABEL_VERSION_PREFIX}-{fingerprint([cutoff.isoformat(), eligible])[:12]}"


def split_manifest(prepared: PreparedAssets) -> dict[str, Any]:
    """输出发布物需要绑定的划分清单。"""
    manifest: dict[str, Any] = {
        split: prepared.items_in_split(split) for split in SPLITS
    }
    manifest["split_unit"] = "style_family"
    manifest["family_counts"] = {
        split: sum(1 for assigned in prepared.split_of_family.values() if assigned == split)
        for split in SPLITS
    }
    manifest["withdrawn_item_ids"] = list(prepared.withdrawn_item_ids)
    manifest["ratios"] = dict(zip(SPLITS, DEFAULT_RATIOS))
    from fashion_data_foundry.governance.asset_ledger import fingerprint

    manifest["split_manifest_fingerprint"] = fingerprint(
        {split: manifest[split] for split in SPLITS}
    )
    return manifest
