"""商品特征处理：只允许从训练集学习处理参数。

数值列按训练集均值/标准差标准化，缺失值先以训练集均值填补并单独记录缺失指示位；
类别列做训练集词表内的 one-hot，未见取值落到 ``<unknown>`` 桶，避免验证/测试信息泄漏。
特征处理结果带版本指纹，发布时与资产版本、标签版本、划分清单一并绑定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fashion_data_foundry.governance.asset_ledger import fingerprint
from fashion_data_foundry.model_review.preparation import PreparedAssets

NUMERIC_COLUMNS = ("weight", "price")
CATEGORICAL_COLUMNS = ("fabric",)
MISSING_INDICATOR_SUFFIX = "_missing"
UNKNOWN_BUCKET = "<unknown>"
FEATURE_SET_FULL = "full_features"
FEATURE_SET_BASELINE = "baseline_features"


@dataclass(frozen=True, slots=True)
class NumericSpec:
    mean: float
    std: float
    fill_value: float


class FeatureProcessor:
    """在训练集上拟合的确定性特征处理器。"""

    def __init__(self, numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
                 categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
                 include_numeric: bool = True) -> None:
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.include_numeric = include_numeric
        self.numeric_specs: dict[str, NumericSpec] = {}
        self.categorical_vocab: dict[str, tuple[str, ...]] = {}
        self.feature_names: tuple[str, ...] = ()
        self.version: str = ""

    @classmethod
    def fit(
        cls,
        prepared: PreparedAssets,
        train_item_ids: Sequence[str],
        include_numeric: bool = True,
    ) -> "FeatureProcessor":
        processor = cls(include_numeric=include_numeric)
        rows = [prepared.items[item_id]["features"] for item_id in train_item_ids]
        if include_numeric:
            for column in processor.numeric_columns:
                values = [_finite_number(row.get(column)) for row in rows]
                observed = [value for value in values if value is not None]
                if not observed:
                    raise ValueError(f"训练集数值列 {column} 全部缺失，无法拟合特征处理")
                fill_value = sum(observed) / len(observed)
                variance = sum((value - fill_value) ** 2 for value in observed) / len(observed)
                std = math.sqrt(variance) if variance > 0 else 1.0
                processor.numeric_specs[column] = NumericSpec(
                    mean=fill_value, std=std, fill_value=fill_value
                )
        for column in processor.categorical_columns:
            vocabulary = sorted(
                {
                    str(row[column])
                    for row in rows
                    if row.get(column) not in (None, "")
                }
            )
            processor.categorical_vocab[column] = tuple(vocabulary + [UNKNOWN_BUCKET])
        processor.feature_names = tuple(processor._build_names())
        processor.version = "feature-" + fingerprint(
            {
                "numeric_columns": processor.numeric_columns if include_numeric else (),
                "categorical_columns": processor.categorical_columns,
                "numeric_specs": [
                    [column, spec.mean, spec.std, spec.fill_value]
                    for column, spec in sorted(processor.numeric_specs.items())
                ],
                "categorical_vocab": {
                    column: list(values)
                    for column, values in sorted(processor.categorical_vocab.items())
                },
            }
        )[:12]
        return processor

    def _build_names(self) -> list[str]:
        names: list[str] = []
        if self.include_numeric:
            for column in self.numeric_columns:
                names.append(column)
                names.append(column + MISSING_INDICATOR_SUFFIX)
        for column in self.categorical_columns:
            for bucket in self.categorical_vocab[column]:
                names.append(f"{column}={bucket}")
        return names

    def transform_row(self, features: Mapping[str, Any]) -> dict[str, float]:
        vector: dict[str, float] = {}
        if self.include_numeric:
            for column in self.numeric_columns:
                spec = self.numeric_specs[column]
                value = _finite_number(features.get(column))
                if value is None:
                    vector[column] = 0.0
                    vector[column + MISSING_INDICATOR_SUFFIX] = 1.0
                else:
                    vector[column] = (value - spec.mean) / spec.std
                    vector[column + MISSING_INDICATOR_SUFFIX] = 0.0
        for column in self.categorical_columns:
            raw = features.get(column)
            bucket = str(raw) if raw not in (None, "") else UNKNOWN_BUCKET
            vocabulary = self.categorical_vocab[column]
            if bucket not in vocabulary:
                bucket = UNKNOWN_BUCKET
            for candidate in vocabulary:
                vector[f"{column}={candidate}"] = 1.0 if candidate == bucket else 0.0
        return vector

    def transform(self, prepared: PreparedAssets, item_ids: Sequence[str]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for item_id in item_ids:
            row = self.transform_row(prepared.items[item_id]["features"])
            matrix.append([row[name] for name in self.feature_names])
        return matrix

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_set": FEATURE_SET_FULL if self.include_numeric else FEATURE_SET_BASELINE,
            "feature_version": self.version,
            "numeric_columns": list(self.numeric_columns) if self.include_numeric else [],
            "categorical_columns": list(self.categorical_columns),
            "missing_value_policy": {
                "numeric": "train_mean_impute_with_indicator",
                "categorical": UNKNOWN_BUCKET,
            },
            "scaling": "train_zscore",
            "vocabulary_fit": "train_only",
            "feature_names": list(self.feature_names),
            "numeric_specs": {
                column: {
                    "mean": round(spec.mean, 6),
                    "std": round(spec.std, 6),
                    "fill_value": round(spec.fill_value, 6),
                }
                for column, spec in sorted(self.numeric_specs.items())
            },
            "categorical_vocab": {
                column: list(values)
                for column, values in sorted(self.categorical_vocab.items())
            },
        }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
