"""模型选型与复核队列主编排，并把发布依据写入治理账本。

发布物绑定四类版本，缺一项都不能复现：

* ``asset_revision``：输入资产版本；
* ``label_version``：截点内有效标签集合的指纹（迟到 / 撤回相关记录被剔除后计算）；
* ``split_manifest``：以 style_family 为单位的 train/validation/test 划分；
* ``feature_manifest``：仅在训练集拟合的特征处理参数与词表。

冻结的模型快照（model bundle）只包含拟合参数与类别表，旧版模型推断时从快照恢复，
不会重新读取当前标签，因此后续标签修订不会改变历史发布结果。
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fashion_data_foundry.governance.asset_ledger import (
    fingerprint,
    stable_json,
)
from fashion_data_foundry.model_review.evaluation import (
    EvaluatedCandidate,
    apply_abstention,
    evaluate_candidate,
    final_test,
    select_candidate,
    sufficiency_blockers,
)
from fashion_data_foundry.model_review.features import (
    FEATURE_SET_BASELINE,
    FEATURE_SET_FULL,
    FeatureProcessor,
)
from fashion_data_foundry.model_review.models import (
    CANDIDATE_LOGISTIC,
    CANDIDATE_MAJORITY,
    CANDIDATE_NAIVE_BAYES,
    GaussianNaiveBayesClassifier,
    LogisticRegressionClassifier,
    MajorityPriorClassifier,
    class_weights,
    model_from_bundle,
)
from fashion_data_foundry.model_review.preparation import (
    SPLITS,
    PreparedAssets,
    prepare_assets,
    split_manifest,
)
from fashion_data_foundry.model_review.review_queue import build_review_queue

RELEASE_STATUS_READY = "released"
RELEASE_STATUS_BLOCKED = "insufficient_data"
SCHEMA_VERSION = 1


class ModelReviewStore:
    """模型发布记录的本地 SQLite 账本，表结构与资产账本风格保持一致。"""

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_releases(
                    release_id TEXT PRIMARY KEY,
                    request_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    asset_revision TEXT NOT NULL,
                    label_version TEXT NOT NULL,
                    selected_candidate TEXT,
                    split_manifest TEXT NOT NULL,
                    feature_manifest TEXT NOT NULL,
                    report TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_review_events(
                    event_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_release(self, request_key: str, report: Mapping[str, Any]) -> str:
        release_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO model_releases(release_id,request_key,status,asset_revision,"
                "label_version,selected_candidate,split_manifest,feature_manifest,report,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    release_id,
                    request_key,
                    report["status"],
                    report["asset_revision"],
                    report["label_version"],
                    report.get("selected", {}).get("candidate_id") if report.get("selected") else None,
                    stable_json(report["split_manifest"]),
                    stable_json(report.get("feature_manifest", {})),
                    stable_json(report),
                    _utc_now(),
                ),
            )
            for event in report.get("events", []):
                connection.execute(
                    "INSERT OR IGNORE INTO model_review_events VALUES(?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        release_id,
                        event["stage"],
                        event["kind"],
                        stable_json(event["detail"]),
                        _utc_now(),
                    ),
                )
        return release_id

    def find_by_request_key(self, request_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT report FROM model_releases WHERE request_key=?", (request_key,)
            ).fetchone()
        if row is None:
            return None
        import json

        return json.loads(row["report"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_model_review(
    payload: Mapping[str, Any],
    database: str | Path,
    request_key: str | None = None,
) -> dict[str, Any]:
    """完整的选型、检验与复核队列流程，返回可审计的发布报告。"""
    store = ModelReviewStore(database)
    key = request_key or "model-review:" + fingerprint(
        [payload.get("asset_revision"), payload.get("final_cutoff"),
         payload.get("seed"), payload.get("items"), payload.get("labels")]
    )
    recalled = store.find_by_request_key(key)
    if recalled is not None:
        recalled["replayed"] = True
        return recalled

    prepared = prepare_assets(payload)
    events: list[dict[str, Any]] = [
        {"stage": "received", "kind": "run_started",
         "detail": {"asset_revision": payload.get("asset_revision"),
                    "final_cutoff": payload.get("final_cutoff")}}
    ]
    manifest = split_manifest(prepared)
    train_ids = prepared.labeled_items_in_split("train")
    validation_ids = prepared.labeled_items_in_split("validation")
    test_ids = prepared.labeled_items_in_split("test")
    train_labels = [prepared.label_states[item_id].hard_label for item_id in train_ids]
    validation_labels = [prepared.label_states[item_id].hard_label for item_id in validation_ids]
    test_labels = [prepared.label_states[item_id].hard_label for item_id in test_ids]
    classes = tuple(sorted({label for label in train_labels if label}))

    class_support = {label: train_labels.count(label) for label in classes}
    family_profile = {
        "smallest_family_size": min((len(members) for members in prepared.families.values()), default=0),
        "largest_family_size": max((len(members) for members in prepared.families.values()), default=0),
        "family_count": len(prepared.families),
    }
    blockers = sufficiency_blockers(train_labels, validation_labels, test_labels, classes)

    base_report = {
        "schema_version": SCHEMA_VERSION,
        "asset_revision": payload.get("asset_revision"),
        "label_version": prepared.label_version,
        "final_cutoff": prepared.cutoff,
        "status": RELEASE_STATUS_BLOCKED if blockers else RELEASE_STATUS_READY,
        "split_manifest": manifest,
        "data_profile": _data_profile(prepared, class_support),
        "excluded_assets": {
            "withdrawn_item_ids": list(prepared.withdrawn_item_ids),
            "excluded_label_records": [
                {"label_id": record.get("label_id"), "item_id": record.get("item_id"),
                 "known_at": record.get("known_at"), "reason": "after_final_cutoff"}
                for record in prepared.excluded_label_records
            ],
        },
        "events": events,
        "replayed": False,
    }
    if blockers:
        report = dict(base_report)
        report.update(
            {
                "candidates": [],
                "selected": None,
                "predictions": [],
                "review_queue": [],
                "feature_manifest": {},
                "blockers": blockers,
            }
        )
        events.append({"stage": "selection", "kind": "release_blocked", "detail": {"blockers": blockers}})
        store.save_release(key, report)
        return report

    full_processor = FeatureProcessor.fit(prepared, train_ids, include_numeric=True)
    baseline_processor = FeatureProcessor.fit(prepared, train_ids, include_numeric=False)
    train_rows_full = full_processor.transform(prepared, train_ids)
    validation_rows_full = full_processor.transform(prepared, validation_ids)
    test_rows_full = full_processor.transform(prepared, test_ids)
    train_rows_base = baseline_processor.transform(prepared, train_ids)
    validation_rows_base = baseline_processor.transform(prepared, validation_ids)
    test_rows_base = baseline_processor.transform(prepared, test_ids)
    weights = [
        class_weights(train_labels).get(label, 1.0) for label in train_labels
    ]

    candidates = [
        _fit_candidate(
            CANDIDATE_MAJORITY, FEATURE_SET_BASELINE, {},
            MajorityPriorClassifier(), train_rows_base, train_labels, weights,
            classes, validation_rows_base, validation_labels,
        ),
        _fit_candidate(
            CANDIDATE_LOGISTIC, FEATURE_SET_FULL,
            {"learning_rate": 0.15, "l2_penalty": 0.02, "max_iterations": 400,
             "class_weight": "inverse_frequency"},
            LogisticRegressionClassifier(), train_rows_full, train_labels, weights,
            classes, validation_rows_full, validation_labels,
        ),
        _fit_candidate(
            CANDIDATE_NAIVE_BAYES, FEATURE_SET_FULL,
            {"var_smoothing": 1e-9, "class_weight": "inverse_frequency"},
            GaussianNaiveBayesClassifier(), train_rows_full, train_labels, weights,
            classes, validation_rows_full, validation_labels,
        ),
    ]
    events.append({"stage": "selection", "kind": "candidates_trained",
                   "detail": {"candidate_ids": [item.candidate_id for item in candidates]}})

    selected_id, selection_notes = select_candidate(
        candidates, class_support, family_profile
    )
    selected = next(item for item in candidates if item.candidate_id == selected_id)
    selected_model = model_from_bundle(selected.model_bundle)
    selected_feature_set = selected.feature_set
    processor = full_processor if selected_feature_set == FEATURE_SET_FULL else baseline_processor
    test_rows = test_rows_full if selected_feature_set == FEATURE_SET_FULL else test_rows_base
    test_metrics = final_test(selected, selected_model, classes, test_rows, test_labels)
    events.append({"stage": "test", "kind": "final_evaluation",
                   "detail": {"candidate_id": selected_id, "test_metrics": test_metrics}})

    predictions = _build_predictions(prepared, selected, selected_model, processor)
    queue = build_review_queue(
        prepared,
        predictions,
        int(payload.get("review_budget", 0)),
        payload.get("confirmed_item_ids", []),
        class_support,
    )
    events.append({"stage": "review", "kind": "queue_built",
                   "detail": {"budget": payload.get("review_budget"), "queued": len(queue)}})

    release_bundle = {
        "release_id": uuid.uuid4().hex,
        "model_bundle": selected.model_bundle,
        "feature_manifest": processor.manifest(),
        "label_version": prepared.label_version,
        "asset_revision": payload.get("asset_revision"),
        "split_manifest_fingerprint": manifest["split_manifest_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "frozen_at": _utc_now(),
    }
    report = dict(base_report)
    report.update(
        {
            "candidates": [_candidate_payload(item) for item in candidates],
            "selected": {
                "candidate_id": selected_id,
                "feature_set": selected_feature_set,
                "selection_notes": selection_notes + selected.selection_rationale,
                "validation": selected.validation,
                "test": selected.test,
                "release_bundle": release_bundle,
            },
            "predictions": predictions,
            "review_queue": queue,
            "feature_manifest": processor.manifest(),
            "blockers": [],
            "review_budget": int(payload.get("review_budget", 0)),
        }
    )
    store.save_release(key, report)
    return report


def _fit_candidate(
    candidate_id: str,
    feature_set: str,
    hyperparameters: Mapping[str, Any],
    model: Any,
    train_rows: Sequence[Sequence[float]],
    train_labels: Sequence[str],
    weights: Sequence[float],
    classes: Sequence[str],
    validation_rows: Sequence[Sequence[float]],
    validation_labels: Sequence[str],
) -> EvaluatedCandidate:
    model.fit(train_rows, train_labels, weights if candidate_id != CANDIDATE_MAJORITY else None)
    return evaluate_candidate(
        candidate_id, feature_set, hyperparameters, model, classes,
        validation_rows, validation_labels,
    )


def _build_predictions(
    prepared: PreparedAssets,
    selected: EvaluatedCandidate,
    model: Any,
    processor: FeatureProcessor,
) -> list[dict[str, Any]]:
    threshold = selected.validation["abstention_threshold"]
    predictions: list[dict[str, Any]] = []
    for item_id in sorted(prepared.label_states):
        state = prepared.label_states[item_id]
        row = processor.transform_row(prepared.items[item_id]["features"])
        vector = [row[name] for name in processor.feature_names]
        decision = apply_abstention([model.predict_proba([vector])[0]], threshold)[0]
        predictions.append(
            {
                "item_id": item_id,
                "family": state.family,
                "split": prepared.split_of_item[item_id],
                "label": decision["label"],
                "abstained": decision["abstained"],
                "confidence": decision["confidence"],
                "probabilities": decision["probabilities"],
                "consensus_hard_label": state.hard_label,
                "consensus_uncertain": state.uncertain,
                "uncertainty_reason": state.uncertainty_reason,
                "vote_distribution": state.votes,
            }
        )
    return predictions


def _candidate_payload(candidate: EvaluatedCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set,
        "hyperparameters": candidate.hyperparameters,
        "validation": candidate.validation,
        "test": candidate.test,
        "abstention_coverage": {
            "validation_coverage": candidate.validation["coverage"],
            "validation_abstention_rate": candidate.validation["abstention_rate"],
            "test_coverage": candidate.test["coverage"] if candidate.test else None,
            "test_abstention_rate": candidate.test["abstention_rate"] if candidate.test else None,
        },
        "selection_rationale": candidate.selection_rationale,
    }


def _data_profile(
    prepared: PreparedAssets, class_support: Mapping[str, int]
) -> dict[str, Any]:
    uncertain = [
        item_id for item_id, state in prepared.label_states.items() if state.uncertain
    ]
    return {
        "item_count": len(prepared.items),
        "eligible_item_count": len(prepared.label_states),
        "withdrawn_count": len(prepared.withdrawn_item_ids),
        "family_count": len(prepared.families),
        "uncertain_item_ids": sorted(uncertain),
        "train_class_support": dict(sorted(class_support.items())),
        "class_imbalance_ratio": round(
            max(class_support.values()) / max(min(class_support.values()), 1), 3
        ) if class_support else None,
        "split_sizes": {split: len(prepared.items_in_split(split)) for split in SPLITS},
    }


def predict_with_frozen_bundle(
    bundle: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    feature_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """用历史发布快照推断：只依赖冻结模型与冻结特征处理，不读取任何标签。"""
    model = model_from_bundle(bundle["model_bundle"])
    processor = _restore_processor(feature_manifest)
    results: dict[str, Any] = {}
    for item_id, features in feature_rows.items():
        row = processor.transform_row(features)
        vector = [row[name] for name in processor.feature_names]
        results[item_id] = model.predict_proba([vector])[0]
    return {
        "release_id": bundle.get("release_id"),
        "asset_revision": bundle.get("asset_revision"),
        "label_version_at_release": bundle.get("label_version"),
        "predictions": results,
    }


def _restore_processor(manifest: Mapping[str, Any]) -> FeatureProcessor:
    include_numeric = bool(manifest.get("numeric_columns"))
    processor = FeatureProcessor(include_numeric=include_numeric)
    if include_numeric:
        for column, spec in manifest["numeric_specs"].items():
            from fashion_data_foundry.model_review.features import NumericSpec

            processor.numeric_specs[column] = NumericSpec(
                mean=float(spec["mean"]), std=float(spec["std"]),
                fill_value=float(spec["fill_value"]),
            )
        processor.numeric_columns = tuple(manifest["numeric_columns"])
    processor.categorical_columns = tuple(manifest["categorical_columns"])
    processor.categorical_vocab = {
        column: tuple(values) for column, values in manifest["categorical_vocab"].items()
    }
    processor.feature_names = tuple(manifest["feature_names"])
    processor.version = manifest["feature_version"]
    return processor
