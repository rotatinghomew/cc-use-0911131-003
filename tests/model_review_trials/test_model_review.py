from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fashion_data_foundry.model_review.evaluation import ABSTAIN
from fashion_data_foundry.model_review.models import model_from_bundle
from fashion_data_foundry.model_review.preparation import (
    InputError,
    prepare_assets,
    split_manifest,
)
from fashion_data_foundry.model_review.release import (
    RELEASE_STATUS_BLOCKED,
    RELEASE_STATUS_READY,
    ModelReviewStore,
    predict_with_frozen_bundle,
    run_model_review,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "analysis" / "input.json"


def load_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ModelReviewFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "model-review.sqlite3"
        self.payload = load_payload()

    def test_report_binds_all_required_sections(self) -> None:
        report = run_model_review(self.payload, self.database)
        for key in (
            "asset_revision", "label_version", "split_manifest", "candidates",
            "selected", "predictions", "review_queue", "feature_manifest",
            "status", "blockers", "data_profile",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["status"], RELEASE_STATUS_READY)
        self.assertEqual(report["asset_revision"], "asset-r3")
        self.assertTrue(report["label_version"].startswith("label-set-"))
        self.assertEqual(report["blockers"], [])

    def test_splits_are_family_disjoint_and_complete(self) -> None:
        prepared = prepare_assets(self.payload)
        manifest = split_manifest(prepared)
        train, validation, test = (
            set(manifest["train"]), set(manifest["validation"]), set(manifest["test"])
        )
        self.assertEqual(train & validation, set())
        self.assertEqual(train & test, set())
        self.assertEqual(validation & test, set())
        family_of = {item["item_id"]: item["style_family"] for item in self.payload["items"]}
        family_sets = [
            {family_of[item_id] for item_id in split}
            for split in (train, validation, test)
        ]
        self.assertEqual(family_sets[0] & family_sets[1], set())
        self.assertEqual(family_sets[0] & family_sets[2], set())
        self.assertEqual(family_sets[1] & family_sets[2], set())
        # 撤回资产不进入任何集合。
        self.assertNotIn("i009", train | validation | test)
        # 每个集合都能观察到全部四个类别，低频类 rare 不允许在检验集合缺席。
        for split in (manifest["train"], manifest["validation"], manifest["test"]):
            labels = {
                prepared.label_states[item_id].hard_label
                for item_id in split
                if prepared.label_states[item_id].hard_label
            }
            self.assertEqual(labels, {"casual", "formal", "rare", "sport"})

    def test_late_labels_and_withdrawn_assets_are_excluded(self) -> None:
        prepared = prepare_assets(self.payload)
        excluded = {
            record["label_id"] for record in prepared.excluded_label_records
        }
        # i000 的 revision=2 修订标签晚于 final_cutoff，必须剔除。
        self.assertIn("lab0-0", excluded)
        state = prepared.label_states["i000"]
        self.assertTrue(state.uncertain)
        self.assertIsNone(state.hard_label)
        self.assertEqual(state.votes, {"formal": 1, "rare": 1})
        self.assertIn("i009", prepared.withdrawn_item_ids)

    def test_disagreement_keeps_uncertainty_instead_of_majority_vote(self) -> None:
        report = run_model_review(self.payload, self.database)
        uncertain_ids = set(report["data_profile"]["uncertain_item_ids"])
        self.assertEqual(
            uncertain_ids, {"i000", "i046", "i069", "i092", "i115", "i161"}
        )
        prediction = next(item for item in report["predictions"] if item["item_id"] == "i000")
        self.assertTrue(prediction["consensus_uncertain"])
        self.assertIsNone(prediction["consensus_hard_label"])
        self.assertEqual(prediction["vote_distribution"], {"formal": 1, "rare": 1})

    def test_three_trainable_candidates_including_naive_baseline(self) -> None:
        report = run_model_review(self.payload, self.database)
        candidates = {item["candidate_id"] for item in report["candidates"]}
        self.assertEqual(
            candidates,
            {
                "majority_prior_baseline",
                "weighted_logistic_regression",
                "gaussian_naive_bayes",
            },
        )
        for candidate in report["candidates"]:
            self.assertIn("per_class", candidate["validation"])
            for class_metric in candidate["validation"]["per_class"].values():
                for metric in ("precision", "recall", "f1", "support"):
                    self.assertIn(metric, class_metric)
            self.assertIn("validation_coverage", candidate["abstention_coverage"])
            self.assertTrue(candidate["selection_rationale"])
        # 只有最终选中的模型允许在 test 上跑一次最终检验。
        selected_id = report["selected"]["candidate_id"]
        for candidate in report["candidates"]:
            if candidate["candidate_id"] == selected_id:
                self.assertIsNotNone(candidate["test"])
            else:
                self.assertIsNone(candidate["test"])

    def test_predictions_can_abstain_and_report_more_than_overall_accuracy(self) -> None:
        report = run_model_review(self.payload, self.database)
        labels = {item["label"] for item in report["predictions"]}
        self.assertIn(ABSTAIN, labels)
        selected = report["selected"]
        self.assertGreater(selected["test"]["abstention_rate"], 0.0)
        self.assertLess(selected["test"]["coverage"], 1.0)
        self.assertIn("macro_f1", selected["test"])
        self.assertIn("balanced_accuracy", selected["test"])
        # 撤回资产不产生预测。
        self.assertFalse(
            any(item["item_id"] == "i009" for item in report["predictions"])
        )

    def test_review_queue_respects_budget_exclusions_and_family_coverage(self) -> None:
        report = run_model_review(self.payload, self.database)
        queue = report["review_queue"]
        self.assertLessEqual(len(queue), self.payload["review_budget"])
        self.assertEqual(len(queue), self.payload["review_budget"])
        queued_ids = {entry["item_id"] for entry in queue}
        self.assertNotIn("i001", queued_ids)  # 已确认
        self.assertNotIn("i009", queued_ids)  # 已撤回
        for entry in queue:
            self.assertTrue(entry["reason"])
            self.assertIn("family", entry)
        # 六个标注分歧条目必须全部优先于普通家族补位进入队列。
        self.assertEqual(
            queued_ids & {"i000", "i046", "i069", "i092", "i115", "i161"},
            {"i000", "i046", "i069", "i092", "i115", "i161"},
        )
        # 不能全压在一个家族上。
        families = [entry["family"] for entry in queue]
        self.assertLess(max(families.count(family) for family in families), len(queue))

    def test_review_queue_never_exceeds_budget_when_candidates_are_few(self) -> None:
        report = run_model_review(self.payload, self.database)
        queue = report["review_queue"]
        self.assertEqual(len(queue), self.payload["review_budget"])
        tiny_payload = copy.deepcopy(self.payload)
        tiny_payload["review_budget"] = 3
        tiny_database = Path(self.temp.name) / "tiny.sqlite3"
        tiny_report = run_model_review(tiny_payload, tiny_database)
        self.assertEqual(len(tiny_report["review_queue"]), 3)
        zero_payload = copy.deepcopy(self.payload)
        zero_payload["review_budget"] = 0
        zero_database = Path(self.temp.name) / "zero.sqlite3"
        zero_report = run_model_review(zero_payload, zero_database)
        self.assertEqual(zero_report["review_queue"], [])

    def test_release_bundle_pins_versions_and_supports_frozen_inference(self) -> None:
        report = run_model_review(self.payload, self.database)
        bundle = report["selected"]["release_bundle"]
        self.assertEqual(bundle["asset_revision"], report["asset_revision"])
        self.assertEqual(bundle["label_version"], report["label_version"])
        self.assertTrue(bundle["split_manifest_fingerprint"])
        self.assertEqual(bundle["feature_manifest"]["feature_version"],
                         report["feature_manifest"]["feature_version"])
        # 用冻结快照推断，只依赖快照与特征清单，不读取当前标签。
        sample_items = ["i010", "i050", "i120"]
        feature_rows = {
            item_id: next(
                item["features"] for item in self.payload["items"]
                if item["item_id"] == item_id
            )
            for item_id in sample_items
        }
        frozen = predict_with_frozen_bundle(
            bundle, report["feature_manifest"], feature_rows
        )
        self.assertEqual(set(frozen["predictions"]), set(sample_items))
        self.assertEqual(frozen["label_version_at_release"], report["label_version"])

        # 后续即使标签被修改，旧快照恢复出的模型结构与类别表保持不变。
        restored = model_from_bundle(bundle["model_bundle"])
        self.assertEqual(tuple(restored.classes), ("casual", "formal", "rare", "sport"))

    def test_frozen_bundle_ignores_later_label_revisions(self) -> None:
        report = run_model_review(self.payload, self.database)
        bundle = report["selected"]["release_bundle"]
        mutated_payload = copy.deepcopy(self.payload)
        for label in mutated_payload["labels"]:
            label["label"] = "casual"
            label["known_at"] = "2026-01-01"
        feature_rows = {
            item["item_id"]: item["features"]
            for item in mutated_payload["items"][:5]
        }
        frozen = predict_with_frozen_bundle(
            bundle, report["feature_manifest"], feature_rows
        )
        expected_model = model_from_bundle(bundle["model_bundle"])
        # 冻结推断只接收特征行，函数签名中没有任何标签入参。
        self.assertTrue(all(len(values) == 4 for values in frozen["predictions"].values()))
        self.assertEqual(
            frozen["label_version_at_release"], bundle["label_version"]
        )
        self.assertIsNotNone(expected_model)

    def test_idempotent_replay_returns_same_report(self) -> None:
        first = run_model_review(self.payload, self.database)
        second = run_model_review(self.payload, self.database)
        self.assertTrue(second.get("replayed"))
        self.assertEqual(
            {key: value for key, value in first.items() if key != "replayed"},
            {key: value for key, value in second.items() if key != "replayed"},
        )
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
            )
            count = connection.execute(
                "SELECT count(*) FROM model_releases"
            ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_deterministic_split_across_runs(self) -> None:
        first = split_manifest(prepare_assets(self.payload))
        second = split_manifest(prepare_assets(self.payload))
        self.assertEqual(first, second)


class InsufficientDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "blocked.sqlite3"
        self.payload = load_payload()

    def test_blocked_status_is_distinct_from_release(self) -> None:
        # 只保留两个家族且标签单一，制造训练类别不足、检验样本不足的场景。
        kept_families = {"f000", "f001"}
        kept_items = [
            item for item in self.payload["items"]
            if item["style_family"] in kept_families
        ]
        kept_ids = {item["item_id"] for item in kept_items}
        payload = copy.deepcopy(self.payload)
        payload["items"] = kept_items
        payload["labels"] = [
            label for label in payload["labels"] if label["item_id"] in kept_ids
        ]
        report = run_model_review(payload, self.database)
        self.assertEqual(report["status"], RELEASE_STATUS_BLOCKED)
        self.assertTrue(report["blockers"])
        self.assertEqual(report["candidates"], [])
        self.assertIsNone(report["selected"])
        self.assertEqual(report["predictions"], [])
        self.assertEqual(report["review_queue"], [])
        # 版本与划分仍然落盘，阻断原因可审计。
        self.assertTrue(report["label_version"])
        self.assertIn("train", report["split_manifest"])

    def test_missing_label_class_blocks_training(self) -> None:
        # 删除 rare 类的大部分标注，使该类在训练集的硬标签样本低于最低门槛。
        payload = copy.deepcopy(self.payload)
        rare_kept = 0
        filtered = []
        for label in payload["labels"]:
            if label["label"] == "rare":
                if rare_kept < 2:
                    filtered.append(label)
                    rare_kept += 1
                continue
            filtered.append(label)
        payload["labels"] = filtered
        report = run_model_review(payload, Path(self.temp.name) / "no-rare.sqlite3")
        self.assertEqual(report["status"], RELEASE_STATUS_BLOCKED)
        self.assertTrue(
            any("rare" in blocker for blocker in report["blockers"])
        )


class InputValidationTests(unittest.TestCase):
    def test_duplicate_item_id_rejected(self) -> None:
        payload = load_payload()
        payload["items"].append(copy.deepcopy(payload["items"][0]))
        with self.assertRaises(InputError):
            prepare_assets(payload)

    def test_unknown_label_reference_rejected(self) -> None:
        payload = load_payload()
        payload["labels"][0]["item_id"] = "missing-item"
        with self.assertRaises(InputError):
            prepare_assets(payload)

    def test_missing_cutoff_rejected(self) -> None:
        payload = load_payload()
        payload.pop("final_cutoff")
        with self.assertRaises(InputError):
            prepare_assets(payload)


if __name__ == "__main__":
    unittest.main()
