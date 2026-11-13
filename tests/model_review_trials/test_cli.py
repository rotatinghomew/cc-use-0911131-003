from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_FIXTURE = ROOT / "fixtures" / "analysis" / "input.json"
SMALL_FIXTURE = ROOT / "fixtures" / "analysis" / "insufficient_input.json"


def run_cli(database: Path, input_path: Path, output_path: Path) -> subprocess.CompletedProcess:
    # -I 隔离模式下显式注入仓库路径，避免现场目录中的同名模块被导入。
    bootstrap = (
        "import sys; sys.path.insert(0, %r); "
        "from fashion_data_foundry.model_review.__main__ import main; "
        "raise SystemExit(main(['--database', %r, '--input', %r, '--output', %r]))"
        % (str(ROOT), str(database), str(input_path), str(output_path))
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", bootstrap],
        capture_output=True,
        text=True,
        check=False,
    )


class ModelReviewCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_cli_releases_main_fixture(self) -> None:
        directory = Path(self.temp.name)
        result = run_cli(
            directory / "main.sqlite3", MAIN_FIXTURE, directory / "main-report.json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((directory / "main-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "released")
        self.assertEqual(len(report["candidates"]), 3)
        self.assertEqual(len(report["review_queue"]), report["review_budget"])
        self.assertTrue((directory / "main.sqlite3").is_file())

    def test_cli_marks_insufficient_fixture_as_blocked(self) -> None:
        directory = Path(self.temp.name)
        result = run_cli(
            directory / "small.sqlite3", SMALL_FIXTURE, directory / "small-report.json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((directory / "small-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "insufficient_data")
        self.assertTrue(report["blockers"])
        self.assertIsNone(report["selected"])
        self.assertEqual(report["review_queue"], [])

    def test_cli_rejects_missing_input(self) -> None:
        directory = Path(self.temp.name)
        result = run_cli(
            directory / "x.sqlite3", directory / "missing.json", directory / "out.json"
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
