"""模型选型与复核队列入口。

用法::

    python3 -m fashion_data_foundry.model_review \
        --database <DB> --input <JSON> --output <JSON>

输入输出均为本地文件，运行过程不访问网络；结果同时写入 SQLite 治理账本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fashion_data_foundry.governance.asset_ledger import DomainError
from fashion_data_foundry.model_review.preparation import InputError
from fashion_data_foundry.model_review.release import run_model_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m fashion_data_foundry.model_review",
        description="风格标签模型选型、家族不相交检验与人工复核队列",
    )
    parser.add_argument("--database", required=True, help="本地 SQLite 账本路径")
    parser.add_argument("--input", required=True, help="本地资产输入 JSON")
    parser.add_argument("--output", required=True, help="发布报告输出 JSON")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"输入文件无法读取或不是合法 JSON: {exc}", file=sys.stderr)
        return 2
    try:
        report = run_model_review(payload, args.database)
    except (InputError, DomainError, ValueError) as exc:
        print(f"模型选型中止: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "asset_revision": report["asset_revision"],
                "label_version": report["label_version"],
                "output": str(output_path),
                "candidates": len(report.get("candidates", [])),
                "review_queue": len(report.get("review_queue", [])),
                "blockers": report.get("blockers", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
