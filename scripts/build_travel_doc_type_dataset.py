from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ROOT = _bootstrap_path()

FOLDER_TO_DOC_TYPE = {
    "go_ticket": "transport_ticket",
    "return_ticket": "transport_ticket",
    "go_payment": "transport_payment",
    "return_payment": "transport_payment",
    "go_detail": "flight_detail",
    "return_detail": "flight_detail",
    "hotel_invoice": "hotel_invoice",
    "hotel_payment": "hotel_payment",
    "hotel_order": "hotel_order",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_rows(source_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for folder in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        expected_doc_type = FOLDER_TO_DOC_TYPE.get(folder.name)
        if not expected_doc_type:
            continue

        files = sorted(path for path in folder.iterdir() if path.is_file())
        for index, file_path in enumerate(files, start=1):
            rows.append(
                {
                    "sample_id": f"travel_{folder.name}_{index:03d}",
                    "file_path": file_path.relative_to(ROOT).as_posix(),
                    "expected_doc_type": expected_doc_type,
                    "note": folder.name,
                }
            )
    return rows


def write_jsonl(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    output_path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a file-level travel doc_type benchmark dataset from folder-sorted images.",
    )
    parser.add_argument(
        "--source",
        default="test dataset",
        help="Source directory whose direct child folders are class labels.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/travel/test_dataset_doc_type.jsonl",
        help="Output JSONL path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    source_dir = (ROOT / args.source).resolve()
    output_path = (ROOT / args.output).resolve()

    _assert(source_dir.exists(), f"source directory not found: {source_dir}")
    _assert(source_dir.is_dir(), f"source is not a directory: {source_dir}")

    rows = build_rows(source_dir)
    _assert(rows, f"no labeled files found under: {source_dir}")
    write_jsonl(rows, output_path)

    counts = Counter(row["expected_doc_type"] for row in rows)
    print(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "output_path": str(output_path),
                "rows": len(rows),
                "doc_type_counts": dict(counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("[OK] travel doc_type dataset generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
