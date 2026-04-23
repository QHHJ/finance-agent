from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ROOT = _bootstrap_path()

TRAVEL_DOC_TYPES = {
    "transport_ticket",
    "transport_payment",
    "flight_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "unknown",
}

TRAVEL_SLOTS = {
    "go_ticket",
    "go_payment",
    "go_detail",
    "return_ticket",
    "return_payment",
    "return_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "unknown",
}


@dataclass(slots=True)
class DatasetRow:
    line_no: int
    sample_id: str
    batch_id: str
    batch_order: int
    file_path: str
    expected_doc_type: str
    expected_slot: str | None
    note: str


class LocalUploadedFile:
    def __init__(self, path: Path):
        self._path = path
        self.name = path.name
        self._bytes = path.read_bytes()
        self.size = len(self._bytes)

    def getvalue(self) -> bytes:
        return self._bytes


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(raw_path: str, dataset_path: Path) -> Path:
    candidate = Path(str(raw_path or "").strip())
    _assert(str(candidate), "file_path is required")
    if candidate.is_absolute():
        return candidate

    dataset_relative = (dataset_path.parent / candidate).resolve()
    if dataset_relative.exists():
        return dataset_relative
    return (ROOT / candidate).resolve()


def _normalize_slot(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text


def load_dataset(dataset_path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    raw_lines = dataset_path.read_text(encoding="utf-8-sig").splitlines()
    for idx, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("sample_id") or "").strip()
        _assert(sample_id, f"line {idx}: sample_id is required")

        file_path = _resolve_path(str(payload.get("file_path") or ""), dataset_path)
        _assert(file_path.exists(), f"line {idx}: file does not exist: {file_path}")

        expected_doc_type = str(payload.get("expected_doc_type") or "").strip()
        _assert(expected_doc_type in TRAVEL_DOC_TYPES, f"line {idx}: invalid expected_doc_type: {expected_doc_type}")

        expected_slot = _normalize_slot(payload.get("expected_slot"))
        if expected_slot is not None:
            _assert(expected_slot in TRAVEL_SLOTS, f"line {idx}: invalid expected_slot: {expected_slot}")

        batch_id = str(payload.get("batch_id") or sample_id).strip()
        _assert(batch_id, f"line {idx}: batch_id cannot be empty")

        batch_order_raw = payload.get("batch_order", payload.get("order", idx))
        try:
            batch_order = int(batch_order_raw)
        except (TypeError, ValueError):
            raise AssertionError(f"line {idx}: invalid batch_order: {batch_order_raw}") from None

        rows.append(
            DatasetRow(
                line_no=idx,
                sample_id=sample_id,
                batch_id=batch_id,
                batch_order=batch_order,
                file_path=str(file_path),
                expected_doc_type=expected_doc_type,
                expected_slot=expected_slot,
                note=str(payload.get("note") or "").strip(),
            )
        )
    _assert(rows, "dataset is empty")
    return rows


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    _assert(0 < percentile <= 100, f"invalid percentile: {percentile}")
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _build_latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": _round_or_none(sum(values) / len(values)),
        "median": _round_or_none(statistics.median(values)),
        "p95": _round_or_none(_nearest_rank_percentile(values, 95)),
        "max": _round_or_none(max(values)),
    }


def _build_classification_metrics(
    rows: list[dict[str, Any]],
    *,
    expected_key: str,
    predicted_key: str,
) -> dict[str, Any]:
    scored = [row for row in rows if row.get(expected_key) and row.get(predicted_key)]
    labels = sorted({str(row.get(expected_key)) for row in scored} | {str(row.get(predicted_key)) for row in scored})

    confusion: dict[str, dict[str, int]] = {}
    for expected in labels:
        confusion[expected] = {predicted: 0 for predicted in labels}
    for row in scored:
        expected = str(row.get(expected_key))
        predicted = str(row.get(predicted_key))
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

    total = len(scored)
    correct = sum(1 for row in scored if row.get(expected_key) == row.get(predicted_key))
    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []

    for label in labels:
        tp = sum(1 for row in scored if row.get(expected_key) == label and row.get(predicted_key) == label)
        fp = sum(1 for row in scored if row.get(expected_key) != label and row.get(predicted_key) == label)
        fn = sum(1 for row in scored if row.get(expected_key) == label and row.get(predicted_key) != label)
        support = sum(1 for row in scored if row.get(expected_key) == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _round_or_none(precision),
            "recall": _round_or_none(recall),
            "f1": _round_or_none(f1),
        }

    return {
        "scored_count": total,
        "correct_count": correct,
        "accuracy": _round_or_none((correct / total) if total > 0 else None),
        "macro_f1": _round_or_none((sum(f1_values) / len(f1_values)) if f1_values else None),
        "labels": labels,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _environment_snapshot() -> dict[str, Any]:
    keys = [
        "USE_OLLAMA_VL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_VL_MODEL",
        "OLLAMA_TEXT_MODEL",
        "OLLAMA_CHAT_MODEL",
        "RAG_BACKEND",
    ]
    return {key: os.getenv(key) for key in keys if os.getenv(key) is not None}


def run_benchmark(dataset_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    from app.runtime import init_runtime
    from app.usecases import travel_agent as travel_usecase
    import streamlit_app as travel_app

    init_runtime()
    rows = load_dataset(dataset_path)

    grouped: dict[str, list[DatasetRow]] = {}
    for row in rows:
        grouped.setdefault(row.batch_id, []).append(row)
    for batch_id in grouped:
        grouped[batch_id] = sorted(grouped[batch_id], key=lambda item: (item.batch_order, item.line_no, item.sample_id))

    sample_results: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for batch_id, batch_rows in grouped.items():
        uploaded_files = [LocalUploadedFile(Path(row.file_path)) for row in batch_rows]
        row_by_index = {index: row for index, row in enumerate(batch_rows)}
        profile_timings: dict[int, float] = {}
        profile_errors: dict[int, str] = {}

        def build_profile(uploaded_file: Any, index: int) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                profile = travel_app._recognize_travel_file(uploaded_file, index=index, retry_tag="benchmark")
                return profile
            except Exception as exc:
                profile_errors[index] = str(exc)
                raise
            finally:
                profile_timings[index] = (time.perf_counter() - started) * 1000.0

        batch_started = time.perf_counter()
        try:
            assignment, profiles = travel_usecase.organize_materials(
                uploaded_files,
                build_profile=build_profile,
            )
            batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0
        except Exception as exc:
            batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0
            error_message = str(exc)
            errors.append(
                {
                    "batch_id": batch_id,
                    "error": error_message,
                    "sample_ids": [row.sample_id for row in batch_rows],
                }
            )
            batch_results.append(
                {
                    "batch_id": batch_id,
                    "sample_count": len(batch_rows),
                    "elapsed_ms": _round_or_none(batch_elapsed_ms),
                    "status": "failed",
                    "error": error_message,
                }
            )
            for index, row in enumerate(batch_rows):
                sample_results.append(
                    {
                        "sample_id": row.sample_id,
                        "batch_id": row.batch_id,
                        "batch_order": row.batch_order,
                        "file_path": row.file_path,
                        "expected_doc_type": row.expected_doc_type,
                        "expected_slot": row.expected_slot,
                        "predicted_doc_type": None,
                        "predicted_slot": None,
                        "doc_type_correct": None,
                        "slot_correct": None,
                        "classification_ms": _round_or_none(profile_timings.get(index)),
                        "batch_elapsed_ms": _round_or_none(batch_elapsed_ms),
                        "source": None,
                        "confidence": None,
                        "amount": None,
                        "date": None,
                        "error": profile_errors.get(index) or error_message,
                        "note": row.note,
                    }
                )
            continue

        slot_map: dict[str, str] = {}
        for profile in profiles:
            slot_map[str(profile.get("profile_id") or "")] = str(profile.get("slot") or "unknown")

        batch_doc_correct = 0
        batch_slot_scored = 0
        batch_slot_correct = 0
        for index, (row, profile) in enumerate(zip(batch_rows, profiles)):
            predicted_doc_type = str(profile.get("doc_type") or "unknown")
            predicted_slot = str(profile.get("slot") or "unknown")
            doc_type_correct = predicted_doc_type == row.expected_doc_type
            slot_correct = None
            if row.expected_slot is not None:
                slot_correct = predicted_slot == row.expected_slot
                batch_slot_scored += 1
                if slot_correct:
                    batch_slot_correct += 1
            if doc_type_correct:
                batch_doc_correct += 1
            sample_results.append(
                {
                    "sample_id": row.sample_id,
                    "batch_id": row.batch_id,
                    "batch_order": row.batch_order,
                    "file_path": row.file_path,
                    "expected_doc_type": row.expected_doc_type,
                    "expected_slot": row.expected_slot,
                    "predicted_doc_type": predicted_doc_type,
                    "predicted_slot": predicted_slot,
                    "doc_type_correct": doc_type_correct,
                    "slot_correct": slot_correct,
                    "classification_ms": _round_or_none(profile_timings.get(index)),
                    "batch_elapsed_ms": _round_or_none(batch_elapsed_ms),
                    "source": str(profile.get("source") or ""),
                    "confidence": _round_or_none(_safe_float(profile.get("confidence"))),
                    "amount": _round_or_none(_safe_float(profile.get("amount"))),
                    "date": str(profile.get("date") or ""),
                    "error": None,
                    "note": row.note,
                }
            )

        batch_results.append(
            {
                "batch_id": batch_id,
                "sample_count": len(batch_rows),
                "elapsed_ms": _round_or_none(batch_elapsed_ms),
                "status": "ok",
                "doc_type_accuracy": _round_or_none(batch_doc_correct / len(batch_rows)) if batch_rows else None,
                "slot_accuracy": _round_or_none(batch_slot_correct / batch_slot_scored) if batch_slot_scored > 0 else None,
                "assignment": {
                    key: [getattr(item, "name", str(item)) for item in value] if isinstance(value, list) else value
                    for key, value in assignment.items()
                },
            }
        )

    doc_type_metrics = _build_classification_metrics(
        sample_results,
        expected_key="expected_doc_type",
        predicted_key="predicted_doc_type",
    )

    slot_rows = [row for row in sample_results if row.get("expected_slot") is not None]
    slot_metrics = _build_classification_metrics(
        slot_rows,
        expected_key="expected_slot",
        predicted_key="predicted_slot",
    ) if slot_rows else None

    classification_latencies = [
        float(row["classification_ms"])
        for row in sample_results
        if row.get("classification_ms") is not None and row.get("error") is None
    ]
    batch_latencies = [
        float(batch["elapsed_ms"])
        for batch in batch_results
        if batch.get("elapsed_ms") is not None and batch.get("status") == "ok"
    ]

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(dataset_path.resolve()),
        "environment": _environment_snapshot(),
        "summary": {
            "samples_total": len(sample_results),
            "samples_scored_doc_type": doc_type_metrics["scored_count"],
            "samples_scored_slot": slot_metrics["scored_count"] if slot_metrics else 0,
            "batches_total": len(batch_results),
            "batches_failed": sum(1 for batch in batch_results if batch.get("status") != "ok"),
            "doc_type_accuracy": doc_type_metrics["accuracy"],
            "doc_type_macro_f1": doc_type_metrics["macro_f1"],
            "slot_accuracy": slot_metrics["accuracy"] if slot_metrics else None,
            "slot_macro_f1": slot_metrics["macro_f1"] if slot_metrics else None,
            "latency_ms": {
                "classification": _build_latency_summary(classification_latencies),
                "batch_e2e": _build_latency_summary(batch_latencies),
            },
        },
        "doc_type_metrics": doc_type_metrics,
        "slot_metrics": slot_metrics,
        "batch_results": batch_results,
        "sample_results": sample_results,
        "errors": errors,
    }

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / "benchmark" / "travel" / "reports" / f"travel_e2e_report_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path.resolve())
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end travel document classification benchmark on labeled files.",
    )
    parser.add_argument(
        "dataset",
        help="Path to a JSONL dataset. Each line should contain sample_id, file_path, expected_doc_type, and optional batch_id/batch_order/expected_slot.",
    )
    parser.add_argument(
        "--output",
        help="Optional output report path. Defaults to benchmark/travel/reports/travel_e2e_report_<timestamp>.json",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    _assert(dataset_path.exists(), f"dataset not found: {dataset_path}")
    output_path = Path(args.output).resolve() if args.output else None

    report = run_benchmark(dataset_path=dataset_path, output_path=output_path)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "output_path": report["output_path"],
                "samples_total": summary["samples_total"],
                "batches_total": summary["batches_total"],
                "batches_failed": summary["batches_failed"],
                "doc_type_accuracy": summary["doc_type_accuracy"],
                "doc_type_macro_f1": summary["doc_type_macro_f1"],
                "slot_accuracy": summary["slot_accuracy"],
                "classification_latency_ms": summary["latency_ms"]["classification"],
                "batch_latency_ms": summary["latency_ms"]["batch_e2e"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("[OK] travel end-to-end benchmark finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
