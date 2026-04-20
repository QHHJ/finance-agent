from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from . import extractor, rag_retriever

ROW_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]


def _text_model() -> str:
    return os.getenv("OLLAMA_TEXT_MODEL") or os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).replace(",", "").replace("，", "").replace("¥", "").replace("￥", "").strip()
    text = re.sub(r"[^\d.\-]", "", text)
    if text in {"", ".", "-", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    normalizer = getattr(extractor, "_normalize_line_item_row", None)
    if callable(normalizer):
        try:
            normalized = normalizer(dict(row or {}))
            return {field: str(normalized.get(field) or "").strip() for field in ROW_FIELDS}
        except Exception:
            pass
    return {field: str((row or {}).get(field) or "").strip() for field in ROW_FIELDS}


def _chunk_rows(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    chunk_size = max(1, int(size))
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _compact(text: str) -> str:
    return re.sub(r"[\s\-_/,，;；:：()（）\[\]【】]+", "", text or "")


def _row_similarity(a: dict[str, str], b: dict[str, str]) -> float:
    score = 0.0
    if _compact(a.get("item_name", "")) and _compact(a.get("item_name", "")) == _compact(b.get("item_name", "")):
        score += 0.55
    if _compact(a.get("spec", "")) and _compact(a.get("spec", "")) == _compact(b.get("spec", "")):
        score += 0.25
    if _compact(a.get("unit", "")) and _compact(a.get("unit", "")) == _compact(b.get("unit", "")):
        score += 0.10
    a_total = _to_float(a.get("line_total_with_tax"))
    b_total = _to_float(b.get("line_total_with_tax"))
    if a_total is not None and b_total is not None and abs(a_total - b_total) <= 0.02:
        score += 0.10
    return score


def _build_row_query(
    row: dict[str, str],
    *,
    header_context: dict[str, Any] | None = None,
    raw_text: str = "",
) -> str:
    header = dict(header_context or {})
    parts = [
        f"bill_type: {header.get('bill_type') or ''}",
        f"item_content: {header.get('item_content') or ''}",
        f"seller: {header.get('seller') or ''}",
        f"buyer: {header.get('buyer') or ''}",
        f"invoice_amount: {header.get('amount') or ''}",
        f"row_item_name: {row.get('item_name') or ''}",
        f"row_spec: {row.get('spec') or ''}",
        f"row_quantity: {row.get('quantity') or ''}",
        f"row_unit: {row.get('unit') or ''}",
        f"row_total_with_tax: {row.get('line_total_with_tax') or ''}",
        f"ocr_context: {str(raw_text or '')[:320]}",
    ]
    return "\n".join(parts)


def _retrieve_case_examples(
    row: dict[str, str],
    *,
    header_context: dict[str, Any] | None = None,
    raw_text: str = "",
    top_k: int = 2,
) -> list[dict[str, Any]]:
    query = _build_row_query(row, header_context=header_context, raw_text=raw_text)
    hits = rag_retriever.retrieve_material_fix_case_hits(query=query, top_k=max(1, top_k))
    examples: list[dict[str, Any]] = []
    for hit in hits[: max(1, top_k)]:
        meta = dict(hit.get("metadata") or {})
        before_row = dict(meta.get("before_row") or {})
        after_row = dict(meta.get("after_row") or {})
        before_norm = _normalize_row(before_row) if before_row else {}
        if before_norm and _row_similarity(row, before_norm) < 0.25:
            continue
        examples.append(
            {
                "score": float(hit.get("score") or 0.0),
                "risk_tags": list(meta.get("risk_tags") or []),
                "before_row": before_norm or before_row,
                "after_row": _normalize_row(after_row) if after_row else after_row,
                "brief": str(hit.get("content") or "")[:220],
            }
        )
    return examples


def _sanitize_suggested_row(suggested: dict[str, Any], base_row: dict[str, str]) -> dict[str, str]:
    candidate = dict(base_row)
    normalized = _normalize_row(suggested)
    for key in ROW_FIELDS:
        new_value = str(normalized.get(key) or "").strip()
        if not new_value:
            continue
        if key == "item_name" and len(new_value) < 2:
            continue
        if key in {"quantity", "line_total_with_tax"} and _to_float(new_value) is None:
            continue
        candidate[key] = new_value
    # Keep one hard safety line: do not erase item name/spec to empty.
    if not candidate.get("item_name"):
        candidate["item_name"] = base_row.get("item_name", "")
    return _normalize_row(candidate)


def _call_llm_chunk(
    row_packets: list[dict[str, Any]],
    *,
    header_context: dict[str, Any] | None = None,
    model: str,
    base_url: str,
    timeout: int,
) -> list[dict[str, Any]]:
    packet_text = json.dumps(row_packets, ensure_ascii=False)
    head_text = json.dumps(dict(header_context or {}), ensure_ascii=False)

    system_prompt = (
        "你是材料费发票明细纠错Agent。你只做“可疑行判定 + 字段修复建议”。"
        "对于每一行必须输出一条结果，不能漏行。"
        "判断是否可疑时优先参考该行的相似纠错案例。"
        "输出严格JSON: {\"results\":[...]}。"
    )
    user_prompt = (
        "输入包含：发票头信息、若干行明细、每行的历史纠错案例。\n"
        "请逐行输出字段：row_no,is_suspicious,confidence,risk_types,reason,suggested_row。\n"
        "规则：\n"
        "1) confidence 为 0~1 小数。\n"
        "2) suggested_row 只包含 item_name/spec/quantity/unit/line_total_with_tax。\n"
        "3) 若不确定，请 is_suspicious=true 且 confidence<0.8，并给出 reason。\n"
        "4) 不要删除任何行。\n"
        f"发票头信息: {head_text}\n"
        f"待判断行与案例: {packet_text}"
    )

    payload_chat = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1},
    }
    request_timeout = (8, max(20, int(timeout)))

    errors: list[str] = []
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=request_timeout)
        resp.raise_for_status()
        parsed = _extract_json_from_text((resp.json().get("message") or {}).get("content", ""))
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
        errors.append("chat returned invalid JSON")
    except Exception as exc:
        errors.append(f"chat failed: {exc}")

    payload_generate = {
        "model": model,
        "stream": False,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=request_timeout)
        resp.raise_for_status()
        parsed = _extract_json_from_text(resp.json().get("response", ""))
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
        errors.append("generate returned invalid JSON")
    except Exception as exc:
        errors.append(f"generate failed: {exc}")

    raise RuntimeError("; ".join(errors))


def run_llm_row_repair(
    rows: list[dict[str, Any]],
    *,
    header_context: dict[str, Any] | None = None,
    raw_text: str = "",
) -> dict[str, Any]:
    normalized_rows = [_normalize_row(row) for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return {
            "rows": [],
            "review_items": [],
            "stats": {"chunks_total": 0, "chunks_ok": 0, "chunks_failed": 0, "suspicious_rows": 0, "auto_fixed_rows": 0},
            "llm_error": None,
        }

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _text_model()
    chunk_size = max(1, int(os.getenv("MATERIAL_LLM_SUSPECT_CHUNK_SIZE", "12")))
    top_cases = max(1, int(os.getenv("MATERIAL_LLM_TOP_CASES", "2")))
    timeout = max(30, int(os.getenv("MATERIAL_LLM_TIMEOUT", "120")))
    auto_fix_conf = float(os.getenv("MATERIAL_LLM_AUTO_FIX_CONF", "0.82"))

    indexed_rows = [{"row_no": idx + 1, "row": row} for idx, row in enumerate(normalized_rows)]
    chunks = _chunk_rows(indexed_rows, chunk_size)

    decisions: dict[int, dict[str, Any]] = {}
    chunk_errors: list[str] = []
    chunks_ok = 0

    for chunk_idx, chunk in enumerate(chunks, start=1):
        packets: list[dict[str, Any]] = []
        for item in chunk:
            row_no = int(item["row_no"])
            row = dict(item["row"])
            cases = _retrieve_case_examples(
                row,
                header_context=header_context,
                raw_text=raw_text,
                top_k=top_cases,
            )
            packets.append(
                {
                    "row_no": row_no,
                    "row": row,
                    "case_examples": cases,
                }
            )
        try:
            result_items = _call_llm_chunk(
                packets,
                header_context=header_context,
                model=model,
                base_url=base_url,
                timeout=timeout,
            )
            for result in result_items:
                row_no = result.get("row_no")
                try:
                    idx = int(row_no)
                except (TypeError, ValueError):
                    continue
                decisions[idx] = dict(result)
            chunks_ok += 1
        except Exception as exc:
            chunk_errors.append(f"chunk_{chunk_idx}: {exc}")

    output_rows: list[dict[str, str]] = []
    review_items: list[dict[str, Any]] = []
    suspicious_rows = 0
    auto_fixed_rows = 0

    for row_no, base_row in enumerate(normalized_rows, start=1):
        decision = dict(decisions.get(row_no) or {})
        is_suspicious = bool(decision.get("is_suspicious"))
        confidence = decision.get("confidence")
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        risk_types = decision.get("risk_types")
        if not isinstance(risk_types, list):
            risk_types = []
        risk_types = [str(item).strip() for item in risk_types if str(item).strip()]
        reason = str(decision.get("reason") or "").strip()
        suggested_row_raw = decision.get("suggested_row")
        suggested_row = dict(suggested_row_raw or {}) if isinstance(suggested_row_raw, dict) else {}

        if is_suspicious:
            suspicious_rows += 1
            candidate = _sanitize_suggested_row(suggested_row, base_row) if suggested_row else dict(base_row)
            changed = any(str(candidate.get(k) or "") != str(base_row.get(k) or "") for k in ROW_FIELDS)
            if confidence_value >= auto_fix_conf and changed:
                output_rows.append(candidate)
                auto_fixed_rows += 1
            else:
                output_rows.append(base_row)
                review_items.append(
                    {
                        "row_no": row_no,
                        "item_name": base_row.get("item_name", ""),
                        "spec": base_row.get("spec", ""),
                        "confidence": round(confidence_value, 4),
                        "risk_types": risk_types,
                        "reason": reason or "LLM标记为可疑，建议人工复核。",
                        "suggested_item_name": candidate.get("item_name", ""),
                        "suggested_spec": candidate.get("spec", ""),
                    }
                )
        else:
            output_rows.append(base_row)

    stats = {
        "model": model,
        "chunks_total": len(chunks),
        "chunks_ok": chunks_ok,
        "chunks_failed": max(0, len(chunks) - chunks_ok),
        "suspicious_rows": suspicious_rows,
        "auto_fixed_rows": auto_fixed_rows,
        "review_rows": len(review_items),
        "auto_fix_confidence": auto_fix_conf,
    }

    return {
        "rows": output_rows,
        "review_items": review_items,
        "stats": stats,
        "llm_error": "; ".join(chunk_errors) if chunk_errors else None,
    }
