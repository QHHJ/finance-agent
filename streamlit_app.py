from __future__ import annotations

import base64
import json
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any
from datetime import datetime
import zipfile

import requests
import streamlit as st
from pypdf import PdfReader

from app.runtime import init_runtime
from app.services import extractor, learning, local_runner, rag_retriever

LINE_ITEM_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
TRUE_VALUES = {"1", "true", "yes", "on"}
TRAVEL_DOC_TYPES = {
    "transport_ticket",
    "transport_payment",
    "flight_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "unknown",
}
TRAVEL_RAG_KEYWORDS = [
    "差旅",
    "报销",
    "机票",
    "高铁",
    "火车",
    "交通",
    "酒店",
    "住宿",
    "订单截图",
    "行程单",
    "支付记录",
    "支付凭证",
    "发票",
    "票据",
    "入账",
    "抬头",
    "税号",
]
CHINESE_NUM_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip()
        text = text.replace("−", "-").replace("—", "-")
        text = text.replace(",", "").replace("，", "").replace("¥", "").replace("￥", "")
        text = re.sub(r"[^\d.\-]", "", text)
        if text in {"", ".", "-", "-."}:
            return None
        return float(text)
    except ValueError:
        return None


def _format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _to_editor_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        # pandas.DataFrame
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _normalize_quantity(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def _normalize_line_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item_name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        quantity = _normalize_quantity(row.get("quantity"))
        unit = str(row.get("unit") or "").strip()
        line_total = _format_amount(_safe_float(row.get("line_total_with_tax")))

        if not any([item_name, spec, quantity, unit, line_total]):
            continue

        normalized.append(
            {
                "item_name": item_name,
                "spec": spec,
                "quantity": quantity,
                "unit": unit,
                "line_total_with_tax": line_total,
            }
        )
    return normalized


def _line_items_total(rows: list[dict[str, str]]) -> float | None:
    values = [_safe_float(row.get("line_total_with_tax")) for row in rows]
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers)


@st.cache_data(show_spinner=False)
def _extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join(chunk for chunk in pages if chunk)


def _env_flag_true(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in TRUE_VALUES


def _extract_amount_from_filename(file_name: str) -> float | None:
    name = file_name or ""
    stem = Path(name).stem

    # 1) Strong signal: number followed by "元".
    match = re.search(r"(\d+(?:\.\d{1,2})?)\s*元", stem)
    if match:
        return _safe_float(match.group(1))

    # 2) Fallback: parse numeric tokens in filename, useful for names like
    # "酒店支付凭证8131.44.jpg" or "支付219.png".
    tokens = re.findall(r"\d+(?:\.\d{1,2})?", stem)
    if not tokens:
        return None

    candidates: list[float] = []
    for token in tokens:
        value = _safe_float(token)
        if value is None or value <= 0:
            continue
        # Exclude obvious date/time or ids (too long integers).
        if "." not in token and len(token) >= 6:
            continue
        if value > 200000:
            continue
        candidates.append(value)

    if not candidates:
        return None
    # Prefer the last numeric token in filename, usually closest to amount label.
    return candidates[-1]


def _extract_amount_from_text(raw_text: str) -> float | None:
    text = raw_text or ""
    amount_token = r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"

    def _is_valid_amount(amount: float, token: str) -> bool:
        if amount == 0:
            return False
        if abs(amount) > 200000:
            return False
        cleaned = token.replace(",", "").replace("，", "").replace("−", "-")
        if re.fullmatch(r"-?\d{4}", cleaned):
            year = int(cleaned)
            if 1900 <= abs(year) <= 2099:
                return False
        return True

    def _pick_best(candidates: list[float]) -> float | None:
        if not candidates:
            return None
        return max(candidates, key=lambda value: abs(value))

    payment_candidates: list[float] = []
    patterns = [
        rf"(?:支付金额|实付金额|付款金额|交易金额|订单金额|应付金额|合计支付|支付合计|实付|支付|payment amount|paid amount|amount paid|transaction amount)[^\d¥￥$-]{{0,20}}[¥￥$]?\s*({amount_token})",
        rf"(?:付款|支付|payment|paid)[^\d¥￥$-]{{0,14}}[¥￥$]?\s*({amount_token})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = match.group(1)
            amount = _safe_float(token)
            if amount is not None and _is_valid_amount(amount, token):
                payment_candidates.append(amount)

    best = _pick_best(payment_candidates)
    if best is not None:
        return best

    currency_candidates: list[float] = []
    for match in re.finditer(rf"[¥￥$]\s*({amount_token})", text, flags=re.IGNORECASE):
        token = match.group(1)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            currency_candidates.append(amount)
    for match in re.finditer(rf"({amount_token})\s*元", text, flags=re.IGNORECASE):
        token = match.group(1)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            currency_candidates.append(amount)

    best = _pick_best(currency_candidates)
    if best is not None:
        return best

    # Last fallback for model outputs without symbols/keywords, e.g. "payment_amount: -1850.00".
    generic_candidates: list[float] = []
    for match in re.finditer(amount_token, text):
        token = match.group(0)
        amount = _safe_float(token)
        if amount is not None and _is_valid_amount(amount, token):
            generic_candidates.append(amount)
    best = _pick_best(generic_candidates)
    if best is not None:
        return best
    return None


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
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_confidence(value: Any) -> float | None:
    confidence = _safe_float(value)
    if confidence is None:
        return None
    if confidence > 1:
        confidence /= 100.0
    if confidence < 0 or confidence > 1:
        return None
    return confidence


def _extract_travel_retrieval_terms(raw_text: str) -> list[str]:
    text = (raw_text or "").strip().lower()
    if not text:
        return TRAVEL_RAG_KEYWORDS[:8]

    terms: list[str] = []
    for keyword in TRAVEL_RAG_KEYWORDS:
        if keyword.lower() in text:
            terms.append(keyword)

    # Additional numeric/context terms that often appear in payment/ticket pages.
    extra = re.findall(r"[a-z0-9\-/]{3,24}", text)
    for token in extra:
        if token in terms:
            continue
        if token in {"http", "https", "www"}:
            continue
        terms.append(token)
        if len(terms) >= 20:
            break
    return terms[:20] if terms else TRAVEL_RAG_KEYWORDS[:8]


@st.cache_data(show_spinner=False)
def _load_travel_policy_corpus() -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    try:
        policies = local_runner.list_policies(limit=200)
    except Exception:
        return corpus

    for policy in policies:
        name = str(getattr(policy, "name", "") or "").strip() or "policy"
        raw_text = str(getattr(policy, "raw_text", "") or "").strip()
        if not raw_text:
            continue
        text = re.sub(r"\s+", " ", raw_text)
        if len(text) > 40000:
            text = text[:40000]
        corpus.append({"name": name, "text": text})
    return corpus


def _policy_snippet_around_terms(text: str, terms: list[str], max_chars: int = 420) -> str:
    if not text:
        return ""
    for term in terms:
        if not term:
            continue
        idx = text.find(term)
        if idx < 0:
            continue
        start = max(0, idx - 120)
        end = min(len(text), idx + max_chars - 120)
        return text[start:end]
    return text[:max_chars]


def _build_travel_rag_context(raw_text: str) -> str:
    try:
        context = rag_retriever.build_travel_policy_context(raw_text, top_k=3)
        if context:
            return context
    except Exception:
        pass

    corpus = _load_travel_policy_corpus()
    if not corpus:
        return ""

    terms = _extract_travel_retrieval_terms(raw_text)
    snippets: list[str] = []
    for idx, doc in enumerate(corpus[:2], start=1):
        snippet = _policy_snippet_around_terms(doc["text"], terms, max_chars=420)
        snippets.append(f"[Policy#{idx} score=fallback name={doc['name']}] {snippet}")
    return "\n".join(snippets)


def _rule_confidence_for_doc_type(doc_type: str, raw_text: str) -> int:
    features = _travel_signal_features("", raw_text)
    hotel_hit = features["hotel_hit"]
    transport_hit = features["transport_hit"]
    invoice_hit = features["invoice_hit"]
    payment_hit = features["payment_hit"]
    detail_hit = features["detail_hit"]
    order_hit = features["order_strong_hit"]

    if doc_type == "hotel_order":
        return 3 if order_hit else 0
    if doc_type == "flight_detail":
        return 3 if detail_hit else 0
    if doc_type == "transport_ticket":
        if invoice_hit and transport_hit and not hotel_hit:
            return 3
        if invoice_hit and transport_hit:
            return 2
        return 1 if transport_hit else 0
    if doc_type == "hotel_invoice":
        if invoice_hit and hotel_hit and not transport_hit:
            return 3
        if invoice_hit and hotel_hit:
            return 2
        return 1 if hotel_hit else 0
    if doc_type == "transport_payment":
        if payment_hit and transport_hit and not hotel_hit:
            return 3
        if payment_hit and transport_hit:
            return 2
        return 1 if payment_hit else 0
    if doc_type == "hotel_payment":
        if payment_hit and hotel_hit and not transport_hit:
            return 3
        if payment_hit and hotel_hit:
            return 2
        return 1 if payment_hit else 0
    return 0


def _resolve_travel_doc_type(
    rule_guess: str,
    llm_guess: str,
    llm_confidence: float | None,
    raw_text: str,
) -> tuple[str, str]:
    normalized_rule = rule_guess if rule_guess in TRAVEL_DOC_TYPES else "unknown"
    normalized_llm = llm_guess if llm_guess in TRAVEL_DOC_TYPES else "unknown"

    if normalized_llm == "unknown":
        if normalized_rule != "unknown":
            return normalized_rule, "rule"
        return "unknown", "rule_unknown"

    if normalized_rule == "unknown":
        return normalized_llm, "llm"

    if normalized_llm == normalized_rule:
        return normalized_llm, "llm+rule_agree"

    rule_strength = _rule_confidence_for_doc_type(normalized_rule, raw_text)
    if rule_strength >= 3 and (llm_confidence is None or llm_confidence < 0.8):
        return normalized_rule, "rule_guard"
    if llm_confidence is not None and llm_confidence < 0.45 and rule_strength >= 2:
        return normalized_rule, "rule_guard"

    return normalized_llm, "llm_override"


def _normalize_payment_amount(value: Any) -> float | None:
    amount = _safe_float(value)
    if amount is None:
        return None
    amount = abs(amount)
    return amount if amount > 0 else None


def _pick_payment_candidate(values: list[Any]) -> float | None:
    normalized: list[float] = []
    for value in values:
        amount = _normalize_payment_amount(value)
        if amount is None:
            continue
        if amount > 200000:
            continue
        normalized.append(amount)
    if not normalized:
        return None
    return max(normalized)


def _extract_payment_amount_from_model_output(content: str) -> float | None:
    parsed = _extract_json_from_text(content)
    if parsed:
        for key in ("payment_amount", "paid_amount", "amount", "transaction_amount"):
            if key in parsed:
                amount = _normalize_payment_amount(parsed.get(key))
                if amount is not None:
                    return amount
        for list_key in ("amount_candidates", "amounts", "candidates", "money_list"):
            value = parsed.get(list_key)
            if isinstance(value, list):
                picked = _pick_payment_candidate(value)
                if picked is not None:
                    return picked

    key_match = re.search(
        r"(?:payment_amount|paid_amount|paymentAmount|transaction_amount)\s*[:=]\s*[\"']?([-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)",
        content or "",
        flags=re.IGNORECASE,
    )
    if key_match:
        amount = _normalize_payment_amount(key_match.group(1))
        if amount is not None:
            return amount

    amount = _extract_amount_from_text(content)
    return _normalize_payment_amount(amount)


@st.cache_data(show_spinner=False)
def _extract_payment_amount_with_ollama(file_bytes: bytes) -> float | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    encoded = base64.b64encode(file_bytes).decode("utf-8")

    prompts = [
        (
            "你是财务助手。请从支付记录截图中提取“本次支付金额（人民币）”。"
            "只输出 JSON：{\"payment_amount\": \"123.45\"}。"
            "若无法识别，输出 {\"payment_amount\": null}。"
        ),
        (
            "请识别截图里所有看起来像金额的数字（可带负号、逗号和小数），"
            "只输出 JSON：{\"amount_candidates\": [\"-8131.44\", \"20.00\"]}。"
        ),
    ]

    for prompt in prompts:
        try:
            payload_chat = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=(8, 45))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

        try:
            payload_generate = {
                "model": model,
                "stream": False,
                "prompt": prompt,
                "images": [encoded],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

    return None


@st.cache_data(show_spinner=False)
def _extract_payment_amount_with_ollama_text(raw_text: str) -> float | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None
    text = (raw_text or "").strip()
    if not text:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    prompts = [
        (
            "你是财务助手。请从以下支付记录文本中提取“本次支付金额（人民币）”。"
            "只输出 JSON：{\"payment_amount\": \"123.45\"}。"
            "若无法识别，输出 {\"payment_amount\": null}。\n\n"
            f"支付记录文本：\n{text[:12000]}"
        ),
        (
            "请从下面文本里找出所有看起来像金额的数字（可带负号、逗号和小数），"
            "只输出 JSON：{\"amount_candidates\": [\"-8131.44\", \"20.00\"]}。\n\n"
            f"支付记录文本：\n{text[:12000]}"
        ),
    ]

    for prompt in prompts:
        try:
            payload_generate = {
                "model": model,
                "stream": False,
                "prompt": prompt,
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
            amount = _extract_payment_amount_from_model_output(content)
            if amount is not None:
                return amount
        except Exception:
            pass

    return None


def _auto_extract_amount_from_ticket(uploaded_file) -> float | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
            # Prefer invoice total-with-tax (价税合计小写) for reimbursement amount.
            amount_small_text = None
            for pattern in [
                r"(?:价税合计\s*[（(]小写[)）]|[（(]小写[)）])[:：]?\s*[¥￥]?\s*([\d,]+(?:\.\d{1,2})?)",
                r"(?:价税合计|金额合计|合计金额)[:：]?\s*[¥￥]?\s*([\d,]+(?:\.\d{1,2})?)",
            ]:
                match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    amount_small_text = (match.group(1) or "").strip()
                    if amount_small_text:
                        break
            amount_small = _safe_float(amount_small_text)
            if amount_small is not None:
                return amount_small

            total_line_match = re.search(
                r"(?:^|[\s\r\n])合计[^\n\r]{0,60}?([¥￥]?\s*[\d,]+\.\d{2})\s+([¥￥]?\s*[\d,]+\.\d{2})",
                raw_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if total_line_match:
                no_tax_total = _safe_float(total_line_match.group(1))
                tax_total = _safe_float(total_line_match.group(2))
                if no_tax_total is not None and tax_total is not None:
                    return float(no_tax_total + tax_total)

            extracted = extractor.extract_invoice_fields(raw_text)
            amount = _safe_float(extracted.get("amount"))
            tax_amount = _safe_float(extracted.get("tax_amount"))
            if amount is not None and tax_amount is not None:
                # Heuristic: when extracted amount is no-tax and tax exists, amount+tax should
                # better match ticket reimbursement than amount itself for transport/hotel invoices.
                has_small_label = re.search(
                    r"(?:价税合计\s*[（(]小写[)）]|[（(]小写[)）])",
                    raw_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if not has_small_label and tax_amount > 0 and abs(tax_amount) < abs(amount):
                    candidate_total = float(amount + tax_amount)
                    if candidate_total > 0:
                        return candidate_total
            if amount is not None:
                return amount
            amount = _extract_amount_from_text(raw_text)
            if amount is not None:
                return amount
            # Last text-only fallback uses model parsing, still based on content.
            return _extract_payment_amount_with_ollama_text(raw_text)
        except Exception:
            return None

    # 对图片票据优先用视觉模型识别。
    amount = _extract_payment_amount_with_ollama(file_bytes)
    if amount is not None:
        return amount
    return None


def _auto_extract_amount_from_payment(uploaded_file) -> float | None:
    if uploaded_file is None:
        return None

    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
            amount = _extract_payment_amount_with_ollama_text(raw_text)
            if amount is not None:
                return amount
        except Exception:
            return None
    else:
        amount = _extract_payment_amount_with_ollama(file_bytes)
        if amount is not None:
            return amount

    return None


def _as_uploaded_list(uploaded_value) -> list[Any]:
    if uploaded_value is None:
        return []
    if isinstance(uploaded_value, list):
        return [item for item in uploaded_value if item is not None]
    return [uploaded_value]


def _files_signature(files: list[Any]) -> str:
    parts: list[str] = []
    for file in files:
        name = str(getattr(file, "name", ""))
        size = str(getattr(file, "size", ""))
        parts.append(f"{name}:{size}")
    return "|".join(parts)


def _travel_file_key(name: str, size: Any) -> str:
    return f"{str(name or '').strip()}:{str(size or '').strip()}"


def _uploaded_file_key(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    return _travel_file_key(
        str(getattr(uploaded_file, "name", "")),
        getattr(uploaded_file, "size", ""),
    )


def _profile_file_key(profile: dict[str, Any]) -> str:
    file_obj = profile.get("file")
    size = getattr(file_obj, "size", "") if file_obj is not None else ""
    return _travel_file_key(str(profile.get("name") or ""), size)


def _set_manual_override_for_profile(manual_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    doc_type = str(profile.get("doc_type") or "").strip()
    if not key or doc_type not in TRAVEL_DOC_TYPES:
        return
    manual_overrides[key] = doc_type


def _remember_manual_overrides(manual_overrides: dict[str, str], profiles: list[dict[str, Any]]) -> int:
    updated = 0
    for profile in profiles:
        source = str(profile.get("source") or "")
        if source not in {"manual_chat", "manual_table", "manual_persist"}:
            continue
        key = _profile_file_key(profile)
        doc_type = str(profile.get("doc_type") or "").strip()
        if not key or doc_type not in TRAVEL_DOC_TYPES:
            continue
        if manual_overrides.get(key) == doc_type:
            continue
        manual_overrides[key] = doc_type
        updated += 1
    return updated


def _remove_manual_override_for_profile(manual_overrides: dict[str, str], profile: dict[str, Any]) -> None:
    key = _profile_file_key(profile)
    if key:
        manual_overrides.pop(key, None)


def _prune_manual_overrides(manual_overrides: dict[str, str], pool_files: list[Any]) -> None:
    valid_keys = {_uploaded_file_key(file) for file in pool_files if file is not None}
    for key in list(manual_overrides.keys()):
        if key not in valid_keys:
            manual_overrides.pop(key, None)


def _apply_manual_overrides_to_profiles(profiles: list[dict[str, Any]], manual_overrides: dict[str, str]) -> int:
    if not manual_overrides:
        return 0
    changed = 0
    for profile in profiles:
        key = _profile_file_key(profile)
        target_doc_type = str(manual_overrides.get(key) or "").strip()
        if target_doc_type not in TRAVEL_DOC_TYPES:
            continue
        current_doc_type = str(profile.get("doc_type") or "unknown")
        current_source = str(profile.get("source") or "")
        if current_doc_type != target_doc_type or not current_source.startswith("manual"):
            profile["doc_type"] = target_doc_type
            profile["source"] = "manual_persist"
            changed += 1
    return changed


def _aggregate_auto_amount(files: list[Any], extractor_func) -> tuple[float | None, int, int]:
    total = 0.0
    recognized = 0
    for file in files:
        amount = extractor_func(file)
        if amount is None:
            continue
        recognized += 1
        total += amount

    total_files = len(files)
    if recognized == 0:
        return None, 0, total_files
    return total, recognized, total_files


def _sync_amount_state(prefix: str, role: str, uploaded_files: list[Any], auto_amount: float | None) -> str:
    file_state_key = f"{prefix}_{role}_file_signature"
    amount_state_key = f"{prefix}_{role}_amount"
    current_signature = _files_signature(uploaded_files)

    if st.session_state.get(file_state_key) != current_signature:
        st.session_state[file_state_key] = current_signature
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    elif amount_state_key not in st.session_state:
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    elif _safe_float(st.session_state.get(amount_state_key)) is None and auto_amount is not None:
        # Preserve manual corrections, but auto-fill when previous value is empty.
        st.session_state[amount_state_key] = _format_amount(auto_amount)
    return amount_state_key


def _render_amount_match_check(prefix: str, ticket_label: str, ticket_files, payment_files) -> dict[str, Any]:
    ticket_file_list = _as_uploaded_list(ticket_files)
    payment_file_list = _as_uploaded_list(payment_files)

    ticket_auto_amount, ticket_recognized, ticket_total = _aggregate_auto_amount(
        ticket_file_list, _auto_extract_amount_from_ticket
    )
    payment_auto_amount, payment_recognized, payment_total = _aggregate_auto_amount(
        payment_file_list, _auto_extract_amount_from_payment
    )

    ticket_amount_key = _sync_amount_state(prefix, "ticket", ticket_file_list, ticket_auto_amount)
    payment_amount_key = _sync_amount_state(prefix, "payment", payment_file_list, payment_auto_amount)

    left, right = st.columns(2)
    left.text_input(f"{ticket_label}金额（合计）", key=ticket_amount_key)
    right.text_input("支付记录金额（合计）", key=payment_amount_key)

    if ticket_auto_amount is not None:
        left.caption(
            f"已自动识别 {ticket_recognized}/{ticket_total} 份，合计：{_format_amount(ticket_auto_amount)}"
        )
    elif ticket_total > 0:
        left.caption(f"已上传 {ticket_total} 份，自动识别 0 份。")
    if payment_auto_amount is not None:
        right.caption(
            f"已自动识别 {payment_recognized}/{payment_total} 份，合计：{_format_amount(payment_auto_amount)}"
        )
    elif payment_total > 0:
        right.caption(f"已上传 {payment_total} 份，自动识别 0 份。")

    ticket_amount = _safe_float(st.session_state.get(ticket_amount_key))
    payment_amount = _safe_float(st.session_state.get(payment_amount_key))

    checked = ticket_amount is not None and payment_amount is not None
    matched = checked and abs(ticket_amount - payment_amount) <= 0.01

    if checked and matched:
        st.success(f"{ticket_label}与支付记录金额一致。")
    elif checked:
        st.error(
            f"{ticket_label}与支付记录金额不一致："
            f"{_format_amount(ticket_amount)} vs {_format_amount(payment_amount)}"
        )
    elif ticket_total > 0 and payment_total > 0:
        st.warning("已上传票据和支付记录，但尚未形成可比对金额，请补充或修正金额。")

    return {
        "ticket_amount": ticket_amount,
        "payment_amount": payment_amount,
        "ticket_files_total": ticket_total,
        "ticket_files_recognized": ticket_recognized,
        "payment_files_total": payment_total,
        "payment_files_recognized": payment_recognized,
        "checked": checked,
        "matched": matched,
    }


def _render_travel_transport_section(section_title: str, prefix: str) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown(f"#### {section_title}")
        transport_type = st.radio(
            "交通方式",
            options=["飞机", "高铁"],
            horizontal=True,
            key=f"{prefix}_transport_type",
        )
        ticket_label = "机票发票" if transport_type == "飞机" else "高铁报销凭证"

        ticket_files = st.file_uploader(
            f"上传{ticket_label}（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_ticket_file",
        )
        payment_files = st.file_uploader(
            "上传支付记录（微信/支付宝/银行，PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_payment_file",
        )
        ticket_detail_files = st.file_uploader(
            "上传机票明细（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_ticket_detail_file",
        )

        amount_check = _render_amount_match_check(prefix, ticket_label, ticket_files, payment_files)

        missing: list[str] = []
        if not ticket_files:
            missing.append(ticket_label)
        if not payment_files:
            missing.append("支付记录")
        if not ticket_detail_files:
            missing.append("机票明细")

        if missing:
            st.info(f"待补充：{'、'.join(missing)}")

        return {
            "section": section_title,
            "transport_type": transport_type,
            "ticket_label": ticket_label,
            "ticket_uploaded": bool(ticket_files),
            "payment_uploaded": bool(payment_files),
            "ticket_detail_uploaded": bool(ticket_detail_files),
            "ticket_file_count": len(ticket_files or []),
            "payment_file_count": len(payment_files or []),
            "ticket_detail_file_count": len(ticket_detail_files or []),
            "complete": not missing,
            **amount_check,
        }


def _render_travel_hotel_section(prefix: str) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown("#### 3) 酒店报销")
        no_cost_stay = st.checkbox(
            "在出差地住宿不产生费用（如亲友家住宿）",
            value=False,
            key=f"{prefix}_no_cost_stay",
        )

        if no_cost_stay:
            explanation = st.text_area(
                "情况说明（必填）",
                height=100,
                key=f"{prefix}_no_cost_explanation",
                placeholder="请说明住宿地点、日期范围及未产生费用原因。",
            )
            complete = bool(explanation.strip())
            if complete:
                st.success("已提供无费用住宿情况说明。")
            else:
                st.warning("请填写无费用住宿情况说明。")
            return {
                "section": "酒店",
                "no_cost_stay": True,
                "complete": complete,
                "checked": True,
                "matched": True,
                "ticket_amount": None,
                "payment_amount": None,
            }

        invoice_files = st.file_uploader(
            "上传酒店发票（PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_invoice",
        )
        payment_files = st.file_uploader(
            "上传酒店支付记录（微信/支付宝/银行，PDF/图片）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_payment",
        )
        order_files = st.file_uploader(
            "上传酒店平台订单截图（如携程）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"{prefix}_hotel_order",
        )

        amount_check = _render_amount_match_check(prefix, "酒店发票", invoice_files, payment_files)

        missing: list[str] = []
        if not invoice_files:
            missing.append("酒店发票")
        if not payment_files:
            missing.append("支付记录")
        if not order_files:
            missing.append("酒店订单截图")

        if missing:
            st.info(f"待补充：{'、'.join(missing)}")

        return {
            "section": "酒店",
            "no_cost_stay": False,
            "invoice_uploaded": bool(invoice_files),
            "payment_uploaded": bool(payment_files),
            "order_uploaded": bool(order_files),
            "invoice_file_count": len(invoice_files or []),
            "payment_file_count": len(payment_files or []),
            "order_file_count": len(order_files or []),
            "complete": not missing,
            **amount_check,
        }


def _render_travel_summary(
    go_section: dict[str, Any],
    return_section: dict[str, Any],
    hotel_section: dict[str, Any],
) -> None:
    st.subheader("差旅流程校验结果")

    issues: list[str] = []
    tips: list[str] = []

    if not go_section.get("complete"):
        issues.append("去程交通材料不完整。")
    if not return_section.get("complete"):
        issues.append("返程交通材料不完整。")
    if not hotel_section.get("complete"):
        issues.append("酒店材料不完整。")

    if go_section.get("complete") and return_section.get("complete"):
        st.success("交通闭环已满足：去程与返程材料均已上传。")
    else:
        st.error("交通闭环未满足：需同时提供去程与返程交通材料。")

    for section_name, section_data in [("去程交通", go_section), ("返程交通", return_section), ("酒店", hotel_section)]:
        if section_data.get("checked") and not section_data.get("matched"):
            issues.append(f"{section_name}票据金额与支付记录金额不一致。")
        elif section_data.get("complete") and not section_data.get("checked"):
            tips.append(f"{section_name}已上传材料，但金额尚未形成有效匹配。")

    if issues:
        st.error("存在高优先级问题，请修正后再提交报销。")
        for issue in issues:
            st.markdown(f"- {issue}")
    else:
        st.success("当前材料满足基础提交要求。")

    if tips:
        st.warning("提示：")
        for tip in tips:
            st.markdown(f"- {tip}")

    summary = {
        "go_transport": go_section,
        "return_transport": return_section,
        "hotel": hotel_section,
        "all_required_uploaded": not issues,
    }
    with st.expander("查看差旅流程结构化结果(JSON)", expanded=False):
        st.json(summary)


def _parse_date_value(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_candidate_dates(text: str) -> list[datetime]:
    content = text or ""
    candidates: list[datetime] = []

    for match in re.finditer(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", content):
        year, month, day = match.groups()
        try:
            candidates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue

    for match in re.finditer(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", content):
        year, month, day = match.groups()
        try:
            candidates.append(datetime(int(year), int(month), int(day)))
        except ValueError:
            continue

    # Keep unique dates and preserve order.
    seen: set[str] = set()
    unique: list[datetime] = []
    for dt in candidates:
        key = dt.strftime("%Y-%m-%d")
        if key in seen:
            continue
        seen.add(key)
        unique.append(dt)
    return unique


def _pick_primary_date(file_name: str, raw_text: str) -> datetime | None:
    text_dates = _extract_candidate_dates(raw_text)
    if text_dates:
        return text_dates[0]
    file_dates = _extract_candidate_dates(file_name)
    if file_dates:
        return file_dates[0]
    return None


def _keyword_score(text: str, keywords: list[str]) -> int:
    return sum(1 for key in keywords if key in text)


def _is_generic_evidence(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    compact = re.sub(r"\s+", "", value)
    generic_markers = {
        "shortreason",
        "reason",
        "unknown",
        "n/a",
        "na",
        "none",
        "无法判断",
        "无法识别",
        "不确定",
        "看不清",
        "无",
        "空",
    }
    if compact in generic_markers:
        return True
    if len(compact) <= 3:
        return True
    return False


def _infer_doc_type_from_visual_keywords(text: str) -> str:
    merged = re.sub(r"\s+", "", (text or "").lower())
    if not merged:
        return "unknown"

    hotel_order_keys = [
        "携程旅行",
        "飞猪旅行",
        "同程旅行",
        "美团酒店",
        "酒店订单",
        "订单详情",
        "订单号",
        "晚明细",
        "费用明细",
        "取消政策",
        "已完成",
        "在线付",
    ]
    flight_detail_keys = [
        "价格明细",
        "机建",
        "燃油",
        "票价",
        "普通成人",
        "退改签",
        "行程单",
        "航段",
        "机票详情",
    ]
    payment_core_keys = [
        "账单详情",
        "交易成功",
        "支付时间",
        "付款方式",
        "商品说明",
        "收单机构",
        "交易单号",
        "交易号",
        "账单分类",
    ]
    transport_keys = [
        "机票",
        "航班",
        "高铁",
        "火车",
        "铁路",
        "航空",
        "客票",
        "代订机票",
        "机建",
        "燃油",
    ]
    hotel_keys = [
        "酒店",
        "住宿",
        "房费",
        "宾馆",
        "旅馆",
        "华住",
        "汉庭",
        "全季",
        "如家",
        "亚朵",
    ]
    invoice_keys = [
        "电子发票",
        "发票号码",
        "开票日期",
        "购买方",
        "销售方",
        "价税合计",
        "税额",
        "经纪代理服务",
        "客运服务",
        "代订机票费",
    ]

    if any(k in merged for k in hotel_order_keys):
        return "hotel_order"
    if any(k in merged for k in flight_detail_keys):
        return "flight_detail"

    payment_hit = any(k in merged for k in payment_core_keys)
    transport_hit = any(k in merged for k in transport_keys)
    hotel_hit = any(k in merged for k in hotel_keys)
    invoice_hit = any(k in merged for k in invoice_keys) or ("发票" in merged)

    if payment_hit:
        if hotel_hit and not transport_hit:
            return "hotel_payment"
        if transport_hit and not hotel_hit:
            return "transport_payment"
        if "在线付" in merged and hotel_hit:
            return "hotel_payment"
        return "transport_payment"

    if invoice_hit:
        if transport_hit and not hotel_hit:
            return "transport_ticket"
        if hotel_hit and not transport_hit:
            return "hotel_invoice"
        if "代订机票费" in merged:
            return "transport_ticket"
        if "住宿服务" in merged:
            return "hotel_invoice"

    if hotel_hit and not transport_hit:
        return "hotel_invoice"
    if transport_hit and not hotel_hit:
        return "transport_ticket"
    return "unknown"


def _travel_signal_features(file_name: str, raw_text: str) -> dict[str, bool]:
    combined = f"{file_name}\n{raw_text}".lower()
    combined = re.sub(r"\s+", "", combined)

    hotel_hit = any(
        key in combined
        for key in ["酒店", "住宿", "房费", "入住", "离店", "旅馆", "宾馆", "hotel", "checkin", "checkout"]
    )
    transport_hit = any(
        key in combined
        for key in [
            "机票",
            "航班",
            "高铁",
            "火车",
            "铁路",
            "航空",
            "乘机",
            "起飞",
            "到达",
            "客票",
            "舱位",
            "boarding",
            "flight",
            "train",
        ]
    )
    invoice_hit = any(
        key in combined
        for key in ["发票", "电子发票", "普通发票", "专用发票", "税额", "价税合计", "购买方", "销售方", "invoice"]
    )
    payment_hit = any(
        key in combined
        for key in [
            "支付凭证",
            "支付记录",
            "交易成功",
            "微信支付",
            "支付宝",
            "余额宝",
            "银行",
            "交易单号",
            "支付时间",
            "payment",
            "transaction",
            "receipt",
        ]
    )
    detail_hit = any(
        key in combined
        for key in ["机票明细", "行程单", "电子客票行程单", "客票行程", "航段", "itinerary", "tripdetail"]
    )

    platform_hit = any(key in combined for key in ["携程", "飞猪", "美团", "同程", "booking", "trip.com"])
    # Do NOT use only "订单号" / "预订" to classify as hotel order, too noisy.
    order_strong_hit = any(
        key in combined
        for key in ["订单截图", "酒店订单", "携程订单", "飞猪订单", "晚明细", "费用明细", "取消政策", "已完成"]
    ) or ("订单" in combined and (hotel_hit or platform_hit) and ("入住" in combined or "离店" in combined))

    return {
        "hotel_hit": hotel_hit,
        "transport_hit": transport_hit,
        "invoice_hit": invoice_hit,
        "payment_hit": payment_hit,
        "detail_hit": detail_hit,
        "order_strong_hit": order_strong_hit,
        "platform_hit": platform_hit,
    }


def _guess_travel_doc_type(file_name: str, raw_text: str) -> str:
    features = _travel_signal_features(file_name, raw_text)
    hotel_hit = features["hotel_hit"]
    transport_hit = features["transport_hit"]
    invoice_hit = features["invoice_hit"]
    payment_hit = features["payment_hit"]
    detail_hit = features["detail_hit"]
    order_strong_hit = features["order_strong_hit"]

    if order_strong_hit:
        return "hotel_order"

    if detail_hit:
        return "flight_detail"

    # Invoices have higher priority than payment, otherwise invoice PDFs are often misread as payment.
    if invoice_hit:
        if transport_hit and not hotel_hit:
            return "transport_ticket"
        if hotel_hit and not transport_hit:
            return "hotel_invoice"

    if payment_hit:
        if hotel_hit and not transport_hit:
            return "hotel_payment"
        if transport_hit and not hotel_hit:
            return "transport_payment"
        return "transport_payment"

    if hotel_hit and invoice_hit and not transport_hit:
        return "hotel_invoice"

    if transport_hit and invoice_hit and not hotel_hit:
        return "transport_ticket"

    if hotel_hit and not transport_hit:
        return "hotel_invoice"
    if transport_hit and not hotel_hit:
        return "transport_ticket"
    return "unknown"


def _infer_doc_type_from_invoice_fields(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "unknown"
    try:
        fields = extractor.extract_invoice_fields(text)
    except Exception:
        return "unknown"

    item_content = str(fields.get("item_content") or "")
    seller = str(fields.get("seller") or "")
    bill_type = str(fields.get("bill_type") or "")
    merged = f"{item_content} {seller} {bill_type}".lower()

    transport_keys = ["航空", "机票", "客运服务", "航班", "铁路", "高铁", "火车", "客票"]
    hotel_keys = ["酒店", "住宿", "房费", "旅馆", "宾馆"]

    if any(key in merged for key in transport_keys):
        return "transport_ticket"
    if any(key in merged for key in hotel_keys):
        return "hotel_invoice"
    return "unknown"


@st.cache_data(show_spinner=False)
def _classify_travel_doc_with_ollama(
    file_bytes: bytes,
    suffix: str,
    file_name: str,
    raw_text: str,
    rule_hint: str = "unknown",
    retry_tag: str = "",
) -> dict[str, Any] | None:
    if not _env_flag_true("USE_OLLAMA_VL"):
        return None

    # Keep parameter for cache key control when user requests forced re-recognition.
    _ = retry_tag

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    rag_context = _build_travel_rag_context(raw_text)
    policy_block = f"\nPolicy snippets:\n{rag_context}\n" if rag_context else ""
    rule_hint_text = rule_hint if rule_hint in TRAVEL_DOC_TYPES else "unknown"
    prompt = (
        "你是差旅报销材料分类助手。必须仅根据材料内容分类，不能根据文件名猜测。\n"
        "允许的 doc_type 只有：transport_ticket, transport_payment, flight_detail, hotel_invoice, hotel_payment, hotel_order, unknown。\n"
        "判定规则（严格按内容）：\n"
        "1) 支付凭证常见关键词：账单详情、交易成功、支付时间、付款方式、商品说明、账单分类。\n"
        "2) 机票明细常见关键词：价格明细、机建、燃油、票价、普通成人、退改签。\n"
        "3) 机票发票常见关键词：电子发票、经纪代理服务*代订机票费、发票号码、购买方、销售方、价税合计。\n"
        "4) 酒店订单明细常见关键词：携程旅行/飞猪旅行、订单号、几晚明细、费用明细、取消政策、在线付。\n"
        "5) 酒店发票常见关键词：电子发票、住宿/房费、入住/离店、价税合计。\n"
        "6) 若同一图有“在线付+几晚明细+费用明细”，优先判为 hotel_order，不要判成 hotel_payment。\n"
        "输出必须是单个 JSON 对象，不要任何额外文本，格式如下：\n"
        '{"doc_type":"transport_payment","confidence":0.88,"amount":"2360.00","date":"2026-03-13","evidence":"命中关键词: 账单详情,交易成功,商品说明","ocr_text":"...","keywords":["账单详情","交易成功","商品说明"]}\n'
        "约束：\n"
        "- confidence 取值 0~1。\n"
        "- amount 为主要金额（支付金额或发票价税合计）；无法识别填 null。\n"
        "- date 用 YYYY-MM-DD；无法识别填 null。\n"
        "- evidence 不要写泛化词（如 short reason / unknown）。\n"
        f"规则引擎提示（content-only）: {rule_hint_text}\n"
        f"{policy_block}"
    )

    content = ""
    text = (raw_text or "").strip()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    try:
        if text:
            payload = {
                "model": model,
                "stream": False,
                "prompt": f"{prompt}\nDocument text:\n{text[:12000]}",
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 45))
            resp.raise_for_status()
            content = resp.json().get("response", "")
        elif suffix in image_suffixes:
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "options": {"temperature": 0},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 45))
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
        else:
            return None
    except Exception:
        # Fallback path for Ollama versions/models where /api/chat or /api/generate behavior varies.
        try:
            if text:
                payload = {
                    "model": model,
                    "stream": False,
                    "messages": [{"role": "user", "content": f"{prompt}\nDocument text:\n{text[:12000]}"}],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = (resp.json().get("message") or {}).get("content", "")
            elif suffix in image_suffixes:
                encoded = base64.b64encode(file_bytes).decode("utf-8")
                payload = {
                    "model": model,
                    "stream": False,
                    "prompt": prompt,
                    "images": [encoded],
                    "options": {"temperature": 0},
                }
                resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 45))
                resp.raise_for_status()
                content = resp.json().get("response", "")
            else:
                return None
        except Exception:
            return None

    parsed = _extract_json_from_text(content)
    if not parsed:
        return None

    doc_type = str(parsed.get("doc_type") or "unknown").strip()
    if doc_type not in TRAVEL_DOC_TYPES:
        doc_type = "unknown"
    amount = _normalize_payment_amount(parsed.get("amount"))
    date_value = str(parsed.get("date") or "").strip()
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if _is_generic_evidence(evidence):
        evidence = ""
    if len(evidence) > 220:
        evidence = evidence[:217] + "..."

    ocr_text = str(parsed.get("ocr_text") or parsed.get("text") or parsed.get("recognized_text") or "").strip()
    if len(ocr_text) > 1200:
        ocr_text = ocr_text[:1200]

    keywords_text = ""
    keywords = parsed.get("keywords")
    if isinstance(keywords, list):
        tokens = [str(item).strip() for item in keywords if str(item).strip()]
        keywords_text = " ".join(tokens[:20])

    keyword_fallback_text = "\n".join(part for part in [ocr_text, keywords_text, evidence] if part)
    keyword_guess = _infer_doc_type_from_visual_keywords(keyword_fallback_text)
    if doc_type == "unknown" and keyword_guess != "unknown":
        doc_type = keyword_guess
        if confidence is None:
            confidence = 0.72
        if not evidence:
            evidence = f"关键词兜底命中: {keyword_guess}"
        else:
            evidence = f"{evidence}; 关键词兜底: {keyword_guess}"

    if len(evidence) > 220:
        evidence = evidence[:217] + "..."
    if doc_type == "unknown":
        confidence = None
    return {
        "doc_type": doc_type,
        "amount": amount,
        "date": date_value,
        "confidence": confidence,
        "evidence": evidence,
        "ocr_text": ocr_text,
        "file_name": file_name,
    }


def _recognize_travel_file(uploaded_file: Any, index: int, retry_tag: str = "") -> dict[str, Any]:
    file_name = str(getattr(uploaded_file, "name", ""))
    suffix = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    raw_text = ""
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text_from_bytes(file_bytes)
        except Exception:
            raw_text = ""

    # Deterministic guess from content only (do not use file name for classification).
    rule_guess = _guess_travel_doc_type("", raw_text)
    invoice_field_guess = _infer_doc_type_from_invoice_fields(raw_text)
    if invoice_field_guess != "unknown":
        rule_guess = invoice_field_guess

    llm_result = _classify_travel_doc_with_ollama(
        file_bytes=file_bytes,
        suffix=suffix,
        file_name=file_name,
        raw_text=raw_text,
        rule_hint=rule_guess,
        retry_tag=retry_tag,
    )
    llm_ocr_text = str((llm_result or {}).get("ocr_text") or "").strip()
    if llm_ocr_text:
        ocr_rule_guess = _guess_travel_doc_type("", llm_ocr_text)
        if rule_guess == "unknown" and ocr_rule_guess != "unknown":
            rule_guess = ocr_rule_guess
        if rule_guess == "unknown":
            ocr_invoice_guess = _infer_doc_type_from_invoice_fields(llm_ocr_text)
            if ocr_invoice_guess != "unknown":
                rule_guess = ocr_invoice_guess

    llm_guess = str((llm_result or {}).get("doc_type") or "unknown")
    llm_confidence = _normalize_confidence((llm_result or {}).get("confidence"))
    merged_signal_text = "\n".join(part for part in [raw_text, llm_ocr_text] if part)
    guessed, source = _resolve_travel_doc_type(rule_guess, llm_guess, llm_confidence, merged_signal_text)

    amount: float | None = None
    llm_amount = _normalize_payment_amount((llm_result or {}).get("amount"))
    if guessed in {"transport_ticket", "hotel_invoice"}:
        # For invoices, reimbursement should use tax-included total from invoice text first.
        amount = _auto_extract_amount_from_ticket(uploaded_file)
        if amount is None and llm_amount is not None:
            amount = llm_amount
    elif guessed in {"transport_payment", "hotel_payment"}:
        if llm_amount is not None:
            amount = llm_amount
        else:
            amount = _auto_extract_amount_from_payment(uploaded_file)
    elif llm_amount is not None:
        amount = llm_amount

    date_obj = _pick_primary_date(file_name, merged_signal_text)
    if date_obj is None:
        llm_date = str((llm_result or {}).get("date") or "").strip()
        candidate_dates = _extract_candidate_dates(llm_date)
        if candidate_dates:
            date_obj = candidate_dates[0]

    evidence = str((llm_result or {}).get("evidence") or "").strip()

    return {
        "profile_id": f"{index}:{file_name}:{getattr(uploaded_file, 'size', '')}",
        "index": index,
        "file": uploaded_file,
        "name": file_name,
        "suffix": suffix,
        "doc_type": guessed,
        "amount": amount,
        "date_obj": date_obj,
        "date": date_obj.strftime("%Y-%m-%d") if date_obj else "",
        "slot": "unknown",
        "source": source,
        "confidence": llm_confidence,
        "evidence": evidence,
    }


def _build_travel_file_profile(uploaded_file: Any, index: int) -> dict[str, Any]:
    return _recognize_travel_file(uploaded_file, index=index, retry_tag="")


def _sum_profile_amount(profiles: list[dict[str, Any]]) -> float | None:
    numbers = [p.get("amount") for p in profiles if p.get("amount") is not None]
    if not numbers:
        return None
    return float(sum(numbers))


def _split_profiles_to_go_return(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not profiles:
        return [], []
    if len(profiles) == 1:
        return profiles[:], []

    with_date = [p for p in profiles if p.get("date_obj") is not None]
    if with_date:
        unique_dates = sorted({p["date_obj"].date() for p in with_date})
        if len(unique_dates) >= 2:
            split = max(1, len(unique_dates) // 2)
            go_dates = set(unique_dates[:split])
            go: list[dict[str, Any]] = []
            ret: list[dict[str, Any]] = []
            undecided: list[dict[str, Any]] = []
            for profile in profiles:
                date_obj = profile.get("date_obj")
                if date_obj is None:
                    undecided.append(profile)
                    continue
                if date_obj.date() in go_dates:
                    go.append(profile)
                else:
                    ret.append(profile)
            for profile in undecided:
                if len(go) <= len(ret):
                    go.append(profile)
                else:
                    ret.append(profile)
            if not ret and len(go) > 1:
                ret.append(go.pop())
            return go, ret

    ordered = sorted(profiles, key=lambda p: p.get("index", 0))
    split = max(1, len(ordered) // 2)
    if split >= len(ordered):
        split = len(ordered) - 1
    return ordered[:split], ordered[split:]


def _split_payment_profiles_to_go_return(
    payments: list[dict[str, Any]],
    go_target: float | None,
    return_target: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not payments:
        return [], []
    if len(payments) == 1:
        return payments[:], []

    can_amount_match = (
        go_target is not None
        and return_target is not None
        and all(p.get("amount") is not None for p in payments)
        and len(payments) <= 18
    )
    if can_amount_match:
        values = [float(p["amount"]) for p in payments]
        total = sum(values)
        n = len(values)
        all_mask = (1 << n) - 1
        best_mask = None
        best_score = float("inf")

        for mask in range(0, all_mask + 1):
            go_sum = 0.0
            for idx in range(n):
                if (mask >> idx) & 1:
                    go_sum += values[idx]
            return_sum = total - go_sum
            score = abs(go_sum - go_target) + abs(return_sum - return_target)
            if mask in (0, all_mask):
                score += 10000.0
            if score < best_score:
                best_score = score
                best_mask = mask

        if best_mask is not None:
            go: list[dict[str, Any]] = []
            ret: list[dict[str, Any]] = []
            for idx, profile in enumerate(payments):
                if (best_mask >> idx) & 1:
                    go.append(profile)
                else:
                    ret.append(profile)
            if go and ret:
                return go, ret

    return _split_profiles_to_go_return(payments)


def _build_assignment_from_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    transport_tickets = [p for p in profiles if p.get("doc_type") == "transport_ticket"]
    transport_payments = [p for p in profiles if p.get("doc_type") == "transport_payment"]
    flight_details = [p for p in profiles if p.get("doc_type") == "flight_detail"]
    hotel_invoices = [p for p in profiles if p.get("doc_type") == "hotel_invoice"]
    hotel_payments = [p for p in profiles if p.get("doc_type") == "hotel_payment"]
    hotel_orders = [p for p in profiles if p.get("doc_type") == "hotel_order"]
    unknowns = [p for p in profiles if p.get("doc_type") == "unknown"]

    go_tickets, return_tickets = _split_profiles_to_go_return(transport_tickets)
    go_details, return_details = _split_profiles_to_go_return(flight_details)

    go_ticket_amount = _sum_profile_amount(go_tickets)
    return_ticket_amount = _sum_profile_amount(return_tickets)
    go_payments, return_payments = _split_payment_profiles_to_go_return(
        transport_payments,
        go_ticket_amount,
        return_ticket_amount,
    )
    go_payment_amount = _sum_profile_amount(go_payments)
    return_payment_amount = _sum_profile_amount(return_payments)
    hotel_invoice_amount = _sum_profile_amount(hotel_invoices)
    hotel_payment_amount = _sum_profile_amount(hotel_payments)

    for p in go_tickets:
        p["slot"] = "go_ticket"
    for p in go_payments:
        p["slot"] = "go_payment"
    for p in go_details:
        p["slot"] = "go_detail"
    for p in return_tickets:
        p["slot"] = "return_ticket"
    for p in return_payments:
        p["slot"] = "return_payment"
    for p in return_details:
        p["slot"] = "return_detail"
    for p in hotel_invoices:
        p["slot"] = "hotel_invoice"
    for p in hotel_payments:
        p["slot"] = "hotel_payment"
    for p in hotel_orders:
        p["slot"] = "hotel_order"
    for p in unknowns:
        p["slot"] = "unknown"

    assignment = {
        "go_ticket": [p["file"] for p in go_tickets],
        "go_payment": [p["file"] for p in go_payments],
        "go_detail": [p["file"] for p in go_details],
        "return_ticket": [p["file"] for p in return_tickets],
        "return_payment": [p["file"] for p in return_payments],
        "return_detail": [p["file"] for p in return_details],
        "hotel_invoice": [p["file"] for p in hotel_invoices],
        "hotel_payment": [p["file"] for p in hotel_payments],
        "hotel_order": [p["file"] for p in hotel_orders],
        "unknown": [p["file"] for p in unknowns],
        "go_ticket_amount": go_ticket_amount,
        "go_payment_amount": go_payment_amount,
        "return_ticket_amount": return_ticket_amount,
        "return_payment_amount": return_payment_amount,
        "hotel_invoice_amount": hotel_invoice_amount,
        "hotel_payment_amount": hotel_payment_amount,
    }
    return assignment


def _organize_travel_materials(
    pool_files: list[Any],
    manual_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = [_build_travel_file_profile(file, idx) for idx, file in enumerate(pool_files)]
    if manual_overrides:
        _apply_manual_overrides_to_profiles(profiles, manual_overrides)
    assignment = _build_assignment_from_profiles(profiles)
    return assignment, profiles


def _slot_label(slot: str) -> str:
    mapping = {
        "go_ticket": "去程机票发票/票据",
        "go_payment": "去程支付记录",
        "go_detail": "去程机票明细",
        "return_ticket": "返程机票发票/票据",
        "return_payment": "返程支付记录",
        "return_detail": "返程机票明细",
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "unknown": "未识别",
    }
    return mapping.get(slot, slot)


def _doc_type_label(doc_type: str) -> str:
    mapping = {
        "transport_ticket": "交通票据",
        "transport_payment": "交通支付记录",
        "flight_detail": "机票明细",
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "unknown": "未知",
    }
    return mapping.get(doc_type, doc_type)


def _build_travel_agent_status(assignment: dict[str, Any]) -> dict[str, Any]:
    required_slots = [
        ("go_ticket", "去程机票发票/票据"),
        ("go_payment", "去程支付记录"),
        ("go_detail", "去程机票明细"),
        ("return_ticket", "返程机票发票/票据"),
        ("return_payment", "返程支付记录"),
        ("return_detail", "返程机票明细"),
        ("hotel_invoice", "酒店发票"),
        ("hotel_payment", "酒店支付记录"),
        ("hotel_order", "酒店订单截图"),
    ]
    missing = [label for key, label in required_slots if not _as_uploaded_list(assignment.get(key))]

    issues: list[str] = []
    comparisons = [
        ("去程交通", assignment.get("go_ticket_amount"), assignment.get("go_payment_amount")),
        ("返程交通", assignment.get("return_ticket_amount"), assignment.get("return_payment_amount")),
        ("酒店", assignment.get("hotel_invoice_amount"), assignment.get("hotel_payment_amount")),
    ]
    for name, left, right in comparisons:
        if left is None or right is None:
            continue
        if abs(float(left) - float(right)) > 0.01:
            issues.append(f"{name}票据金额与支付记录金额不一致：{_format_amount(left)} vs {_format_amount(right)}")

    unknown_files = _as_uploaded_list(assignment.get("unknown"))
    tips: list[str] = []
    if unknown_files:
        tips.append(f"有 {len(unknown_files)} 份材料尚未识别到明确类型，可在聊天区说明用途后重传。")

    complete = not missing and not issues
    return {"missing": missing, "issues": issues, "tips": tips, "complete": complete}


def _merge_uploaded_lists(first: list[Any], second: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for item in list(first) + list(second):
        name = str(getattr(item, "name", ""))
        size = str(getattr(item, "size", ""))
        key = f"{name}:{size}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _target_doc_type_from_user_text(user_text: str, file_name: str) -> str | None:
    text = (user_text or "").lower()
    name = (file_name or "").lower()
    merged = f"{text} {name}"

    if any(key in merged for key in ["订单截图", "酒店订单", "订单图", "hotel order"]):
        return "hotel_order"
    if any(key in merged for key in ["机票明细", "行程单", "客票行程", "itinerary", "detail"]):
        return "flight_detail"
    if any(key in merged for key in ["酒店支付", "酒店支付记录", "酒店支付凭证"]):
        return "hotel_payment"
    if any(key in merged for key in ["交通支付", "机票支付", "高铁支付", "支付记录", "支付凭证"]):
        if any(key in merged for key in ["酒店", "住宿"]):
            return "hotel_payment"
        return "transport_payment"
    if any(key in merged for key in ["酒店发票", "住宿发票"]):
        return "hotel_invoice"
    if any(key in merged for key in ["机票发票", "高铁报销凭证", "交通发票", "交通票据"]):
        return "transport_ticket"
    if "发票" in merged:
        if any(key in merged for key in ["酒店", "住宿"]):
            return "hotel_invoice"
        return "transport_ticket"
    return None


def _parse_relabel_count_hint(user_text: str, max_count: int) -> int:
    if max_count <= 0:
        return 0
    text = (user_text or "").strip()
    if not text:
        return 0

    if any(key in text for key in ["全部", "所有", "全都", "都改", "都归类", "都算", "全部未知", "所有未知"]):
        return max_count

    digit_match = re.search(r"(\d{1,3})\s*(?:个|份|张|条|项)?", text)
    if digit_match:
        try:
            value = int(digit_match.group(1))
            if value > 0:
                return min(max_count, value)
        except ValueError:
            pass

    for zh, value in CHINESE_NUM_MAP.items():
        if any(token in text for token in [f"这{zh}", f"{zh}个", f"{zh}份", f"{zh}张", f"{zh}条", f"{zh}项"]):
            return min(max_count, value)

    return max_count


def _apply_manual_relabel_from_user_text(user_text: str, profiles: list[dict[str, Any]]) -> tuple[int, list[str], str | None]:
    text = (user_text or "").strip()
    if not text:
        return 0, [], None

    target = _target_doc_type_from_user_text(text, "")

    matched_profiles = []
    for profile in profiles:
        name = str(profile.get("name") or "")
        if name and name in text:
            matched_profiles.append(profile)
            if target is None:
                target = _target_doc_type_from_user_text(text, name)

    if not matched_profiles and target and any(key in text for key in ["未知", "未识别", "unknown"]):
        unknown_profiles = [p for p in profiles if str(p.get("doc_type") or "") == "unknown"]
        if unknown_profiles:
            count = _parse_relabel_count_hint(text, len(unknown_profiles))
            matched_profiles = unknown_profiles[:count]

    if not matched_profiles or not target:
        return 0, [], None

    if target not in TRAVEL_DOC_TYPES:
        return 0, [], None

    changed_names: list[str] = []
    for profile in matched_profiles:
        if str(profile.get("doc_type") or "") == target:
            continue
        profile["doc_type"] = target
        profile["source"] = "manual_chat"
        changed_names.append(str(profile.get("name") or ""))

    return len(changed_names), changed_names, target


def _is_reclassify_command(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    markers = ["重新识别", "再识别", "重识别", "重新分类", "再分类", "重新判定", "重跑识别"]
    return any(marker in text for marker in markers)


def _extract_target_profiles_for_reclassify(user_text: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = (user_text or "").strip()
    if not text:
        return []

    lower_text = text.lower()
    sorted_profiles = sorted(profiles, key=lambda p: len(str(p.get("name") or "")), reverse=True)
    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_profile(profile: dict[str, Any]) -> None:
        pid = str(profile.get("profile_id") or "")
        if not pid or pid in seen_ids:
            return
        seen_ids.add(pid)
        matched.append(profile)

    for profile in sorted_profiles:
        name = str(profile.get("name") or "")
        if not name:
            continue
        if name.lower() in lower_text:
            _add_profile(profile)

    tail_match = re.search(r"(?:重新识别|再识别|重识别|重新分类|再分类|重新判定|重跑识别)\s*[:：]?\s*(.+)$", text, flags=re.IGNORECASE)
    if tail_match:
        tail = tail_match.group(1).strip().strip("`\"'“”")
        if tail:
            for profile in sorted_profiles:
                name = str(profile.get("name") or "")
                if not name:
                    continue
                if tail in name or name in tail:
                    _add_profile(profile)

    if not matched and any(key in lower_text for key in ["未知", "未识别", "unknown"]):
        for profile in profiles:
            if str(profile.get("doc_type") or "") == "unknown":
                _add_profile(profile)

    return matched


def _rerun_profile_recognition(profile: dict[str, Any], retry_tag: str) -> dict[str, Any]:
    uploaded_file = profile.get("file")
    if uploaded_file is None:
        return profile
    index = int(profile.get("index") or 0)
    return _recognize_travel_file(uploaded_file, index=index, retry_tag=retry_tag)


def _apply_reclassify_from_user_text(
    user_text: str,
    profiles: list[dict[str, Any]],
    manual_overrides: dict[str, str] | None = None,
) -> tuple[int, list[str], str | None]:
    if not _is_reclassify_command(user_text):
        return 0, [], None

    targets = _extract_target_profiles_for_reclassify(user_text, profiles)
    if not targets:
        return 0, [], "没有匹配到可重新识别的文件名。请直接写完整文件名，例如：`重新识别 2360元_支付凭证长春到上海机票4人.jpg`。"

    detail_lines: list[str] = []
    total = len(targets)
    for idx, profile in enumerate(targets, start=1):
        if manual_overrides is not None:
            _remove_manual_override_for_profile(manual_overrides, profile)
        name = str(profile.get("name") or "")
        before_doc = str(profile.get("doc_type") or "unknown")
        before_label = _doc_type_label(before_doc)
        refreshed = _rerun_profile_recognition(
            profile,
            retry_tag=f"manual_recheck_{datetime.now().timestamp()}_{idx}",
        )

        # In-place overwrite to keep session list object stable.
        for key in [
            "profile_id",
            "index",
            "file",
            "name",
            "suffix",
            "doc_type",
            "amount",
            "date_obj",
            "date",
            "slot",
            "source",
            "confidence",
            "evidence",
        ]:
            profile[key] = refreshed.get(key)

        after_doc = str(profile.get("doc_type") or "unknown")
        after_label = _doc_type_label(after_doc)
        source = str(profile.get("source") or "")
        detail_lines.append(f"{name}: {before_label} -> {after_label}（来源：{source}）")

    return total, detail_lines, None


def _generate_travel_agent_reply_rule(
    user_text: str,
    assignment: dict[str, Any],
    status: dict[str, Any],
    profiles: list[dict[str, Any]] | None = None,
) -> str:
    text = (user_text or "").strip()
    if not text:
        return "请继续描述你的问题，比如“我还缺什么”或“有哪些金额不一致”。"
    lower = text.lower()

    if lower in {"你好", "您好", "hi", "hello"}:
        return "你好。你可以直接问：还缺什么、哪里金额不一致、每个文件分到了哪里。"

    if any(key in text for key in ["缺", "补", "全不全", "齐不齐", "还差"]):
        if status["missing"]:
            return "当前还缺这些材料：\n- " + "\n- ".join(status["missing"])
        return "当前必需材料已齐全。"

    if any(key in text for key in ["问题", "风险", "不一致", "金额"]):
        if status["issues"]:
            return "当前发现的问题：\n- " + "\n- ".join(status["issues"])
        return "目前未发现金额不一致问题。"

    if any(key in text for key in ["分配", "归类", "对应", "怎么放", "分到", "放到"]):
        lines = []
        for slot in [
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
        ]:
            files = _as_uploaded_list(assignment.get(slot))
            if files:
                names = "、".join(str(getattr(f, "name", "")) for f in files[:5])
                if len(files) > 5:
                    names += f" 等{len(files)}份"
                lines.append(f"- {_slot_label(slot)}：{names}")
        if not lines:
            return "当前还没有可分配的材料。"
        return "我目前的分配结果如下：\n" + "\n".join(lines)

    if any(key in text for key in ["为什么", "原因"]) and profiles:
        matched = []
        for profile in profiles:
            name = str(profile.get("name") or "")
            if name and name in text:
                matched.append(profile)
        if matched:
            lines = []
            for profile in matched[:8]:
                lines.append(
                    f"- {profile.get('name')} -> {_doc_type_label(str(profile.get('doc_type') or 'unknown'))}"
                    f"（来源：{profile.get('source') or 'unknown'}）"
                )
            return "当前文件识别说明：\n" + "\n".join(lines)

    if any(key in text for key in ["导出", "打包", "压缩包"]):
        return "可以直接在下方“差旅材料打包导出”输入压缩包名称，然后点击“导出差旅材料压缩包”。"

    summary = "材料整理完成。"
    if status["missing"]:
        summary += f" 目前缺 {len(status['missing'])} 项。"
    if status["issues"]:
        summary += f" 发现 {len(status['issues'])} 个金额核对问题。"
    if not status["missing"] and not status["issues"]:
        summary += " 当前可进入导出。"
    return summary


def _build_travel_agent_context_text(assignment: dict[str, Any], status: dict[str, Any], profiles: list[dict[str, Any]]) -> str:
    slot_keys = [
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
    ]
    slot_summary = {}
    for key in slot_keys:
        files = _as_uploaded_list(assignment.get(key))
        slot_summary[key] = [str(getattr(f, "name", "")) for f in files[:20]]

    profile_summary = []
    for profile in profiles[:80]:
        profile_summary.append(
            {
                "name": profile.get("name"),
                "doc_type": profile.get("doc_type"),
                "slot": profile.get("slot"),
                "amount": _format_amount(_safe_float(profile.get("amount"))),
                "date": profile.get("date"),
                "source": profile.get("source"),
                "confidence": _normalize_confidence(profile.get("confidence")),
                "evidence": str(profile.get("evidence") or ""),
            }
        )

    context = {
        "missing": status.get("missing", []),
        "issues": status.get("issues", []),
        "tips": status.get("tips", []),
        "slot_summary": slot_summary,
        "amounts": {
            "go_ticket_amount": assignment.get("go_ticket_amount"),
            "go_payment_amount": assignment.get("go_payment_amount"),
            "return_ticket_amount": assignment.get("return_ticket_amount"),
            "return_payment_amount": assignment.get("return_payment_amount"),
            "hotel_invoice_amount": assignment.get("hotel_invoice_amount"),
            "hotel_payment_amount": assignment.get("hotel_payment_amount"),
        },
        "profiles": profile_summary,
    }
    return json.dumps(context, ensure_ascii=False)


def _generate_travel_agent_reply_llm(
    user_text: str,
    assignment: dict[str, Any],
    status: dict[str, Any],
    profiles: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

    system_prompt = (
        "你是差旅报销Agent。基于给定上下文回答用户问题，禁止编造未提供的信息。"
        "优先回答：缺件、金额核对问题、分配结果、下一步补件建议。"
        "可以引用制度片段或历史样例，但只能使用已提供RAG上下文。"
        "不要重复欢迎语，不要复述“已进入模式”等固定句。"
        "输出中文，简洁，用要点列表。"
    )
    context_text = _build_travel_agent_context_text(assignment, status, profiles)
    rag_context = ""
    try:
        rag_query = f"{user_text}\n{context_text[:1200]}"
        rag_context = rag_retriever.build_travel_policy_context(rag_query, top_k=3)
    except Exception:
        rag_context = ""

    llm_messages = [{"role": "system", "content": system_prompt}]
    # Keep short recent history to retain context while controlling token size.
    for item in messages[-6:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        llm_messages.append({"role": role, "content": content[:1200]})
    llm_messages.append(
        {
            "role": "user",
            "content": (
                f"当前材料上下文(JSON)：\n{context_text}\n\n"
                f"RAG上下文：\n{rag_context or '无'}\n\n"
                f"用户问题：{user_text}"
            ),
        }
    )

    invalid_markers = ["我已进入差旅材料整理模式", "可以问我", "还缺什么、哪里金额不一致"]

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": llm_messages,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()
        if content and not any(marker in content for marker in invalid_markers):
            return content
    except Exception:
        pass

    # Fallback for Ollama versions/models that don't support /api/chat reliably.
    history_lines = []
    for item in llm_messages:
        role = item.get("role")
        content = str(item.get("content") or "")
        if not content:
            continue
        history_lines.append(f"[{role}] {content}")
    prompt = "\n\n".join(history_lines[-8:])

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("response") or "").strip()
        if content and not any(marker in content for marker in invalid_markers):
            return content
    except Exception:
        return None

    return None


def _render_travel_conversation_agent() -> dict[str, Any]:
    st.subheader("会话式差旅 Agent")
    st.caption("把本次差旅材料一次性上传，Agent 会自动归类、提示缺件，并支持继续对话追问。")
    st.caption("当前模式：材料识别=内容优先（LLM + Rule Guard）；对话=本地LLM（Ollama）")

    pool_files = st.file_uploader(
        "上传本次差旅全部材料（PDF/图片，可多选）",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key="travel_agent_pool_files",
    )
    pool_list = _as_uploaded_list(pool_files)
    manual_overrides = st.session_state.setdefault("travel_agent_manual_overrides", {})
    if not isinstance(manual_overrides, dict):
        manual_overrides = {}
        st.session_state["travel_agent_manual_overrides"] = manual_overrides
    _prune_manual_overrides(manual_overrides, pool_list)
    current_signature = _files_signature(pool_list)

    action_left, action_mid, action_right = st.columns(3)
    refresh_clicked = action_left.button("Agent整理材料", use_container_width=True, key="travel_agent_refresh")
    clear_chat_clicked = action_mid.button("清空会话记录", use_container_width=True, key="travel_agent_clear_chat")
    clear_cache_clicked = action_right.button("清空识别缓存", use_container_width=True, key="travel_agent_clear_cache")

    if clear_chat_clicked:
        st.session_state.pop("travel_agent_messages", None)
    if clear_cache_clicked:
        st.cache_data.clear()
        st.session_state.pop("travel_agent_pool_signature", None)
        st.session_state.pop("travel_agent_assignment", None)
        st.session_state.pop("travel_agent_profiles", None)
        st.session_state.pop("travel_agent_manual_overrides", None)
        st.success("识别缓存已清空，请重新点击“Agent整理材料”。")
        st.rerun()

    need_rebuild = refresh_clicked or st.session_state.get("travel_agent_pool_signature") != current_signature
    if need_rebuild:
        with st.spinner("Agent 正在识别并分配材料..."):
            assignment, profiles = _organize_travel_materials(pool_list, manual_overrides=manual_overrides)
        st.session_state["travel_agent_pool_signature"] = current_signature
        st.session_state["travel_agent_assignment"] = assignment
        st.session_state["travel_agent_profiles"] = profiles

    assignment = st.session_state.get("travel_agent_assignment", {})
    profiles = st.session_state.get("travel_agent_profiles", [])
    if pool_list and not assignment:
        assignment, profiles = _organize_travel_materials(pool_list, manual_overrides=manual_overrides)
        st.session_state["travel_agent_pool_signature"] = current_signature
        st.session_state["travel_agent_assignment"] = assignment
        st.session_state["travel_agent_profiles"] = profiles

    if not pool_list:
        st.info("请先上传差旅材料，我会自动分类到去程/返程/酒店，并告诉你缺什么。")
        return {"missing": [], "issues": [], "tips": [], "complete": False}

    status = _build_travel_agent_status(assignment)
    if status["missing"]:
        st.warning("仍缺材料：" + "、".join(status["missing"]))
    else:
        st.success("必需材料已齐全。")

    if status["issues"]:
        st.error("发现核对问题：")
        for issue in status["issues"]:
            st.markdown(f"- {issue}")
    elif status["complete"]:
        st.success("金额核对通过，可以导出材料。")

    for tip in status["tips"]:
        st.info(tip)

    profile_rows = []
    for profile in profiles:
        confidence = _normalize_confidence(profile.get("confidence"))
        evidence = str(profile.get("evidence") or "").strip()
        if len(evidence) > 60:
            evidence = evidence[:57] + "..."
        profile_rows.append(
            {
                "文件名": profile.get("name"),
                "识别类型": _doc_type_label(str(profile.get("doc_type") or "unknown")),
                "分配槽位": _slot_label(str(profile.get("slot") or "unknown")),
                "金额": _format_amount(_safe_float(profile.get("amount"))),
                "日期": profile.get("date") or "",
                "识别来源": profile.get("source") or "",
                "置信度": f"{confidence * 100:.0f}%" if confidence is not None else "",
                "识别依据": evidence,
            }
        )
    if profile_rows:
        st.dataframe(profile_rows, use_container_width=True, hide_index=True)
    st.caption(
        "可直接在对话里纠正分类，例如：`朱洪良.pdf 应该是机票发票`、`某某.jpg 应该是酒店订单截图`、"
        "`未知这三个是机票明细`、`重新识别 2360元_支付凭证长春到上海机票4人.jpg`。"
    )

    if profiles:
        doc_type_to_label = {
            "transport_ticket": "交通票据",
            "transport_payment": "交通支付记录",
            "flight_detail": "机票明细",
            "hotel_invoice": "酒店发票",
            "hotel_payment": "酒店支付记录",
            "hotel_order": "酒店订单截图",
            "unknown": "未知",
        }
        label_to_doc_type = {v: k for k, v in doc_type_to_label.items()}
        with st.expander("手工修正分类（可选）", expanded=False):
            edit_rows = []
            for profile in profiles:
                edit_rows.append(
                    {
                        "profile_id": profile.get("profile_id"),
                        "文件名": profile.get("name"),
                        "识别类型": doc_type_to_label.get(str(profile.get("doc_type") or "unknown"), "未知"),
                    }
                )
            edited_rows = st.data_editor(
                edit_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "profile_id": st.column_config.TextColumn("profile_id", disabled=True, width="small"),
                    "文件名": st.column_config.TextColumn("文件名", disabled=True, width="large"),
                    "识别类型": st.column_config.SelectboxColumn(
                        "识别类型",
                        options=list(doc_type_to_label.values()),
                        required=True,
                        width="medium",
                    ),
                },
                key="travel_agent_manual_editor",
            )
            if st.button("应用修正并重新分配", key="travel_agent_apply_manual", use_container_width=True):
                id_to_profile = {str(p.get("profile_id")): p for p in profiles}
                changed = 0
                for row in edited_rows:
                    pid = str(row.get("profile_id") or "")
                    label = str(row.get("识别类型") or "")
                    target_doc_type = label_to_doc_type.get(label, "unknown")
                    profile = id_to_profile.get(pid)
                    if not profile:
                        continue
                    if str(profile.get("doc_type") or "") == target_doc_type:
                        continue
                    profile["doc_type"] = target_doc_type
                    profile["source"] = "manual_table"
                    _set_manual_override_for_profile(manual_overrides, profile)
                    changed += 1

                if changed > 0:
                    assignment = _build_assignment_from_profiles(profiles)
                    st.session_state["travel_agent_assignment"] = assignment
                    st.session_state["travel_agent_profiles"] = profiles
                    try:
                        learning.learn_from_travel_profiles(profiles, assignment, reason="manual_table")
                    except Exception:
                        pass
                    st.success(f"已应用 {changed} 条分类修正。")
                    st.rerun()
                else:
                    st.info("没有检测到分类变更。")

    messages = st.session_state.setdefault("travel_agent_messages", [])
    if not messages:
        messages.append(
            {
                "role": "assistant",
                "content": "我已进入差旅材料整理模式。你可以问我：还缺什么、哪里金额不一致、当前是怎么分配的。",
            }
        )

    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_input = st.chat_input("例如：我现在还缺什么？", key="travel_agent_chat_input")
    if user_input:
        messages.append({"role": "user", "content": user_input})

        # Chat command: force re-recognize selected files.
        if _is_reclassify_command(user_input):
            with st.spinner("正在重新识别指定材料..."):
                recheck_count, recheck_lines, recheck_error = _apply_reclassify_from_user_text(
                    user_input,
                    profiles,
                    manual_overrides=manual_overrides,
                )
            if recheck_error:
                messages.append({"role": "assistant", "content": recheck_error})
                st.rerun()
            if recheck_count > 0:
                assignment = _build_assignment_from_profiles(profiles)
                status = _build_travel_agent_status(assignment)
                st.session_state["travel_agent_assignment"] = assignment
                st.session_state["travel_agent_profiles"] = profiles
                preview = "\n".join(f"- {line}" for line in recheck_lines[:8])
                if len(recheck_lines) > 8:
                    preview += f"\n- ... 共 {len(recheck_lines)} 个文件"
                reply = (
                    f"已重新识别 {recheck_count} 个文件：\n{preview}\n\n"
                    f"当前仍缺：{'、'.join(status.get('missing', [])) if status.get('missing') else '无'}。"
                )
                messages.append({"role": "assistant", "content": reply})
                st.rerun()

        changed_count, changed_names, target_doc_type = _apply_manual_relabel_from_user_text(user_input, profiles)
        if changed_count > 0:
            _remember_manual_overrides(manual_overrides, profiles)
            assignment = _build_assignment_from_profiles(profiles)
            status = _build_travel_agent_status(assignment)
            st.session_state["travel_agent_assignment"] = assignment
            st.session_state["travel_agent_profiles"] = profiles
            try:
                learning.learn_from_travel_profiles(profiles, assignment, reason="manual_chat")
            except Exception:
                pass
            changed_preview = "、".join(changed_names[:3])
            if changed_count > 3:
                changed_preview += f" 等{changed_count}个文件"
            reply = (
                f"已按你的指令修正分类为 `{_doc_type_label(str(target_doc_type or 'unknown'))}`：{changed_preview}。\n\n"
                f"当前仍缺：{'、'.join(status.get('missing', [])) if status.get('missing') else '无'}。"
            )
            messages.append({"role": "assistant", "content": reply})
            st.rerun()

        reply = _generate_travel_agent_reply_llm(user_input, assignment, status, profiles, messages)
        if not reply:
            reply = (
                "LLM 当前未返回有效结果。请检查本地 Ollama 服务与模型状态：\n"
                "- `ollama ps` 是否有正在运行模型\n"
                "- `OLLAMA_BASE_URL` 与端口是否正确\n"
                "- 可设置 `OLLAMA_CHAT_MODEL` 为稳定文本模型后重试"
            )
        messages.append({"role": "assistant", "content": reply})
        st.rerun()

    return status


def _sanitize_export_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "差旅报销材料"


def _safe_uploaded_filename(name: str, default_stem: str) -> str:
    raw = Path(name or "").name
    if not raw:
        raw = default_stem
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", raw).strip()
    return cleaned or default_stem


def _amount_suffix(amount: float | None) -> str:
    if amount is None:
        return "金额未知"
    if abs(amount - round(amount)) <= 0.01:
        return f"{int(round(amount))}元"
    return f"{amount:.2f}元"


def _zip_ensure_dir(zip_file: zipfile.ZipFile, dir_path: str) -> None:
    normalized = dir_path.replace("\\", "/").rstrip("/") + "/"
    zip_file.writestr(normalized, b"")


def _zip_write_uploaded_files(zip_file: zipfile.ZipFile, target_dir: str, files: list[Any]) -> None:
    _zip_ensure_dir(zip_file, target_dir)
    for idx, uploaded in enumerate(files, start=1):
        original_name = str(getattr(uploaded, "name", ""))
        safe_name = _safe_uploaded_filename(original_name, f"file_{idx}")
        stored_name = f"{idx:02d}_{safe_name}"
        zip_file.writestr(f"{target_dir}/{stored_name}", uploaded.getvalue())


def _build_travel_package_zip(
    package_name: str,
    go_ticket_files: list[Any],
    go_payment_files: list[Any],
    go_detail_files: list[Any],
    return_ticket_files: list[Any],
    return_payment_files: list[Any],
    return_detail_files: list[Any],
    hotel_invoice_files: list[Any],
    hotel_payment_files: list[Any],
    hotel_order_files: list[Any],
    go_ticket_amount: float | None,
    go_payment_amount: float | None,
    return_ticket_amount: float | None,
    return_payment_amount: float | None,
    hotel_invoice_amount: float | None,
    hotel_payment_amount: float | None,
) -> bytes:
    root_name = _sanitize_export_name(package_name)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        go_root = f"{root_name}/出差去程交通报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程机票发票_{_amount_suffix(go_ticket_amount)}",
            go_ticket_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程支付记录_{_amount_suffix(go_payment_amount)}",
            go_payment_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{go_root}/去程机票明细",
            go_detail_files,
        )

        return_root = f"{root_name}/出差返程交通报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程机票发票_{_amount_suffix(return_ticket_amount)}",
            return_ticket_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程支付记录_{_amount_suffix(return_payment_amount)}",
            return_payment_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{return_root}/返程机票明细",
            return_detail_files,
        )

        hotel_root = f"{root_name}/酒店报销"
        _zip_write_uploaded_files(
            zip_file,
            f"{hotel_root}/酒店发票_{_amount_suffix(hotel_invoice_amount)}",
            hotel_invoice_files,
        )
        _zip_write_uploaded_files(
            zip_file,
            f"{hotel_root}/支付记录_{_amount_suffix(hotel_payment_amount)}",
            hotel_payment_files,
        )
        _zip_write_uploaded_files(zip_file, f"{hotel_root}/订单截图", hotel_order_files)
    buffer.seek(0)
    return buffer.getvalue()


def _render_travel_package_export() -> None:
    st.subheader("差旅材料打包导出")
    default_name = f"差旅报销材料_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_name = st.text_input(
        "压缩包名称（无需 .zip）",
        value=default_name,
        key="travel_export_package_name",
    )

    assignment = st.session_state.get("travel_agent_assignment") or {}

    go_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_ticket")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_file")),
    )
    go_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_payment")),
        _as_uploaded_list(st.session_state.get("travel_go_payment_file")),
    )
    go_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("go_detail")),
        _as_uploaded_list(st.session_state.get("travel_go_ticket_detail_file")),
    )

    return_ticket_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_ticket")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_file")),
    )
    return_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_payment")),
        _as_uploaded_list(st.session_state.get("travel_return_payment_file")),
    )
    return_detail_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("return_detail")),
        _as_uploaded_list(st.session_state.get("travel_return_ticket_detail_file")),
    )

    hotel_invoice_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_invoice")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_invoice")),
    )
    hotel_payment_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_payment")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_payment")),
    )
    hotel_order_files = _merge_uploaded_lists(
        _as_uploaded_list(assignment.get("hotel_order")),
        _as_uploaded_list(st.session_state.get("travel_hotel_hotel_order")),
    )

    go_ticket_amount = _safe_float(st.session_state.get("travel_go_ticket_amount"))
    go_payment_amount = _safe_float(st.session_state.get("travel_go_payment_amount"))
    return_ticket_amount = _safe_float(st.session_state.get("travel_return_ticket_amount"))
    return_payment_amount = _safe_float(st.session_state.get("travel_return_payment_amount"))
    hotel_invoice_amount = _safe_float(st.session_state.get("travel_hotel_ticket_amount"))
    hotel_payment_amount = _safe_float(st.session_state.get("travel_hotel_payment_amount"))

    if go_ticket_amount is None:
        go_ticket_amount = _safe_float(assignment.get("go_ticket_amount"))
    if go_payment_amount is None:
        go_payment_amount = _safe_float(assignment.get("go_payment_amount"))
    if return_ticket_amount is None:
        return_ticket_amount = _safe_float(assignment.get("return_ticket_amount"))
    if return_payment_amount is None:
        return_payment_amount = _safe_float(assignment.get("return_payment_amount"))
    if hotel_invoice_amount is None:
        hotel_invoice_amount = _safe_float(assignment.get("hotel_invoice_amount"))
    if hotel_payment_amount is None:
        hotel_payment_amount = _safe_float(assignment.get("hotel_payment_amount"))

    total_uploaded = (
        len(go_ticket_files)
        + len(go_payment_files)
        + len(go_detail_files)
        + len(return_ticket_files)
        + len(return_payment_files)
        + len(return_detail_files)
        + len(hotel_invoice_files)
        + len(hotel_payment_files)
        + len(hotel_order_files)
    )
    if total_uploaded == 0:
        st.info("请先上传差旅材料，再导出压缩包。")
        return

    missing_for_recommended = []
    if not go_ticket_files:
        missing_for_recommended.append("去程机票发票")
    if not go_payment_files:
        missing_for_recommended.append("去程支付记录")
    if not go_detail_files:
        missing_for_recommended.append("去程机票明细")
    if not return_ticket_files:
        missing_for_recommended.append("返程机票发票")
    if not return_payment_files:
        missing_for_recommended.append("返程支付记录")
    if not return_detail_files:
        missing_for_recommended.append("返程机票明细")
    if not hotel_invoice_files:
        missing_for_recommended.append("酒店发票")
    if not hotel_payment_files:
        missing_for_recommended.append("酒店支付记录")
    if not hotel_order_files:
        missing_for_recommended.append("酒店订单截图")

    if missing_for_recommended:
        st.warning(f"当前仍缺少：{'、'.join(missing_for_recommended)}。你仍可先导出已有材料。")

    zip_bytes = _build_travel_package_zip(
        package_name=package_name,
        go_ticket_files=go_ticket_files,
        go_payment_files=go_payment_files,
        go_detail_files=go_detail_files,
        return_ticket_files=return_ticket_files,
        return_payment_files=return_payment_files,
        return_detail_files=return_detail_files,
        hotel_invoice_files=hotel_invoice_files,
        hotel_payment_files=hotel_payment_files,
        hotel_order_files=hotel_order_files,
        go_ticket_amount=go_ticket_amount,
        go_payment_amount=go_payment_amount,
        return_ticket_amount=return_ticket_amount,
        return_payment_amount=return_payment_amount,
        hotel_invoice_amount=hotel_invoice_amount,
        hotel_payment_amount=hotel_payment_amount,
    )
    zip_file_name = f"{_sanitize_export_name(package_name)}.zip"
    st.download_button(
        label="导出差旅材料压缩包",
        data=zip_bytes,
        file_name=zip_file_name,
        mime="application/zip",
        use_container_width=True,
    )


def _render_export_download(task, key_scope: str = "default") -> None:
    excel_path = task.export_excel_path
    text_path = task.export_text_path

    if not excel_path and not text_path:
        st.info("当前任务还没有导出文件。可点击“重新处理”或“导出 Excel + 文本”。")
        return

    st.markdown("**导出文件**")
    cols = st.columns(2)

    if excel_path:
        excel_file = Path(excel_path)
        if excel_file.exists():
            with excel_file.open("rb") as fp:
                cols[0].download_button(
                    label="下载 Excel",
                    data=fp.read(),
                    file_name=excel_file.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_excel_{key_scope}_{task.id}",
                )
        else:
            cols[0].warning("Excel 路径已记录，但文件不存在。")

    if text_path:
        text_file = Path(text_path)
        if text_file.exists():
            with text_file.open("rb") as fp:
                cols[1].download_button(
                    label="下载文本",
                    data=fp.read(),
                    file_name=text_file.name,
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_text_{key_scope}_{task.id}",
                )
        else:
            cols[1].warning("文本路径已记录，但文件不存在。")


MATERIAL_AGENT_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax"]


def _material_agent_extract_fields(task) -> dict[str, Any]:
    extracted = dict(task.extracted_data or {})
    rows = _normalize_line_items(_to_editor_rows(extracted.get("line_items")))
    extracted["line_items"] = rows

    if not str(extracted.get("item_content") or "").strip() and rows:
        names = [str(row.get("item_name") or "").strip() for row in rows if str(row.get("item_name") or "").strip()]
        extracted["item_content"] = "；".join(names[:8])

    amount_text = str(extracted.get("amount") or "").strip()
    if not amount_text:
        total = _line_items_total(rows)
        if total is not None:
            extracted["amount"] = _format_amount(total)
    return extracted


def _material_agent_build_fields_payload(fields: dict[str, Any]) -> dict[str, Any]:
    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    amount_text = str(fields.get("amount") or "").strip()
    if not amount_text:
        total = _line_items_total(rows)
        if total is not None:
            amount_text = _format_amount(total)

    item_content = str(fields.get("item_content") or "").strip()
    if not item_content and rows:
        names = [str(row.get("item_name") or "").strip() for row in rows if str(row.get("item_name") or "").strip()]
        item_content = "；".join(names[:8])

    return {
        "invoice_number": str(fields.get("invoice_number") or "").strip() or None,
        "invoice_date": str(fields.get("invoice_date") or "").strip() or None,
        "amount": amount_text or None,
        "tax_amount": str(fields.get("tax_amount") or "").strip() or None,
        "seller": str(fields.get("seller") or "").strip() or None,
        "buyer": str(fields.get("buyer") or "").strip() or None,
        "bill_type": str(fields.get("bill_type") or "").strip() or None,
        "item_content": item_content or None,
        "line_items": rows,
    }


def _material_agent_apply_updates(task_id: str, fields: dict[str, Any]) -> tuple[bool, str]:
    try:
        corrections = {
            "expense_category": "材料费",
            "extracted_fields": _material_agent_build_fields_payload(fields),
        }
        local_runner.apply_corrections(task_id, corrections)
        local_runner.export_task(task_id, export_format="both")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _material_agent_quality_hints(fields: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    if not rows:
        hints.append("未识别到明细行，建议用“重新识别”或手工新增行。")
        return hints

    amount_value = _safe_float(fields.get("amount"))
    row_total = _line_items_total(rows)
    if amount_value is not None and row_total is not None and abs(amount_value - row_total) > 0.1:
        hints.append(f"发票总金额与明细合计不一致：{_format_amount(amount_value)} vs {_format_amount(row_total)}")

    long_stats = fields.get("long_mode_stats")
    if isinstance(long_stats, dict):
        candidate_rows = int(long_stats.get("candidate_rows") or 0)
        final_rows = int(long_stats.get("final_rows") or 0)
        if candidate_rows > final_rows > 0:
            hints.append(f"长票候选行 {candidate_rows}，最终行 {final_rows}，可能仍有漏项。")

    for idx, row in enumerate(rows, start=1):
        name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        if spec and name and spec in name and len(spec) >= 3:
            hints.append(f"第{idx}行项目名称可能混入规格：{name}")
            if len(hints) >= 4:
                break
    return hints


def _material_agent_split_name_spec(name: str, spec: str) -> tuple[str, str]:
    new_name = str(name or "").strip()
    new_spec = str(spec or "").strip()

    splitter = getattr(extractor, "_split_item_name_and_spec", None)
    refiner = getattr(extractor, "_refine_name_spec_boundary", None)
    if callable(splitter):
        try:
            new_name, new_spec = splitter(new_name, new_spec)
        except Exception:
            pass
    if callable(refiner):
        try:
            new_name, new_spec = refiner(new_name, new_spec)
        except Exception:
            pass

    if not new_spec:
        match = re.search(
            r"(M\d+(?:[Xx*]\d+(?:\.\d+)?)?|[A-Za-z]{2,}\d[\w\-./]*|\d+(?:\.\d+)?(?:mm|cm|m|kg|g|V|W|A|Hz)|\d+(?:\.\d+)?\s*(?:-|~|～|x|X|[*])\s*\d+(?:\.\d+)?(?:mm|cm|m)?)$",
            new_name,
        )
        if match:
            candidate = match.group(1).strip()
            head = new_name[: match.start()].strip(" -_/，,;；:：")
            if head and candidate and candidate not in {"电子元件", "金属制品"}:
                new_name, new_spec = head, candidate

    if new_spec and new_name and new_name.endswith(new_spec):
        trimmed = new_name[: -len(new_spec)].rstrip(" -_/，,;；:：")
        if trimmed:
            new_name = trimmed

    return new_name, new_spec


def _material_agent_auto_split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    normalized_rows = _normalize_line_items(_to_editor_rows(rows))
    changed = 0
    output: list[dict[str, Any]] = []

    for row in normalized_rows:
        name = str(row.get("item_name") or "").strip()
        spec = str(row.get("spec") or "").strip()
        new_name, new_spec = _material_agent_split_name_spec(name, spec)
        if new_name != name or new_spec != spec:
            changed += 1

        output.append(
            {
                "item_name": new_name,
                "spec": new_spec,
                "quantity": str(row.get("quantity") or "").strip(),
                "unit": str(row.get("unit") or "").strip(),
                "line_total_with_tax": str(row.get("line_total_with_tax") or "").strip(),
            }
        )
    return output, changed


def _material_agent_parse_chinese_number(token: str) -> int | None:
    text = str(token or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    if text == "十":
        return 10
    if text.startswith("十") and len(text) >= 2:
        tail = CHINESE_NUM_MAP.get(text[1:])
        if tail is not None:
            return 10 + tail
    if text.endswith("十") and len(text) >= 2:
        head = CHINESE_NUM_MAP.get(text[:-1])
        if head is not None:
            return head * 10
    if "十" in text:
        parts = text.split("十", 1)
        head = CHINESE_NUM_MAP.get(parts[0], 1 if parts[0] == "" else None)
        tail = CHINESE_NUM_MAP.get(parts[1], 0 if parts[1] == "" else None)
        if head is not None and tail is not None:
            return head * 10 + tail

    return CHINESE_NUM_MAP.get(text)


def _material_agent_resolve_row_index(text: str, row_count: int) -> int | None:
    if row_count <= 0:
        return None
    source = str(text or "")

    if any(token in source for token in ["最后一行", "末行", "最后1行"]):
        return row_count - 1
    if any(token in source for token in ["第一行", "首行"]):
        return 0

    match = re.search(r"倒数第\s*([0-9一二两三四五六七八九十]+)\s*行", source)
    if match:
        n = _material_agent_parse_chinese_number(match.group(1))
        if n is None:
            return None
        idx = row_count - n
        return idx if 0 <= idx < row_count else None

    match = re.search(r"第\s*([0-9一二两三四五六七八九十]+)\s*行", source)
    if match:
        n = _material_agent_parse_chinese_number(match.group(1))
        if n is None:
            return None
        idx = n - 1
        return idx if 0 <= idx < row_count else None

    return None


def _material_agent_extract_row_updates(text: str) -> dict[str, str]:
    field_map = {
        "项目名称": "item_name",
        "项目名": "item_name",
        "规格型号": "spec",
        "规格": "spec",
        "数量": "quantity",
        "单位": "unit",
        "每项含税总价": "line_total_with_tax",
        "含税总价": "line_total_with_tax",
        "金额": "line_total_with_tax",
    }
    labels = "|".join(field_map.keys())
    updates: dict[str, str] = {}

    for alias, target in field_map.items():
        pattern = (
            rf"(?:{re.escape(alias)})\s*(?:应为|改为|设置为|设为|为|=|:|：)\s*(.+?)"
            rf"(?=(?:{labels})\s*(?:应为|改为|设置为|设为|为|=|:|：)|$)"
        )
        match = re.search(pattern, text)
        if not match:
            continue
        value = str(match.group(1) or "").strip().strip("。；;，,")
        if not value:
            continue
        if target == "line_total_with_tax":
            amount_value = _format_amount(_safe_float(value))
            if amount_value:
                updates[target] = amount_value
        else:
            updates[target] = value

    return updates


def _material_agent_looks_like_edit_intent(text: str) -> bool:
    source = str(text or "")
    action_hit = any(
        token in source
        for token in ["改为", "应为", "设置为", "设为", "删除", "新增一行", "添加一行", "分列修复", "重新识别"]
    )
    target_hit = any(
        token in source
        for token in [
            "第",
            "最后一行",
            "倒数第",
            "行",
            "项目名称",
            "项目名",
            "规格",
            "数量",
            "单位",
            "金额",
            "含税总价",
        ]
    )
    return action_hit and target_hit


def _material_agent_apply_chat_command(
    user_text: str,
    task,
    fields: dict[str, Any],
) -> tuple[bool, str, Any, dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return True, "请给我一个明确指令，例如：`第3行规格改为M20X1.5`。", task, fields

    if "重新识别" in text:
        try:
            local_runner.process_task(task.id)
            local_runner.export_task(task.id, export_format="both")
            updated_task = local_runner.get_task(task.id) or task
            updated_fields = _material_agent_extract_fields(updated_task)
            line_count = len(_normalize_line_items(_to_editor_rows(updated_fields.get("line_items"))))
            return True, f"已重新识别，当前识别到明细 {line_count} 行。", updated_task, updated_fields
        except Exception as exc:
            return True, f"重新识别失败：{exc}", task, fields

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    if not rows:
        rows = []

    delete_match = re.search(r"删除\s*(?:第\s*[0-9一二两三四五六七八九十]+\s*行|最后一行|末行|首行|第一行|倒数第\s*[0-9一二两三四五六七八九十]+\s*行)", text)
    if delete_match:
        idx = _material_agent_resolve_row_index(text, len(rows))
        if idx is None:
            return True, "未识别到要删除的行号，请用“第N行/最后一行/倒数第N行”。", task, fields
        if idx < 0 or idx >= len(rows):
            return True, "行号超出范围。", task, fields
        removed = rows.pop(idx)
        new_fields = dict(fields)
        new_fields["line_items"] = rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"删除失败：{err}", task, fields
        updated_task = local_runner.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        return True, f"已删除第{idx + 1}行：{removed.get('item_name', '')}。", updated_task, updated_fields

    if "新增一行" in text or "添加一行" in text:
        key_map = {
            "项目名称": "item_name",
            "项目名": "item_name",
            "规格": "spec",
            "规格型号": "spec",
            "数量": "quantity",
            "单位": "unit",
            "金额": "line_total_with_tax",
            "含税总价": "line_total_with_tax",
            "每项含税总价": "line_total_with_tax",
        }
        pairs = re.findall(r"(项目名称|项目名|规格|规格型号|数量|单位|金额|含税总价|每项含税总价)\s*[=:：]\s*([^,，;；]+)", text)
        payload = {"item_name": "", "spec": "", "quantity": "", "unit": "", "line_total_with_tax": ""}
        for key, raw_value in pairs:
            target = key_map.get(key)
            if not target:
                continue
            value = raw_value.strip()
            if target == "line_total_with_tax":
                value = _format_amount(_safe_float(value))
            payload[target] = value

        if not payload["item_name"]:
            return True, "新增一行至少要提供项目名称。示例：`新增一行 项目名称=电子元件*连接器, 规格=Y50EX, 数量=20, 单位=只, 金额=2396.04`", task, fields

        updated_rows = [dict(row) for row in rows]
        updated_rows.append(payload)
        new_fields = dict(fields)
        new_fields["line_items"] = updated_rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"新增失败：{err}", task, fields
        updated_task = local_runner.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        return True, "已新增1行并保存。", updated_task, updated_fields

    row_updates = _material_agent_extract_row_updates(text)
    if row_updates:
        idx = _material_agent_resolve_row_index(text, len(rows))
        if idx is None:
            return True, "未识别到行号，请用“第N行/最后一行/倒数第N行”。", task, fields
        if idx < 0 or idx >= len(rows):
            return True, "行号超出范围。", task, fields

        updated_rows = [dict(row) for row in rows]
        for field_name, value in row_updates.items():
            updated_rows[idx][field_name] = value

        new_fields = dict(fields)
        new_fields["line_items"] = updated_rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"更新失败：{err}", task, fields
        updated_task = local_runner.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        changed_fields = []
        if "item_name" in row_updates:
            changed_fields.append("项目名称")
        if "spec" in row_updates:
            changed_fields.append("规格型号")
        if "quantity" in row_updates:
            changed_fields.append("数量")
        if "unit" in row_updates:
            changed_fields.append("单位")
        if "line_total_with_tax" in row_updates:
            changed_fields.append("每项含税总价")
        fields_text = "、".join(changed_fields) if changed_fields else "字段"
        return True, f"已更新第{idx + 1}行：{fields_text}。", updated_task, updated_fields

    if any(token in text for token in ["分列修复", "修复项目规格", "修复项目名称和规格", "规格拆分"]):
        fixed_rows, changed = _material_agent_auto_split_rows(rows)
        if changed <= 0:
            return True, "没有检测到可修复的项目名称/规格混杂行。", task, fields
        new_fields = dict(fields)
        new_fields["line_items"] = fixed_rows
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if not ok:
            return True, f"修复失败：{err}", task, fields
        updated_task = local_runner.get_task(task.id) or task
        updated_fields = _material_agent_extract_fields(updated_task)
        return True, f"已完成分列修复，调整 {changed} 行。", updated_task, updated_fields

    if _material_agent_looks_like_edit_intent(text):
        return (
            True,
            "我没有执行任何修改。请按以下格式重试：`第3行规格改为M20X1.5`、`最后一行项目名称应为... 规格型号应为...`、`删除最后一行`。",
            task,
            fields,
        )

    return False, "", task, fields


def _generate_material_agent_reply_llm(
    user_text: str,
    task,
    fields: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    hints = _material_agent_quality_hints(fields)
    raw_text = str(getattr(task, "raw_text", "") or "")

    rag_bundle = rag_retriever.build_material_references(fields, raw_text)
    policy_refs = list(rag_bundle.get("policy_refs") or [])
    case_hits = list(rag_bundle.get("case_hits") or [])
    case_lines = []
    for hit in case_hits[:3]:
        category = str((hit.get("metadata") or {}).get("expense_category") or "")
        case_lines.append(f"{hit.get('score', 0):.2f} | {category} | {str(hit.get('content') or '')[:80]}")

    context = {
        "task_id": task.id,
        "filename": task.original_filename,
        "invoice_number": fields.get("invoice_number"),
        "invoice_date": fields.get("invoice_date"),
        "amount": fields.get("amount"),
        "seller": fields.get("seller"),
        "buyer": fields.get("buyer"),
        "processing_mode": fields.get("processing_mode"),
        "long_mode_stats": fields.get("long_mode_stats"),
        "line_count": len(rows),
        "line_items": rows[:120],
        "quality_hints": hints,
        "policy_refs": policy_refs[:4],
        "case_refs": case_lines,
    }
    context_text = json.dumps(context, ensure_ascii=False)

    system_prompt = (
        "你是材料费报销助手。"
        "你的职责是：检查漏项风险、项目名称与规格是否混杂、金额一致性，并给出下一步可执行建议。"
        "禁止编造未给出的发票字段。"
        "回答简短，使用要点。"
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for item in messages[-6:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        llm_messages.append({"role": role, "content": content[:1200]})

    llm_messages.append(
        {
            "role": "user",
            "content": f"当前材料上下文(JSON)：\n{context_text}\n\n用户问题：{user_text}",
        }
    )

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": llm_messages,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()
        if content:
            return content
    except Exception:
        pass

    prompt = "\n\n".join(f"[{item.get('role')}] {item.get('content')}" for item in llm_messages[-8:])
    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, 90))
        resp.raise_for_status()
        content = (resp.json().get("response") or "").strip()
        if content:
            return content
    except Exception:
        return None

    return None


def _render_material_conversation_agent() -> None:
    st.subheader("2) 材料费会话式 Agent")
    st.caption("上传材料费发票后，Agent 自动抽取并生成明细表。支持对话修正与重新识别，修改后会持续学习。")

    uploaded = st.file_uploader(
        "上传材料费发票（PDF，可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="material_agent_upload_files",
    )
    upload_list = _as_uploaded_list(uploaded)

    task_ids = st.session_state.setdefault("material_agent_task_ids", [])
    if not isinstance(task_ids, list):
        task_ids = []
        st.session_state["material_agent_task_ids"] = task_ids

    action1, action2, action3 = st.columns(3)
    process_clicked = action1.button("Agent识别材料发票", use_container_width=True, key="material_agent_process")
    clear_tasks_clicked = action2.button("清空材料任务缓存", use_container_width=True, key="material_agent_clear_tasks")
    clear_chat_clicked = action3.button("清空材料会话", use_container_width=True, key="material_agent_clear_chat")

    if clear_tasks_clicked:
        st.session_state.pop("material_agent_task_ids", None)
        st.session_state.pop("material_agent_chat_map", None)
        st.success("已清空材料任务缓存。")
        st.rerun()
    if clear_chat_clicked:
        st.session_state.pop("material_agent_chat_map", None)
        st.success("已清空材料会话。")
        st.rerun()

    if process_clicked:
        if not upload_list:
            st.warning("请先上传 PDF。")
        else:
            with st.spinner("正在识别材料发票并生成明细..."):
                new_ids: list[str] = []
                for file in upload_list:
                    task = local_runner.create_and_process_task(
                        file.name,
                        file.getvalue(),
                        auto_process=True,
                        auto_export=True,
                    )
                    if task is not None:
                        new_ids.append(task.id)
                if new_ids:
                    merged = list(dict.fromkeys(new_ids + task_ids))
                    st.session_state["material_agent_task_ids"] = merged
            st.success("材料费任务已更新。")
            st.rerun()

    valid_tasks = []
    for task_id in st.session_state.get("material_agent_task_ids", []):
        task = local_runner.get_task(task_id)
        if task is not None:
            valid_tasks.append(task)
    st.session_state["material_agent_task_ids"] = [task.id for task in valid_tasks]

    if not valid_tasks:
        st.info("先上传材料费发票并点击“Agent识别材料发票”。")
        return

    options = {f"{task.original_filename} | {task.id} | {task.status}": task.id for task in valid_tasks}
    selected_label = st.selectbox("选择当前材料任务", options=list(options.keys()), key="material_agent_selected_task")
    selected_task_id = options[selected_label]
    task = local_runner.get_task(selected_task_id)
    if task is None:
        st.error("任务不存在。")
        return

    fields = _material_agent_extract_fields(task)
    rows = _normalize_line_items(_to_editor_rows(fields.get("line_items")))
    row_total = _line_items_total(rows)
    amount_value = _safe_float(fields.get("amount"))
    display_rows = [{"row_no": idx + 1, **row} for idx, row in enumerate(rows)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("明细行数", len(rows))
    m2.metric("发票总金额(含税)", _format_amount(amount_value) if amount_value is not None else "-")
    m3.metric("明细合计", _format_amount(row_total) if row_total is not None else "-")
    m4.metric("识别模式", str(fields.get("processing_mode") or fields.get("extraction_source") or "default"))

    quality_hints = _material_agent_quality_hints(fields)
    if quality_hints:
        st.warning("发现质量风险：")
        for hint in quality_hints:
            st.markdown(f"- {hint}")
    else:
        st.success("当前明细质量检查通过。")

    edited_rows = st.data_editor(
        display_rows,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=f"material_agent_editor_{task.id}",
        column_config={
            "row_no": st.column_config.NumberColumn("行号", disabled=True, width="small"),
            "item_name": st.column_config.TextColumn("项目名称(含星号)", required=False),
            "spec": st.column_config.TextColumn("规格型号", required=False),
            "quantity": st.column_config.TextColumn("数量", required=False),
            "unit": st.column_config.TextColumn("单位", required=False),
            "line_total_with_tax": st.column_config.TextColumn("每项含税总价", required=False),
        },
    )
    editor_rows = _normalize_line_items(_to_editor_rows(edited_rows))
    editor_total = _line_items_total(editor_rows)
    st.caption(f"编辑区明细合计：{_format_amount(editor_total) if editor_total is not None else '-'}")

    e1, e2 = st.columns(2)
    if e1.button("应用表格修正并重导出", use_container_width=True, key=f"material_agent_apply_table_{task.id}"):
        new_fields = dict(fields)
        new_fields["line_items"] = editor_rows
        if editor_total is not None:
            new_fields["amount"] = _format_amount(editor_total)
        ok, err = _material_agent_apply_updates(task.id, new_fields)
        if ok:
            st.success("表格修正已保存并重导出。")
            st.rerun()
        else:
            st.error(f"保存失败：{err}")

    if e2.button("执行分列修复（项目/规格）", use_container_width=True, key=f"material_agent_split_fix_{task.id}"):
        fixed_rows, changed = _material_agent_auto_split_rows(editor_rows)
        if changed <= 0:
            st.info("没有检测到可修复行。")
        else:
            new_fields = dict(fields)
            new_fields["line_items"] = fixed_rows
            fixed_total = _line_items_total(fixed_rows)
            if fixed_total is not None:
                new_fields["amount"] = _format_amount(fixed_total)
            ok, err = _material_agent_apply_updates(task.id, new_fields)
            if ok:
                st.success(f"分列修复完成，调整 {changed} 行。")
                st.rerun()
            else:
                st.error(f"修复失败：{err}")

    _render_export_download(task, key_scope="material_agent")

    with st.expander("查看抽取结果(JSON)", expanded=False):
        st.json(task.extracted_data or {})

    chat_map = st.session_state.setdefault("material_agent_chat_map", {})
    if not isinstance(chat_map, dict):
        chat_map = {}
        st.session_state["material_agent_chat_map"] = chat_map

    task_messages = chat_map.setdefault(
        task.id,
        [
            {
                "role": "assistant",
                "content": (
                    "已进入材料费整理模式。你可以直接说：`第3行规格改为M20X1.5`、`删除第5行`、"
                    "`新增一行 项目名称=... 规格=... 数量=... 单位=... 金额=...`、`重新识别`、`分列修复`。"
                ),
            }
        ],
    )

    for message in task_messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_input = st.chat_input("例如：第2行规格改为Y50EX-1208TK2+", key=f"material_agent_chat_input_{task.id}")
    if user_input:
        task_messages.append({"role": "user", "content": user_input})
        handled, reply, updated_task, _ = _material_agent_apply_chat_command(user_input, task, fields)
        if handled:
            task_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        llm_reply = _generate_material_agent_reply_llm(user_input, task, fields, task_messages)
        if not llm_reply:
            llm_reply = "当前未获得有效LLM回复。你可以直接使用结构化指令修改表格，例如：`删除第3行`。"
        task_messages.append({"role": "assistant", "content": llm_reply})
        st.rerun()


def _render_material_flow() -> None:
    _render_material_conversation_agent()


def _render_travel_flow() -> None:
    st.subheader("差旅费流程")
    st.caption("差旅流程提供会话式 Agent 自动整理，也支持手工槽位补录；可导出标准归档压缩包。")
    st.markdown(
        "- 去程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 返程交通：机票发票/高铁报销凭证 + 支付记录 + 机票明细\n"
        "- 酒店：发票 + 支付记录 + 平台订单截图（无住宿费用时需提供情况说明）"
    )

    _render_travel_conversation_agent()
    st.divider()

    with st.expander("手工槽位校对（可选）", expanded=False):
        go_section = _render_travel_transport_section("1) 出差去程交通报销", "travel_go")
        return_section = _render_travel_transport_section("2) 出差返程交通报销", "travel_return")
        hotel_section = _render_travel_hotel_section("travel_hotel")
        _render_travel_summary(go_section, return_section, hotel_section)

    st.divider()
    _render_travel_package_export()


def main() -> None:
    st.set_page_config(page_title="Finance Agent", layout="wide")
    st.title("财务 Agent（本地工具版）")
    st.caption("按费用类型选择流程：材料费 / 差旅费")

    init_runtime()

    flow_mode = st.radio(
        "选择报销流程",
        options=["材料费流程", "差旅费流程"],
        horizontal=True,
    )
    st.divider()

    if flow_mode == "材料费流程":
        _render_material_flow()
    else:
        _render_travel_flow()


if __name__ == "__main__":
    main()

