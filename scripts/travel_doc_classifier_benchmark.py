from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    import fitz  # type: ignore[import-not-found]
except Exception:
    fitz = None


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ROOT = _bootstrap_path()

from app.services import parser as ocr_parser  # noqa: E402


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

FOLDER_LABELS = {
    "go_detail",
    "go_payment",
    "go_ticket",
    "hotel_invoice",
    "hotel_order",
    "hotel_payment",
    "return_detail",
    "return_payment",
    "return_ticket",
}

FOLDER_TO_DOC = {
    "go_ticket": ("transport_ticket", "go"),
    "return_ticket": ("transport_ticket", "return"),
    "go_payment": ("transport_payment", "go"),
    "return_payment": ("transport_payment", "return"),
    "go_detail": ("flight_detail", "go"),
    "return_detail": ("flight_detail", "return"),
    "hotel_invoice": ("hotel_invoice", None),
    "hotel_payment": ("hotel_payment", None),
    "hotel_order": ("hotel_order", None),
}

DOC_TYPES = {
    "transport_ticket",
    "transport_payment",
    "flight_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "unknown",
}

FINAL_LABEL_ORDER = [
    "go_ticket",
    "go_payment",
    "go_detail",
    "return_ticket",
    "return_payment",
    "return_detail",
    "hotel_invoice",
    "hotel_payment",
    "hotel_order",
    "__unknown__",
]


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_root: Path
    output_dir: Path
    base_url: str
    ocr_model: str
    text_model: str
    vl_model: str
    timeout_sec: int
    fallback_timeout_sec: int
    max_pages: int
    render_scale: float
    disable_vl_fallback: bool
    classify_min_text_chars: int
    classify_min_confidence: float
    invoice_refine_min_confidence: float
    transport_refine_min_confidence: float
    hotel_refine_min_confidence: float
    direction_min_confidence: float
    limit: int
    verbose: bool


@dataclass(slots=True)
class Sample:
    sample_id: str
    truth_label: str
    truth_doc_type: str
    truth_direction: str | None
    path: Path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1:
        confidence = confidence / 100.0
    if confidence < 0 or confidence > 1:
        return None
    return confidence


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        parsed = json.loads(source)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", source)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_ocr_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^```[a-zA-Z]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _fallback_doc_type_payload(raw_content: str) -> dict[str, Any] | None:
    text = str(raw_content or "").strip()
    if not text:
        return None
    lowered = text.lower()
    for label in (
        "transport_ticket",
        "transport_payment",
        "flight_detail",
        "hotel_invoice",
        "hotel_payment",
        "hotel_order",
        "unknown",
    ):
        if re.search(rf"\b{re.escape(label)}\b", lowered):
            return {"doc_type": label, "confidence": 0.55, "evidence": "fallback:label_in_response"}

    merged = lowered
    if ("入住" in text and "离店" in text) or ("取消政策" in text) or ("几晚" in text):
        return {"doc_type": "hotel_order", "confidence": 0.6, "evidence": "fallback:hotel_order_markers"}
    if ("支付成功" in text or "交易成功" in text or "付款方式" in text) and ("酒店" in text or "住宿" in text):
        return {"doc_type": "hotel_payment", "confidence": 0.6, "evidence": "fallback:hotel_payment_markers"}
    if ("支付成功" in text or "交易成功" in text or "付款方式" in text) and ("机票" in text or "客运" in text or "航班" in text):
        return {"doc_type": "transport_payment", "confidence": 0.6, "evidence": "fallback:transport_payment_markers"}
    if "机建" in text or "燃油" in text or "价格明细" in text or "退改签" in text:
        return {"doc_type": "flight_detail", "confidence": 0.6, "evidence": "fallback:flight_detail_markers"}
    if ("发票" in text or "电子发票" in text) and ("住宿服务" in text or "房费" in text):
        return {"doc_type": "hotel_invoice", "confidence": 0.6, "evidence": "fallback:hotel_invoice_markers"}
    if ("发票" in text or "电子发票" in text) and ("代订机票费" in text or "客运服务" in text or "机票" in text):
        return {"doc_type": "transport_ticket", "confidence": 0.6, "evidence": "fallback:transport_ticket_markers"}
    if "unknown" in merged or "无法判断" in text or "不确定" in text:
        return {"doc_type": "unknown", "confidence": 0.5, "evidence": "fallback:unknown_markers"}
    return None


def _fallback_direction_payload(raw_content: str) -> dict[str, Any] | None:
    text = str(raw_content or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if re.search(r"\breturn\b", lowered) or any(token in text for token in ["返程", "回程", "返航", "返回", "回到"]):
        return {"direction": "return", "confidence": 0.55, "evidence": "fallback:return_markers"}
    if re.search(r"\bgo\b", lowered) or any(token in text for token in ["去程", "出发", "前往", "单程"]):
        return {"direction": "go", "confidence": 0.55, "evidence": "fallback:go_markers"}
    if "unknown" in lowered or "无法判断" in text or "不确定" in text:
        return {"direction": "unknown", "confidence": 0.5, "evidence": "fallback:unknown_markers"}
    return None


_CITY_STOP_TOKENS = {
    "电子发票",
    "普通发票",
    "机票",
    "客运服务",
    "经纪代理服务",
    "订单",
    "订单详情",
    "支付",
    "价格明细",
    "发票",
    "酒店",
    "住宿服务",
    "房费",
    "统一社会信用代码",
    "纳税人识别号",
    "交易渠道",
    "修改姓名",
    "更改规则",
    "行票",
    "证件",
    "场所",
    "项目名称",
    "价税合计",
    "账单详情",
}


def _normalize_city_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    text = re.sub(r"[\s·•,，。:：;；()（）【】\[\]<>《》]+", "", text)
    text = re.sub(r"(国际机场|机场|火车站|高铁站|客运站|汽车站|南站|北站|东站|西站)$", "", text)
    text = re.sub(r"(T\d+)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(市)$", "", text)
    if len(text) < 2 or len(text) > 6:
        return ""
    if text in _CITY_STOP_TOKENS:
        return ""
    if any(key in text for key in ["发票", "订单", "支付", "金额", "价税", "服务"]):
        return ""
    return text


def _extract_city_pairs(raw_text: str) -> list[tuple[str, str]]:
    text = str(raw_text or "")
    if not text:
        return []
    patterns = [
        r"([一-龥]{2,12})\s*[-—~～→至到]\s*([一-龥]{2,12})",
        r"([一-龥]{2,12})\s+到\s+([一-龥]{2,12})",
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            src = _normalize_city_token(match.group(1))
            dst = _normalize_city_token(match.group(2))
            if not src or not dst or src == dst:
                continue
            pair = (src, dst)
            if pair not in pairs:
                pairs.append(pair)

    iata_to_city = {
        "CGQ": "长春",
        "PVG": "上海",
        "SHA": "上海",
        "PEK": "北京",
        "PKX": "北京",
        "NKG": "南京",
        "CAN": "广州",
        "SZX": "深圳",
        "CTU": "成都",
        "HGH": "杭州",
        "TSN": "天津",
        "XIY": "西安",
        "HRB": "哈尔滨",
    }
    codes = re.findall(r"\b[A-Z]{3}\b", text.upper())
    for idx in range(len(codes) - 1):
        src_code = codes[idx]
        dst_code = codes[idx + 1]
        if src_code not in iata_to_city or dst_code not in iata_to_city:
            continue
        src = iata_to_city[src_code]
        dst = iata_to_city[dst_code]
        if src == dst:
            continue
        pair = (src, dst)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _anonymous_name(file_bytes: bytes, suffix: str) -> str:
    digest = _sha1_bytes(file_bytes)[:12]
    cleaned_suffix = str(suffix or "").strip().lower()
    if cleaned_suffix and not cleaned_suffix.startswith("."):
        cleaned_suffix = f".{cleaned_suffix}"
    return f"doc_{digest}{cleaned_suffix}"


def _collect_samples(dataset_root: Path, limit: int = 0) -> list[Sample]:
    rows: list[Sample] = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        label = folder.name
        if label not in FOLDER_LABELS:
            continue
        truth_doc_type, truth_direction = FOLDER_TO_DOC[label]
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        for idx, path in enumerate(files, start=1):
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            rows.append(
                Sample(
                    sample_id=f"{label}_{idx:04d}",
                    truth_label=label,
                    truth_doc_type=truth_doc_type,
                    truth_direction=truth_direction,
                    path=path,
                )
            )
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows


def _render_pdf_to_images_b64(file_bytes: bytes, *, max_pages: int, render_scale: float) -> list[str]:
    if fitz is None or not file_bytes:
        return []
    pages: list[str] = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            if page_count <= 0:
                return []
            limit = page_count if max_pages <= 0 else min(page_count, max_pages)
            for page_idx in range(limit):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
                png_bytes = pix.tobytes("png")
                pages.append(base64.b64encode(png_bytes).decode("utf-8"))
    except Exception:
        return []
    return pages


def _image_inputs_for_vl(file_bytes: bytes, suffix: str, cfg: BenchmarkConfig) -> list[str]:
    ext = str(suffix or "").lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return [base64.b64encode(file_bytes).decode("utf-8")]
    if ext == ".pdf":
        return _render_pdf_to_images_b64(file_bytes, max_pages=cfg.max_pages, render_scale=cfg.render_scale)
    return []


def _post_ocr_text(images_b64: list[str], model: str, cfg: BenchmarkConfig) -> tuple[str, str]:
    if not images_b64:
        return "", "no_images"
    errors: list[str] = []
    system_prompt = "你是OCR助手，只提取可见文本，按阅读顺序输出，不要解释。"
    user_prompt = "请执行OCR，只输出正文文本。"
    timeout_sec = max(12, cfg.timeout_sec + 6)

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": images_b64},
            ],
            "options": {"temperature": 0.1},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/chat",
            json=payload,
            timeout=(8, timeout_sec),
        )
        resp.raise_for_status()
        content = _clean_ocr_text(str((resp.json().get("message") or {}).get("content") or ""))
        if content:
            return content, ""
        errors.append("chat_empty")
    except Exception as exc:
        errors.append(f"chat_error:{exc}")

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": f"[system]{system_prompt}\n[user]{user_prompt}",
            "images": images_b64,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/generate",
            json=payload,
            timeout=(8, timeout_sec),
        )
        resp.raise_for_status()
        content = _clean_ocr_text(str(resp.json().get("response") or ""))
        if content:
            return content, ""
        errors.append("generate_empty")
    except Exception as exc:
        errors.append(f"generate_error:{exc}")

    return "", ";".join(errors)


def _ocr_with_retry(file_bytes: bytes, suffix: str, cfg: BenchmarkConfig) -> tuple[str, str]:
    parser_text = ""
    try:
        parser_text = _clean_ocr_text(str(ocr_parser.parse_file_bytes(file_bytes, suffix, max_pages=cfg.max_pages) or ""))
    except Exception:
        parser_text = ""
    if parser_text:
        return parser_text, "parser"

    images = _image_inputs_for_vl(file_bytes, suffix, cfg)
    if not images:
        return "", "ocr_no_images"

    final_error = ""
    for attempt in range(1):
        text, err = _post_ocr_text(images, cfg.ocr_model, cfg)
        if text:
            return text, f"vl_retry_{attempt + 1}"
        final_error = err
    return "", f"ocr_empty:{final_error}"


def _post_generate_json(
    prompt: str,
    model: str,
    cfg: BenchmarkConfig,
    *,
    timeout_sec_override: int | None = None,
    fallback_timeout_sec_override: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    errors: list[str] = []
    content = ""
    timeout_sec = int(timeout_sec_override) if timeout_sec_override is not None else cfg.timeout_sec
    fallback_timeout_sec = (
        int(fallback_timeout_sec_override) if fallback_timeout_sec_override is not None else cfg.fallback_timeout_sec
    )
    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/generate",
            json=payload,
            timeout=(8, timeout_sec),
        )
        resp.raise_for_status()
        content = str(resp.json().get("response") or "")
        parsed = _extract_json_from_text(content)
        if parsed:
            return parsed, content, ""
        errors.append("generate_non_json")
    except Exception as exc:
        errors.append(f"generate_error:{exc}")

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/chat",
            json=payload,
            timeout=(8, fallback_timeout_sec),
        )
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "")
        parsed = _extract_json_from_text(content)
        if parsed:
            return parsed, content, ""
        errors.append("chat_non_json")
    except Exception as exc:
        errors.append(f"chat_error:{exc}")

    return None, content, ";".join(errors)


def _post_vl_json(prompt: str, images_b64: list[str], model: str, cfg: BenchmarkConfig) -> tuple[dict[str, Any] | None, str, str]:
    errors: list[str] = []
    content = ""
    if not images_b64:
        return None, "", "no_images"

    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "options": {"temperature": 0},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/chat",
            json=payload,
            timeout=(8, cfg.timeout_sec),
        )
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "")
        parsed = _extract_json_from_text(content)
        if parsed:
            return parsed, content, ""
        errors.append("chat_non_json")
    except Exception as exc:
        errors.append(f"chat_error:{exc}")

    try:
        payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "images": images_b64,
            "options": {"temperature": 0},
        }
        resp = requests.post(
            f"{cfg.base_url}/api/generate",
            json=payload,
            timeout=(8, cfg.fallback_timeout_sec),
        )
        resp.raise_for_status()
        content = str(resp.json().get("response") or "")
        parsed = _extract_json_from_text(content)
        if parsed:
            return parsed, content, ""
        errors.append("generate_non_json")
    except Exception as exc:
        errors.append(f"generate_error:{exc}")

    return None, content, ";".join(errors)


def _doc_type_prompt() -> str:
    return (
        "你是差旅报销材料分类助手。必须仅根据文档正文分类，禁止使用文件名/路径。\n"
        "只允许 doc_type: transport_ticket, transport_payment, flight_detail, hotel_invoice, hotel_payment, hotel_order, unknown。\n"
        "判定要点：\n"
        "- transport_ticket: 交通发票/票据，常见客运服务、代订机票费、发票抬头与价税合计。\n"
        "- transport_payment: 交通支付凭证，常见交易成功、支付时间、付款方式、账单详情。\n"
        "- flight_detail: 机票明细页，常见价格明细、票价、机建、燃油、退改签、航段。\n"
        "- hotel_invoice: 酒店发票，常见住宿服务、房费、酒店开票主体、价税合计。\n"
        "- hotel_payment: 酒店支付凭证，常见酒店订单支付成功记录。\n"
        "- hotel_order: 酒店订单明细，常见订单号、入住/离店、几晚明细、取消政策。\n"
        "强规则：项目名称出现“代订机票费/客运服务”优先 transport_ticket；出现“住宿服务/房费”优先 hotel_invoice。\n"
        "返回单个JSON对象："
        '{"doc_type":"transport_ticket","confidence":0.86,"evidence":"...","ocr_text":"..."}'
    )


def _invoice_refine_prompt() -> str:
    return (
        "你是差旅发票细分类助手。只判断 transport_ticket / hotel_invoice / unknown。\n"
        "只看正文，不允许使用文件名。\n"
        "优先级：项目名称 > 备注行程信息 > 销售方信息。\n"
        "项目名称包含代订机票费/客运服务/机票相关服务 => transport_ticket。\n"
        "项目名称包含住宿服务/房费/酒店住宿 => hotel_invoice。\n"
        "返回单个JSON："
        '{"doc_type":"transport_ticket","confidence":0.90,"evidence":"..."}'
    )


def _transport_refine_prompt() -> str:
    return (
        "你是交通类材料细分类助手。只判断 transport_ticket / transport_payment / flight_detail / unknown。\n"
        "只看正文，不允许使用文件名。\n"
        "规则：\n"
        "- 含发票结构（发票号码/购买方/销售方/价税合计）且项目为客运/代订机票 => transport_ticket。\n"
        "- 含交易成功/支付时间/付款方式/账单详情 => transport_payment。\n"
        "- 含价格明细/机建/燃油/票价/退改签/航段 => flight_detail。\n"
        "返回JSON："
        '{"doc_type":"flight_detail","confidence":0.84,"evidence":"..."}'
    )


def _hotel_refine_prompt() -> str:
    return (
        "你是酒店类材料细分类助手。只判断 hotel_invoice / hotel_payment / hotel_order / unknown。\n"
        "只看正文，不允许使用文件名。\n"
        "规则：\n"
        "- 发票结构 + 住宿服务/房费 => hotel_invoice。\n"
        "- 支付成功账单（支付时间、支付方式、实付金额、交易状态）=> hotel_payment。\n"
        "- 订单详情（入住离店、几晚、订单号、取消政策、房型/间夜）=> hotel_order。\n"
        "- 同时有“订单详情+支付成功”时：若以入住离店/几晚信息为主，优先 hotel_order。\n"
        "返回JSON："
        '{"doc_type":"hotel_order","confidence":0.80,"evidence":"..."}'
    )


def _direction_prompt(doc_type: str) -> str:
    normalized = str(doc_type or "transport_ticket").strip()
    if normalized not in {"transport_ticket", "transport_payment", "flight_detail"}:
        normalized = "transport_ticket"
    return (
        "你是行程方向判别助手。只根据正文判断 go / return / unknown。\n"
        f"当前材料类型: {normalized}\n"
        "只看正文，不允许使用文件名。\n"
        "规则：\n"
        "- 出现“返程/回程/返航/返回”优先 return。\n"
        "- 出现“去程/出发/前往/单程(且无返程词)”优先 go。\n"
        "- 出现航线“A-长春”倾向 return，出现“长春-A”倾向 go。\n"
        "- 只有完全无行程线索时才输出 unknown。\n"
        "返回JSON："
        '{"direction":"go","confidence":0.75,"evidence":"..."}'
    )


def _normalize_doc_type_result(parsed: dict[str, Any]) -> dict[str, Any]:
    doc_type = str(parsed.get("doc_type") or "unknown").strip()
    if doc_type not in DOC_TYPES:
        doc_type = "unknown"
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if len(evidence) > 220:
        evidence = evidence[:217] + "..."
    ocr_text = str(parsed.get("ocr_text") or parsed.get("text") or parsed.get("recognized_text") or "").strip()
    if len(ocr_text) > 1200:
        ocr_text = ocr_text[:1200]
    return {
        "doc_type": doc_type,
        "confidence": confidence,
        "evidence": evidence,
        "ocr_text": ocr_text,
    }


def _normalize_direction_result(parsed: dict[str, Any]) -> dict[str, Any]:
    direction = str(parsed.get("direction") or "unknown").strip().lower()
    if direction not in {"go", "return", "unknown"}:
        direction = "unknown"
    confidence = _normalize_confidence(parsed.get("confidence"))
    evidence = str(parsed.get("evidence") or parsed.get("reason") or "").strip()
    if len(evidence) > 180:
        evidence = evidence[:177] + "..."
    return {
        "direction": direction,
        "confidence": confidence,
        "evidence": evidence,
    }


def _invoice_field_guard(raw_text: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "unknown", ""
    sample = text[:1800]
    merged = sample.lower()

    transport_markers = ["代订机票费", "客运服务", "机票", "航班", "航空服务", "飞猪订单", "携程机票"]
    hotel_markers = ["住宿服务", "房费", "酒店", "入住", "离店", "几晚", "酒店管理"]
    t_hits = [token for token in transport_markers if token in merged]
    h_hits = [token for token in hotel_markers if token in merged]
    if t_hits and not h_hits:
        return "transport_ticket", f"field_guard:transport:{','.join(t_hits[:3])}"
    if h_hits and not t_hits:
        return "hotel_invoice", f"field_guard:hotel:{','.join(h_hits[:3])}"

    line_hint = ""
    for line in sample.splitlines():
        line_text = line.strip()
        if not line_text:
            continue
        if any(token in line_text for token in ["项目名称", "货物或应税劳务", "服务名称", "备注"]):
            line_hint = line_text
            break
    line_lower = line_hint.lower()
    if any(token in line_lower for token in ["代订机票费", "客运服务", "机票"]):
        return "transport_ticket", f"field_line:{line_hint[:40]}"
    if any(token in line_lower for token in ["住宿服务", "房费", "酒店"]):
        return "hotel_invoice", f"field_line:{line_hint[:40]}"
    return "unknown", ""


def _should_use_vl(raw_text: str, text_result: dict[str, Any] | None, cfg: BenchmarkConfig) -> bool:
    compact_len = len(re.sub(r"\s+", "", str(raw_text or "")))
    if compact_len < max(20, cfg.classify_min_text_chars):
        return True
    if not text_result:
        return True
    doc_type = str(text_result.get("doc_type") or "unknown")
    if doc_type == "unknown":
        return True
    return False


def _direction_heuristic(raw_text: str) -> str:
    text = str(raw_text or "").lower()
    if any(token in text for token in ["返程", "回程", "返航", "返回"]):
        return "return"
    if any(token in text for token in ["去程", "前往", "出发"]):
        return "go"
    if "单程" in text and "回程" not in text and "返程" not in text:
        return "go"
    if re.search(r"[一-龥]{2,8}\s*[-—~～/→至到]\s*长春", text):
        return "return"
    if re.search(r"长春\s*[-—~～/→至到]\s*[一-龥]{2,8}", text):
        return "go"
    return "unknown"


def _label_from_doc_direction(doc_type: str, direction: str) -> str:
    doc = str(doc_type or "unknown")
    direct = str(direction or "unknown")
    if doc == "hotel_invoice":
        return "hotel_invoice"
    if doc == "hotel_payment":
        return "hotel_payment"
    if doc == "hotel_order":
        return "hotel_order"
    if doc == "transport_ticket" and direct in {"go", "return"}:
        return f"{direct}_ticket"
    if doc == "transport_payment" and direct in {"go", "return"}:
        return f"{direct}_payment"
    if doc == "flight_detail" and direct in {"go", "return"}:
        return f"{direct}_detail"
    return "__unknown__"


def _infer_direction_from_route_pair(route_pair: tuple[str, str], anchor_city: str) -> str:
    if not route_pair or not anchor_city:
        return "unknown"
    src, dst = route_pair
    if src == anchor_city and dst != anchor_city:
        return "go"
    if dst == anchor_city and src != anchor_city:
        return "return"
    return "unknown"


def _extract_amount_hint(raw_text: str) -> float | None:
    text = str(raw_text or "")
    if not text:
        return None

    priority_patterns = [
        r"(?:总计|价税合计|订单金额|支付金额|支出金额|实付(?:金额)?)\D{0,10}(?:¥|￥)?\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"(?:\(小写\)|合计)\D{0,10}(?:¥|￥)?\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
    ]
    for pattern in priority_patterns:
        values = re.findall(pattern, text)
        numbers: list[float] = []
        for value in values:
            try:
                amount = abs(float(str(value).replace(",", "")))
            except Exception:
                continue
            if 1 <= amount <= 50000:
                numbers.append(amount)
        if numbers:
            return round(max(numbers), 2)

    values = re.findall(r"(?:¥|￥)\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)", text)
    values.extend(re.findall(r"\b([+-]?[0-9]{1,6}\.[0-9]{1,2})\b", text))
    numbers = []
    for value in values:
        try:
            amount = abs(float(str(value).replace(",", "")))
        except Exception:
            continue
        if 1 <= amount <= 50000:
            numbers.append(amount)
    if not numbers:
        return None
    return round(max(numbers), 2)


def _extract_date_candidates(raw_text: str) -> list[str]:
    text = str(raw_text or "")
    if not text:
        return []
    candidates: set[str] = set()

    explicit_year_dates = re.findall(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", text)
    for y, m, d in explicit_year_dates:
        try:
            dt = datetime(int(y), int(m), int(d))
        except Exception:
            continue
        candidates.add(dt.strftime("%Y-%m-%d"))

    year_hint = None
    if explicit_year_dates:
        try:
            year_hint = int(explicit_year_dates[0][0])
        except Exception:
            year_hint = None
    if year_hint is None:
        year_hint = datetime.now().year

    month_day_dates = re.findall(r"(?<!\d)(\d{1,2})[月/\-.](\d{1,2})(?:日)?(?!\d)", text)
    for m, d in month_day_dates:
        try:
            dt = datetime(int(year_hint), int(m), int(d))
        except Exception:
            continue
        candidates.add(dt.strftime("%Y-%m-%d"))

    return sorted(candidates)


def _date_to_ordinal(date_text: str) -> int | None:
    value = str(date_text or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().toordinal()
    except Exception:
        return None


def _earliest_date_ordinal(values: list[str]) -> int | None:
    ordinals = [_date_to_ordinal(item) for item in list(values or [])]
    valid = [item for item in ordinals if item is not None]
    if not valid:
        return None
    return min(valid)


def _assign_direction(
    row: dict[str, Any],
    *,
    direction: str,
    source_tag: str,
    evidence: str,
    confidence: float,
    allow_override: bool = False,
) -> bool:
    if direction not in {"go", "return"}:
        return False
    current = str(row.get("pred_direction") or "unknown")
    current_conf = _normalize_confidence(row.get("direction_confidence"))
    if current in {"go", "return"} and current != direction and not allow_override:
        return False
    if current == direction and current_conf is not None and current_conf >= confidence:
        return False

    row["pred_direction"] = direction
    row["direction_confidence"] = round(confidence, 3)
    source = str(row.get("source") or "")
    if source:
        if source_tag not in source:
            row["source"] = f"{source}+{source_tag}"
    else:
        row["source"] = source_tag
    ev_text = str(row.get("evidence") or "")
    patch = str(evidence or "").strip()
    if patch and patch not in ev_text:
        row["evidence"] = f"{ev_text}; {patch}".strip("; ").strip()

    doc_type = str(row.get("pred_doc_type") or "unknown")
    row["pred_label"] = _label_from_doc_direction(doc_type, direction)
    return True


def _normalize_route_pair(raw_pair: Any) -> tuple[str, str] | None:
    if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
        return None
    src = _normalize_city_token(str(raw_pair[0]))
    dst = _normalize_city_token(str(raw_pair[1]))
    if not src or not dst or src == dst:
        return None
    return src, dst


def _pick_primary_route_pair(route_pairs: list[tuple[str, str]]) -> tuple[str, str] | None:
    pairs: list[tuple[str, str]] = []
    for pair in list(route_pairs or []):
        normalized = _normalize_route_pair(pair)
        if normalized and normalized not in pairs:
            pairs.append(normalized)
    if not pairs:
        return None
    return pairs[0]


def _batch_direction_prompt(entries: list[dict[str, Any]]) -> str:
    return (
        "你是差旅行程方向归并助手。只根据给定字段判断每个 sample_id 属于 go / return / unknown。\n"
        "关键目标：\n"
        "1) 先看 route(出发地-到达地) 分组；互为反向的两组应分别归到 go 和 return。\n"
        "2) 若两组方向不确定，比较日期，较早日期为 go，较晚日期为 return。\n"
        "3) payment 没有路线时，可参考金额/日期关联到已确定方向的票据或明细。\n"
        "4) 信息不足输出 unknown，不要瞎猜。\n"
        "输出单个JSON对象，格式：\n"
        '{"assignments":[{"sample_id":"...","direction":"go","confidence":0.82,"evidence":"route+date"}]}\n'
        "输入条目如下：\n"
        f"{json.dumps(entries, ensure_ascii=False)}"
    )


def _parse_llm_direction_assignments(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items: list[dict[str, Any]] = []
    raw_items = payload.get("assignments")
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                items.append(item)
    elif isinstance(payload.get("results"), list):
        for item in payload.get("results") or []:
            if isinstance(item, dict):
                items.append(item)
    elif isinstance(payload.get("data"), list):
        for item in payload.get("data") or []:
            if isinstance(item, dict):
                items.append(item)
    else:
        # mapping-style fallback: {"sample_id":"go", ...}
        for key, value in payload.items():
            if key in {"assignments", "results", "data"}:
                continue
            direction = str(value or "").strip().lower()
            if direction in {"go", "return", "unknown"}:
                items.append({"sample_id": key, "direction": direction, "confidence": 0.5, "evidence": "mapping_fallback"})
    return items


def _parse_llm_direction_assignments_from_raw(raw_text: str, sample_ids: list[str]) -> list[dict[str, Any]]:
    source = str(raw_text or "")
    if not source:
        return []
    items: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sid = str(sample_id or "").strip()
        if not sid:
            continue
        pattern = rf"{re.escape(sid)}[\s:=\-，,]*([a-zA-Z_]+)"
        match = re.search(pattern, source)
        if not match:
            continue
        direction = str(match.group(1) or "").strip().lower()
        if direction not in {"go", "return", "unknown"}:
            continue
        items.append(
            {
                "sample_id": sid,
                "direction": direction,
                "confidence": 0.5,
                "evidence": "raw_parse",
            }
        )
    return items


def _resolve_anchor_city(rows: list[dict[str, Any]]) -> str:
    depart_count: dict[str, int] = defaultdict(int)
    arrive_count: dict[str, int] = defaultdict(int)
    for row in rows:
        if str(row.get("pred_doc_type") or "") not in {"transport_ticket", "transport_payment", "flight_detail"}:
            continue
        route_pairs = list(row.get("_route_pairs") or [])
        for src, dst in route_pairs:
            depart_count[src] += 1
            arrive_count[dst] += 1

    candidates: list[tuple[int, int, int, str]] = []
    all_cities = set(depart_count.keys()) | set(arrive_count.keys())
    for city in all_cities:
        dep = int(depart_count.get(city, 0))
        arr = int(arrive_count.get(city, 0))
        if dep <= 0 or arr <= 0:
            continue
        total = dep + arr
        balance = abs(dep - arr)
        candidates.append((total, -balance, max(dep, arr), city))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    top_total, _, _, top_city = candidates[0]
    if top_total < 3:
        return ""
    return top_city


def _apply_batch_direction_resolution(rows: list[dict[str, Any]], cfg: BenchmarkConfig) -> None:
    transport_docs = {"transport_ticket", "transport_payment", "flight_detail"}
    transport_rows = [row for row in rows if str(row.get("pred_doc_type") or "") in transport_docs]
    if not transport_rows:
        return

    # 0) enrich row-level fields
    for row in transport_rows:
        raw_text = str(row.get("_raw_text") or "")
        if not isinstance(row.get("_route_pairs"), list):
            row["_route_pairs"] = _extract_city_pairs(raw_text)
        row["_route_pairs"] = [pair for pair in (_normalize_route_pair(p) for p in row.get("_route_pairs") or []) if pair]
        date_candidates = row.get("_date_candidates")
        if not isinstance(date_candidates, list):
            date_candidates = _extract_date_candidates(raw_text)
        row["_date_candidates"] = date_candidates
        row["_amount_hint"] = _extract_amount_hint(raw_text)

    # 1) batch LLM direction assignment
    entries: list[dict[str, Any]] = []
    sample_map: dict[str, dict[str, Any]] = {}
    for row in transport_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        sample_map[sample_id] = row
        routes = [f"{src}-{dst}" for src, dst in list(row.get("_route_pairs") or [])[:3]]
        dates = list(row.get("_date_candidates") or [])[:4]
        entries.append(
            {
                "sample_id": sample_id,
                "doc_type": row.get("pred_doc_type"),
                "routes": routes,
                "dates": dates,
                "amount": row.get("_amount_hint"),
            }
        )

    if entries:
        batch_errors: list[str] = []
        for start in range(0, len(entries), 14):
            chunk = entries[start : start + 14]
            prompt = _batch_direction_prompt(chunk)
            llm_payload, llm_raw, llm_err = _post_generate_json(
                prompt,
                cfg.text_model,
                cfg,
                timeout_sec_override=max(40, cfg.timeout_sec),
                fallback_timeout_sec_override=max(20, cfg.fallback_timeout_sec),
            )
            if llm_payload is None:
                llm_payload = _extract_json_from_text(llm_raw)

            llm_items = _parse_llm_direction_assignments(llm_payload)
            if not llm_items:
                llm_items = _parse_llm_direction_assignments_from_raw(
                    llm_raw,
                    [str(item.get("sample_id") or "") for item in chunk],
                )
            for item in llm_items:
                sample_id = str(item.get("sample_id") or "").strip()
                if not sample_id or sample_id not in sample_map:
                    continue
                direction = str(item.get("direction") or "unknown").strip().lower()
                confidence = _normalize_confidence(item.get("confidence"))
                if confidence is None:
                    confidence = 0.58
                evidence = str(item.get("evidence") or item.get("reason") or "").strip() or "batch_llm"
                if direction in {"go", "return"} and confidence >= 0.45:
                    _assign_direction(
                        sample_map[sample_id],
                        direction=direction,
                        source_tag="direction_batch_llm",
                        evidence=f"direction_llm_batch:{evidence[:80]}",
                        confidence=confidence,
                        allow_override=False,
                    )
            if llm_err:
                batch_errors.append(llm_err)

        if batch_errors:
            err_join = ";".join(batch_errors)
            for row in transport_rows:
                ev = str(row.get("evidence") or "")
                err_text = f"direction_batch_llm_err:{err_join[:120]}"
                if err_text not in ev:
                    row["evidence"] = f"{ev}; {err_text}".strip("; ").strip()

    # 2) route+date fallback (same destination pair + earliest date => go)
    unresolved = [row for row in transport_rows if str(row.get("pred_direction") or "unknown") not in {"go", "return"}]
    route_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unresolved:
        primary_route = _pick_primary_route_pair(list(row.get("_route_pairs") or []))
        if primary_route:
            route_groups[primary_route].append(row)

    undirected: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for pair in route_groups:
        undirected_key = tuple(sorted(pair))
        if pair not in undirected[undirected_key]:
            undirected[undirected_key].append(pair)

    for _, oriented_pairs in undirected.items():
        if len(oriented_pairs) < 2:
            continue
        pair_a, pair_b = oriented_pairs[0], oriented_pairs[1]
        rows_a = route_groups.get(pair_a, [])
        rows_b = route_groups.get(pair_b, [])
        if not rows_a or not rows_b:
            continue
        date_a = _earliest_date_ordinal(
            [date for row in rows_a for date in list(row.get("_date_candidates") or [])]
        )
        date_b = _earliest_date_ordinal(
            [date for row in rows_b for date in list(row.get("_date_candidates") or [])]
        )

        go_pair: tuple[str, str] | None = None
        return_pair: tuple[str, str] | None = None
        if date_a is not None and date_b is not None and date_a != date_b:
            if date_a < date_b:
                go_pair, return_pair = pair_a, pair_b
            else:
                go_pair, return_pair = pair_b, pair_a
        elif pair_a[0] == "长春" and pair_b[1] == "长春":
            go_pair, return_pair = pair_a, pair_b
        elif pair_b[0] == "长春" and pair_a[1] == "长春":
            go_pair, return_pair = pair_b, pair_a

        if go_pair and return_pair:
            for row in route_groups.get(go_pair, []):
                _assign_direction(
                    row,
                    direction="go",
                    source_tag="direction_route_date_rule",
                    evidence=f"route_date_rule:{go_pair[0]}-{go_pair[1]}",
                    confidence=0.72,
                    allow_override=False,
                )
            for row in route_groups.get(return_pair, []):
                _assign_direction(
                    row,
                    direction="return",
                    source_tag="direction_route_date_rule",
                    evidence=f"route_date_rule:{return_pair[0]}-{return_pair[1]}",
                    confidence=0.72,
                    allow_override=False,
                )

    # 3) anchor city fallback for remaining unknown
    unresolved = [row for row in transport_rows if str(row.get("pred_direction") or "unknown") not in {"go", "return"}]
    anchor = _resolve_anchor_city(transport_rows)
    if anchor:
        for row in unresolved:
            primary_route = _pick_primary_route_pair(list(row.get("_route_pairs") or []))
            if not primary_route:
                continue
            guessed = _infer_direction_from_route_pair(primary_route, anchor)
            if guessed not in {"go", "return"}:
                continue
            _assign_direction(
                row,
                direction=guessed,
                source_tag="direction_batch_anchor",
                evidence=f"batch_anchor:{anchor}->{primary_route[0]}-{primary_route[1]}",
                confidence=0.68,
                allow_override=False,
            )

    # 4) payment amount-link fallback
    refs: list[tuple[float, str]] = []
    for row in transport_rows:
        direction = str(row.get("pred_direction") or "unknown")
        if direction not in {"go", "return"}:
            continue
        amount = row.get("_amount_hint")
        try:
            amount_val = float(amount)
        except Exception:
            continue
        refs.append((amount_val, direction))

    if not refs:
        return

    for row in transport_rows:
        if str(row.get("pred_doc_type") or "") != "transport_payment":
            continue
        if str(row.get("pred_direction") or "unknown") in {"go", "return"}:
            continue
        try:
            amount_val = float(row.get("_amount_hint"))
        except Exception:
            continue
        scored = sorted((abs(amount_val - ref_amt), ref_dir) for ref_amt, ref_dir in refs)
        if not scored:
            continue
        best_diff = scored[0][0]
        if best_diff > 3.0:
            continue
        tied_dirs = {ref_dir for diff, ref_dir in scored if abs(diff - best_diff) < 0.01}
        if len(tied_dirs) != 1:
            continue
        guessed = next(iter(tied_dirs))
        _assign_direction(
            row,
            direction=guessed,
            source_tag="direction_amount_link",
            evidence=f"amount_link:{amount_val:.2f},diff={best_diff:.2f}",
            confidence=0.7,
            allow_override=False,
        )


def _classification_metrics(
    rows: list[dict[str, Any]],
    *,
    true_key: str,
    pred_key: str,
    labels: list[str],
) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {label: {pred: 0 for pred in labels} for label in labels}
    total = 0
    correct = 0

    for row in rows:
        true_value = str(row.get(true_key) or "")
        pred_value = str(row.get(pred_key) or "")
        if true_value not in labels:
            continue
        if pred_value not in labels:
            pred_value = labels[-1]
        confusion[true_value][pred_value] += 1
        total += 1
        if true_value == pred_value:
            correct += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    macro_f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            macro_f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    return {
        "scored": total,
        "correct": correct,
        "accuracy": round((correct / total), 4) if total > 0 else None,
        "macro_f1": round(sum(macro_f1_values) / len(macro_f1_values), 4) if macro_f1_values else None,
        "labels": labels,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _write_confusion_csv(confusion: dict[str, dict[str, int]], labels: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred", *labels])
        for true_label in labels:
            row = [true_label]
            for pred_label in labels:
                row.append(int(confusion.get(true_label, {}).get(pred_label, 0)))
            writer.writerow(row)


def _write_predictions_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fields = [
        "sample_id",
        "file_path",
        "truth_label",
        "truth_doc_type",
        "truth_direction",
        "pred_label",
        "pred_doc_type",
        "pred_direction",
        "doc_type_confidence",
        "direction_confidence",
        "source",
        "evidence",
        "ocr_chars",
        "timing_ocr_sec",
        "timing_doc_type_sec",
        "timing_direction_sec",
        "timing_total_sec",
        "error",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _env_snapshot(cfg: BenchmarkConfig) -> dict[str, Any]:
    return {
        "dataset_root": str(cfg.dataset_root),
        "base_url": cfg.base_url,
        "ocr_model": cfg.ocr_model,
        "text_model": cfg.text_model,
        "vl_model": cfg.vl_model,
        "disable_vl_fallback": cfg.disable_vl_fallback,
        "classify_min_text_chars": cfg.classify_min_text_chars,
        "classify_min_confidence": cfg.classify_min_confidence,
        "invoice_refine_min_confidence": cfg.invoice_refine_min_confidence,
        "transport_refine_min_confidence": cfg.transport_refine_min_confidence,
        "hotel_refine_min_confidence": cfg.hotel_refine_min_confidence,
        "direction_min_confidence": cfg.direction_min_confidence,
        "max_pages": cfg.max_pages,
        "render_scale": cfg.render_scale,
    }


def _run_sample(sample: Sample, cfg: BenchmarkConfig) -> dict[str, Any]:
    started_total = time.perf_counter()
    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "file_path": str(sample.path.relative_to(ROOT).as_posix()),
        "truth_label": sample.truth_label,
        "truth_doc_type": sample.truth_doc_type,
        "truth_direction": sample.truth_direction or "",
        "pred_label": "__unknown__",
        "pred_doc_type": "unknown",
        "pred_direction": "unknown",
        "doc_type_confidence": None,
        "direction_confidence": None,
        "source": "",
        "evidence": "",
        "ocr_chars": 0,
        "timing_ocr_sec": None,
        "timing_doc_type_sec": None,
        "timing_direction_sec": None,
        "timing_total_sec": None,
        "error": "",
        "_raw_text": "",
        "_route_pairs": [],
        "_date_candidates": [],
    }

    try:
        file_bytes = sample.path.read_bytes()
    except Exception as exc:
        result["error"] = f"read_error:{exc}"
        result["timing_total_sec"] = round(time.perf_counter() - started_total, 3)
        return result

    suffix = sample.path.suffix.lower()
    _ = _anonymous_name(file_bytes, suffix)

    ocr_started = time.perf_counter()
    raw_text, ocr_source = _ocr_with_retry(file_bytes, suffix, cfg)
    result["timing_ocr_sec"] = round(time.perf_counter() - ocr_started, 3)
    result["_raw_text"] = raw_text
    result["ocr_chars"] = len(raw_text)
    result["_route_pairs"] = _extract_city_pairs(raw_text)
    result["_date_candidates"] = _extract_date_candidates(raw_text)
    if not raw_text and ocr_source.startswith("ocr_empty:"):
        result["error"] = ocr_source

    evidence_parts: list[str] = []
    source_parts: list[str] = []
    if ocr_source and ocr_source != "parser":
        source_parts.append(ocr_source)

    doc_type_started = time.perf_counter()
    text_prompt = f"{_doc_type_prompt()}\nDocument text:\n{raw_text[:12000]}"
    text_payload, text_raw, text_err = _post_generate_json(text_prompt, cfg.text_model, cfg)
    if text_payload is None:
        text_payload = _fallback_doc_type_payload(text_raw)
    text_result = _normalize_doc_type_result(text_payload) if text_payload else None
    if text_result:
        source_parts.append("llm_text")
    elif text_err:
        evidence_parts.append(f"text_cls_error:{text_err[:120]}")

    final_doc_result = text_result
    if not cfg.disable_vl_fallback and _should_use_vl(raw_text, text_result, cfg):
        images = _image_inputs_for_vl(file_bytes, suffix, cfg)
        vl_prompt = _doc_type_prompt()
        if raw_text:
            vl_prompt += f"\nOCR text for reference:\n{raw_text[:7000]}"
        vl_payload, vl_raw, vl_err = _post_vl_json(vl_prompt, images, cfg.vl_model, cfg)
        if vl_payload is None:
            vl_payload = _fallback_doc_type_payload(vl_raw)
        vl_result = _normalize_doc_type_result(vl_payload) if vl_payload else None
        vl_doc_type = str((vl_result or {}).get("doc_type") or "unknown")
        if vl_result:
            text_doc_type = str((final_doc_result or {}).get("doc_type") or "unknown")
            text_conf = _normalize_confidence((final_doc_result or {}).get("confidence"))
            vl_conf = _normalize_confidence((vl_result or {}).get("confidence"))
            if final_doc_result is None or text_doc_type == "unknown":
                if vl_doc_type != "unknown" or final_doc_result is None:
                    final_doc_result = vl_result
                    source_parts.append("llm_vl_fallback")
            elif vl_doc_type == text_doc_type:
                if (vl_conf or 0.0) > (text_conf or 0.0):
                    final_doc_result = vl_result
                    source_parts.append("llm_vl_consensus")
            elif vl_doc_type != "unknown" and (vl_conf or 0.0) >= 0.88 and (text_conf is None or text_conf < 0.5):
                final_doc_result = vl_result
                source_parts.append("llm_vl_override_high")
        elif vl_err:
            evidence_parts.append(f"vl_cls_error:{vl_err[:120]}")
    result["timing_doc_type_sec"] = round(time.perf_counter() - doc_type_started, 3)

    if final_doc_result is None:
        final_doc_result = {"doc_type": "unknown", "confidence": None, "evidence": "", "ocr_text": ""}

    guessed_doc_type = str(final_doc_result.get("doc_type") or "unknown")
    guessed_confidence = _normalize_confidence(final_doc_result.get("confidence"))
    guessed_evidence = str(final_doc_result.get("evidence") or "").strip()
    if guessed_evidence:
        evidence_parts.append(guessed_evidence)

    # Invoice disambiguation
    invoice_like = guessed_doc_type in {"transport_ticket", "hotel_invoice", "unknown"} or (
        "发票" in raw_text or "电子发票" in raw_text
    )
    if invoice_like:
        invoice_prompt = f"{_invoice_refine_prompt()}\nDocument text:\n{raw_text[:9000]}"
        invoice_payload, invoice_raw, invoice_err = _post_generate_json(invoice_prompt, cfg.text_model, cfg)
        if invoice_payload is None:
            invoice_payload = _fallback_doc_type_payload(invoice_raw)
        invoice_result = _normalize_doc_type_result(invoice_payload) if invoice_payload else None
        invoice_doc_type = str((invoice_result or {}).get("doc_type") or "unknown")
        invoice_confidence = _normalize_confidence((invoice_result or {}).get("confidence"))
        if invoice_doc_type in {"transport_ticket", "hotel_invoice"} and (
            guessed_doc_type == "unknown"
            or (invoice_confidence is not None and invoice_confidence >= _clamp(cfg.invoice_refine_min_confidence, 0.0, 1.0))
        ):
            if guessed_doc_type != invoice_doc_type:
                guessed_doc_type = invoice_doc_type
                source_parts.append("invoice_refine")
            ev = str((invoice_result or {}).get("evidence") or "").strip()
            if ev:
                evidence_parts.append(f"invoice_refine:{ev}")
        elif invoice_err:
            evidence_parts.append(f"invoice_refine_err:{invoice_err[:120]}")

        guard_doc_type, guard_evidence = _invoice_field_guard(raw_text)
        if guard_doc_type in {"transport_ticket", "hotel_invoice"} and guessed_doc_type != guard_doc_type:
            guessed_doc_type = guard_doc_type
            source_parts.append("invoice_field_guard")
        if guard_evidence:
            evidence_parts.append(guard_evidence)

    # Transport refinement
    if guessed_doc_type in {"transport_ticket", "transport_payment", "flight_detail", "unknown"} and (
        any(token in raw_text for token in ["机票", "航班", "机建", "燃油", "客运", "代订机票", "交易成功"])
    ):
        transport_prompt = f"{_transport_refine_prompt()}\nDocument text:\n{raw_text[:9000]}"
        transport_payload, transport_raw, transport_err = _post_generate_json(transport_prompt, cfg.text_model, cfg)
        if transport_payload is None:
            transport_payload = _fallback_doc_type_payload(transport_raw)
        transport_result = _normalize_doc_type_result(transport_payload) if transport_payload else None
        transport_doc_type = str((transport_result or {}).get("doc_type") or "unknown")
        transport_confidence = _normalize_confidence((transport_result or {}).get("confidence"))
        if transport_doc_type in {"transport_ticket", "transport_payment", "flight_detail"} and (
            guessed_doc_type == "unknown"
            or (
                transport_confidence is not None
                and transport_confidence >= _clamp(cfg.transport_refine_min_confidence, 0.0, 1.0)
            )
        ):
            if guessed_doc_type != transport_doc_type:
                guessed_doc_type = transport_doc_type
                source_parts.append("transport_refine")
            ev = str((transport_result or {}).get("evidence") or "").strip()
            if ev:
                evidence_parts.append(f"transport_refine:{ev}")
        elif transport_err:
            evidence_parts.append(f"transport_refine_err:{transport_err[:120]}")

    # Hotel refinement
    if guessed_doc_type in {"hotel_invoice", "hotel_payment", "hotel_order", "unknown"} and (
        any(token in raw_text for token in ["酒店", "住宿", "入住", "离店", "订单号", "房费"])
    ):
        hotel_prompt = f"{_hotel_refine_prompt()}\nDocument text:\n{raw_text[:9000]}"
        hotel_payload, hotel_raw, hotel_err = _post_generate_json(hotel_prompt, cfg.text_model, cfg)
        if hotel_payload is None:
            hotel_payload = _fallback_doc_type_payload(hotel_raw)
        hotel_result = _normalize_doc_type_result(hotel_payload) if hotel_payload else None
        hotel_doc_type = str((hotel_result or {}).get("doc_type") or "unknown")
        hotel_confidence = _normalize_confidence((hotel_result or {}).get("confidence"))
        if hotel_doc_type in {"hotel_invoice", "hotel_payment", "hotel_order"} and (
            guessed_doc_type == "unknown"
            or (hotel_confidence is not None and hotel_confidence >= _clamp(cfg.hotel_refine_min_confidence, 0.0, 1.0))
        ):
            if guessed_doc_type != hotel_doc_type:
                guessed_doc_type = hotel_doc_type
                source_parts.append("hotel_refine")
            ev = str((hotel_result or {}).get("evidence") or "").strip()
            if ev:
                evidence_parts.append(f"hotel_refine:{ev}")
        elif hotel_err:
            evidence_parts.append(f"hotel_refine_err:{hotel_err[:120]}")

    hotel_order_tokens = ["入住", "离店", "几晚", "间夜", "取消政策", "房型", "最晚到店", "订单详情"]
    hotel_order_score = sum(1 for token in hotel_order_tokens if token in raw_text)
    if hotel_order_score >= 2 and guessed_doc_type in {"hotel_invoice", "hotel_payment", "unknown"} and "酒店" in raw_text:
        guessed_doc_type = "hotel_order"
        source_parts.append("hotel_order_guard")
        evidence_parts.append(f"hotel_order_guard:score={hotel_order_score}")

    transport_payment_tokens = ["交易成功", "支付成功", "付款方式", "支付时间", "实付", "账单详情"]
    transport_payment_score = sum(1 for token in transport_payment_tokens if token in raw_text)
    invoice_structure_hit = any(token in raw_text for token in ["发票号码", "购买方", "销售方", "价税合计"])
    if (
        transport_payment_score >= 2
        and not invoice_structure_hit
        and any(token in raw_text for token in ["机票", "航班", "客运", "交通"])
        and guessed_doc_type in {"transport_ticket", "flight_detail", "unknown"}
    ):
        guessed_doc_type = "transport_payment"
        source_parts.append("transport_payment_guard")
        evidence_parts.append(f"transport_payment_guard:score={transport_payment_score}")

    # Direction is deferred to batch stage after all documents are recognized.
    direction = "unknown"
    direction_confidence = None
    result["timing_direction_sec"] = 0.0
    if guessed_doc_type in {"transport_ticket", "transport_payment", "flight_detail"}:
        source_parts.append("direction_deferred")

    result["pred_doc_type"] = guessed_doc_type if guessed_doc_type in DOC_TYPES else "unknown"
    result["doc_type_confidence"] = guessed_confidence
    result["pred_direction"] = direction
    result["direction_confidence"] = direction_confidence
    result["pred_label"] = _label_from_doc_direction(result["pred_doc_type"], direction)
    result["source"] = "+".join(source_parts) if source_parts else "none"
    evidence_text = "; ".join(part for part in evidence_parts if str(part).strip())
    if len(evidence_text) > 400:
        evidence_text = evidence_text[:397] + "..."
    result["evidence"] = evidence_text
    result["timing_total_sec"] = round(time.perf_counter() - started_total, 3)
    return result


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    _assert(cfg.dataset_root.exists(), f"dataset root not found: {cfg.dataset_root}")
    _assert(cfg.dataset_root.is_dir(), f"dataset root is not directory: {cfg.dataset_root}")

    samples = _collect_samples(cfg.dataset_root, limit=cfg.limit)
    _assert(samples, f"no samples found in dataset root: {cfg.dataset_root}")

    results: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for idx, sample in enumerate(samples, start=1):
        row = _run_sample(sample, cfg)
        results.append(row)
        if cfg.verbose:
            print(
                f"[{idx}/{len(samples)}] {sample.truth_label} -> {row.get('pred_label')} "
                f"(doc={row.get('pred_doc_type')}, src={row.get('source')}, t={row.get('timing_total_sec')}s)"
            )

    elapsed_all = time.perf_counter() - start_all
    _apply_batch_direction_resolution(results, cfg)

    final_metrics = _classification_metrics(
        results,
        true_key="truth_label",
        pred_key="pred_label",
        labels=FINAL_LABEL_ORDER,
    )

    doc_type_labels = [
        "transport_ticket",
        "transport_payment",
        "flight_detail",
        "hotel_invoice",
        "hotel_payment",
        "hotel_order",
        "unknown",
    ]
    doc_type_metrics = _classification_metrics(
        results,
        true_key="truth_doc_type",
        pred_key="pred_doc_type",
        labels=doc_type_labels,
    )

    transport_rows = [row for row in results if str(row.get("truth_direction") or "").strip() in {"go", "return"}]
    direction_labels = ["go", "return", "unknown"]
    direction_metrics = _classification_metrics(
        transport_rows,
        true_key="truth_direction",
        pred_key="pred_direction",
        labels=direction_labels,
    )

    badcases = [row for row in results if str(row.get("truth_label")) != str(row.get("pred_label"))]
    badcases = sorted(badcases, key=lambda item: (item.get("truth_label", ""), item.get("pred_label", "")))

    timing_values = [float(row.get("timing_total_sec") or 0.0) for row in results]
    avg_timing = (sum(timing_values) / len(timing_values)) if timing_values else 0.0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (cfg.output_dir / f"travel_doc_cls_{stamp}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = report_dir / "predictions.csv"
    final_confusion_csv = report_dir / "confusion_final_label.csv"
    doc_type_confusion_csv = report_dir / "confusion_doc_type.csv"
    direction_confusion_csv = report_dir / "confusion_direction.csv"
    badcase_json = report_dir / "badcases.json"
    report_json = report_dir / "report.json"

    _write_predictions_csv(results, predictions_csv)
    _write_confusion_csv(final_metrics["confusion_matrix"], FINAL_LABEL_ORDER, final_confusion_csv)
    _write_confusion_csv(doc_type_metrics["confusion_matrix"], doc_type_labels, doc_type_confusion_csv)
    _write_confusion_csv(direction_metrics["confusion_matrix"], direction_labels, direction_confusion_csv)
    badcase_json.write_text(json.dumps(badcases, ensure_ascii=False, indent=2), encoding="utf-8")

    support_count: dict[str, int] = defaultdict(int)
    for row in results:
        support_count[str(row.get("truth_label") or "")] += 1

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed_all, 3),
        "avg_latency_sec": round(avg_timing, 3),
        "sample_count": len(samples),
        "dataset_support": dict(sorted(support_count.items())),
        "environment": _env_snapshot(cfg),
        "metrics": {
            "final_label": final_metrics,
            "doc_type": doc_type_metrics,
            "direction": direction_metrics,
        },
        "files": {
            "predictions_csv": str(predictions_csv),
            "final_confusion_csv": str(final_confusion_csv),
            "doc_type_confusion_csv": str(doc_type_confusion_csv),
            "direction_confusion_csv": str(direction_confusion_csv),
            "badcases_json": str(badcase_json),
        },
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_json)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Travel document classification benchmark on folder-labeled dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "test dataset"),
        help="Dataset root directory. Direct child folder names are truth labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "benchmark" / "travel_doc_classifier"),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--ocr-model",
        default=os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"),
        help="OCR model (VL).",
    )
    parser.add_argument(
        "--text-model",
        default=os.getenv("OLLAMA_TRAVEL_DOC_TEXT_MODEL") or os.getenv("OLLAMA_TEXT_MODEL") or "qwen2.5:7b-instruct",
        help="Text classification model.",
    )
    parser.add_argument(
        "--vl-model",
        default=os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"),
        help="VL classification fallback model.",
    )
    parser.add_argument("--timeout-sec", type=int, default=16, help="Primary request timeout seconds.")
    parser.add_argument("--fallback-timeout-sec", type=int, default=8, help="Fallback request timeout seconds.")
    parser.add_argument("--max-pages", type=int, default=2, help="OCR/VL max pages for PDF.")
    parser.add_argument("--render-scale", type=float, default=2.0, help="PDF render scale for VL fallback.")
    parser.add_argument("--disable-vl-fallback", action="store_true", help="Disable VL fallback classification.")
    parser.add_argument("--classify-min-text-chars", type=int, default=50, help="Fallback threshold for short OCR text.")
    parser.add_argument("--classify-min-confidence", type=float, default=0.55, help="Fallback threshold for text classifier confidence.")
    parser.add_argument("--invoice-refine-min-confidence", type=float, default=0.62)
    parser.add_argument("--transport-refine-min-confidence", type=float, default=0.58)
    parser.add_argument("--hotel-refine-min-confidence", type=float, default=0.58)
    parser.add_argument("--direction-min-confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N samples.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file progress.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        dataset_root=Path(args.dataset_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        base_url=str(args.base_url or "").rstrip("/"),
        ocr_model=str(args.ocr_model or "").strip(),
        text_model=str(args.text_model or "").strip(),
        vl_model=str(args.vl_model or "").strip(),
        timeout_sec=max(8, int(args.timeout_sec)),
        fallback_timeout_sec=max(6, int(args.fallback_timeout_sec)),
        max_pages=max(1, int(args.max_pages)),
        render_scale=max(1.0, float(args.render_scale)),
        disable_vl_fallback=bool(args.disable_vl_fallback),
        classify_min_text_chars=max(20, int(args.classify_min_text_chars)),
        classify_min_confidence=_clamp(float(args.classify_min_confidence), 0.0, 1.0),
        invoice_refine_min_confidence=_clamp(float(args.invoice_refine_min_confidence), 0.0, 1.0),
        transport_refine_min_confidence=_clamp(float(args.transport_refine_min_confidence), 0.0, 1.0),
        hotel_refine_min_confidence=_clamp(float(args.hotel_refine_min_confidence), 0.0, 1.0),
        direction_min_confidence=_clamp(float(args.direction_min_confidence), 0.0, 1.0),
        limit=max(0, int(args.limit)),
        verbose=bool(args.verbose),
    )

    os.environ["OLLAMA_BASE_URL"] = cfg.base_url
    os.environ["OLLAMA_VL_MODEL"] = cfg.ocr_model
    os.environ["OLLAMA_OCR_TIMEOUT"] = str(max(16, cfg.timeout_sec))
    os.environ["OLLAMA_OCR_FALLBACK_TIMEOUT"] = str(max(8, cfg.fallback_timeout_sec))
    os.environ["OLLAMA_OCR_MAX_PAGES"] = str(cfg.max_pages)
    os.environ["OLLAMA_OCR_RENDER_SCALE"] = str(cfg.render_scale)

    report = run_benchmark(cfg)
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "sample_count": report["sample_count"],
                "elapsed_sec": report["elapsed_sec"],
                "avg_latency_sec": report["avg_latency_sec"],
                "final_label_accuracy": report["metrics"]["final_label"]["accuracy"],
                "final_label_macro_f1": report["metrics"]["final_label"]["macro_f1"],
                "doc_type_accuracy": report["metrics"]["doc_type"]["accuracy"],
                "direction_accuracy": report["metrics"]["direction"]["accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("[OK] travel doc classifier benchmark finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
