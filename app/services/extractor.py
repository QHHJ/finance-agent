from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import fitz
import requests

UNIT_TOKENS = ["个", "件", "套", "台", "支", "张", "盒", "瓶", "米", "公斤", "kg", "m"]
COMPANY_HINTS = ["公司", "大学", "学院", "中心", "研究院", "研究所", "事务所", "经销处", "门市", "酒店"]
LINE_ITEM_BASE_FIELDS = ["item_name", "spec", "quantity", "unit", "line_total_with_tax", "amount_no_tax", "tax_amount"]
SPEC_HINT_WORDS = {
    "组合式",
    "附图",
    "不锈钢",
    "屏蔽",
    "金属件",
    "中号",
    "纯铜",
    "开口",
    "绝缘子",
    "铜螺柱",
    "垂直夹具",
    "压板",
    "法兰",
    "FPGA",
    "JYT",
}


def _first_match(patterns: Iterable[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        value = (match.group(1) or "").strip()
        if value:
            return value
    return None


def _pick(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in ("", None):
            return data[key]
    return default


def _env_flag_true(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(f"\ufeff{name}")
    if value is None:
        for k, v in os.environ.items():
            if k.lstrip("\ufeff") == name:
                value = v
                break
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def _normalize_amount(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).replace(",", "").replace("¥", "").replace("￥", "").strip()
    cleaned = re.sub(r"[^\d.]", "", cleaned)
    if not cleaned:
        return None
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).replace(",", "").replace("¥", "").replace("￥", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _format_amount(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _normalize_item_name(name: str) -> str:
    normalized = _normalize_spaces(name)
    # Remove accidental spaces split in Chinese words caused by PDF extraction line wraps.
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    return normalized


def _normalize_spec_text(spec: str) -> str:
    text = _normalize_spaces(spec)
    text = text.strip("，,;；:：()（）[]【】")
    return text


def _is_spec_like(text: str) -> bool:
    value = _normalize_spec_text(text)
    if not value:
        return False
    if len(value) > 40:
        return False
    patterns = [
        r"(?:量程|规格|型号)",
        r"\bM\d+(?:[Xx\*]\d+(?:\.\d+)?)?\b",
        r"\d+(?:\.\d+)?\s*(?:-|~|～|x|X|\*)\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|um|μm|nm|kg|g|V|A|W|Hz|℃|°C)?",
        r"\d+(?:\.\d+)?\s*(?:mm|cm|m|um|μm|nm|kg|g|V|A|W|Hz|℃|°C)\b",
        r"[A-Za-z]{1,3}\d+(?:\.\d+)?(?:[-_/][A-Za-z0-9.]+)+",
        r"\d+(?:\.\d+)?\s*(?:组|kv|kV|KV|平方|平米|mm|cm|m|kg|g)\b",
        r"[A-Za-z]{2,}\d*[A-Za-z0-9\-_.]*$",
    ]
    return any(re.search(pattern, value) for pattern in patterns)


def _strip_suffix_ignore_spaces(text: str, suffix: str) -> str:
    source = text
    target = suffix.replace(" ", "")
    if not target:
        return source
    for idx in range(len(source) + 1):
        part = source[idx:]
        if part.replace(" ", "") == target:
            return source[:idx]
    return source


def _split_item_name_and_spec(item_name: str, spec: str) -> tuple[str, str]:
    name = _normalize_item_name(item_name)
    spec_text = _normalize_spec_text(spec)

    # 1) When spec already exists, remove duplicated suffix from item name.
    if spec_text and name:
        candidate_name = name
        stripped = False
        if name.endswith(spec_text):
            candidate_name = name[: -len(spec_text)]
            stripped = True
        elif name.replace(" ", "").endswith(spec_text.replace(" ", "")):
            candidate_name = _strip_suffix_ignore_spaces(name, spec_text)
            stripped = True
        candidate_name = candidate_name.rstrip(" -_/，,;；:：")

        # Keep at least one concrete item token after category prefix.
        if not re.fullmatch(r"\*[\u4e00-\u9fa5]{2,10}\*", candidate_name):
            name = candidate_name
        elif stripped:
            # If stripping would empty the item name to category-only, keep item name
            # and drop duplicated spec to avoid repeated text in both columns.
            spec_text = ""

    # 2) If spec is still empty, try extracting spec-like tail from item name.
    if not spec_text and name:
        tail_patterns = [
            r"^(?P<name>.*?)(?P<spec>量程\s*\d+(?:\.\d+)?\s*(?:-|~|～)\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|um|μm|nm)?)$",
            r"^(?P<name>.*?)[\s_-]+(?P<spec>M\d+(?:[Xx\*]\d+(?:\.\d+)?)?)$",
            r"^(?P<name>.*?)[\s_-]+(?P<spec>\d+(?:\.\d+)?\s*(?:-|~|～|x|X|\*)\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|um|μm|nm|kg|g|V|A|W|Hz|℃|°C)?)$",
            r"^(?P<name>.*?)[\s_-]+(?P<spec>[A-Za-z]{1,3}\d+(?:\.\d+)?(?:[-_/][A-Za-z0-9.]+)+)$",
        ]
        for pattern in tail_patterns:
            match = re.match(pattern, name)
            if not match:
                continue
            candidate_name = _normalize_item_name(match.group("name")).rstrip(" -_/，,;；:：")
            candidate_spec = _normalize_spec_text(match.group("spec"))
            if candidate_name and _is_spec_like(candidate_spec):
                name = candidate_name
                spec_text = candidate_spec
                break

    return _normalize_item_name(name), spec_text


def _refine_name_spec_boundary(name: str, spec: str) -> tuple[str, str]:
    clean_name = _normalize_item_name(name).rstrip(" -_/，,;；:：")
    clean_spec = _normalize_spec_text(spec)
    if not clean_name or not clean_spec:
        return clean_name, clean_spec

    # Avoid splitting when the left part is only a category prefix like "*金属制品*".
    if clean_name.endswith("*") and re.fullmatch(r"[\u4e00-\u9fa5]{1,10}", clean_spec):
        return _normalize_item_name(f"{clean_name}{clean_spec}"), ""

    # If spec starts with Chinese words then model text (e.g. "字圆头螺丝M4*10"),
    # move the leading Chinese words back to item_name.
    anchor = re.search(
        r"(?:量程\s*\d|M\d|[A-Za-z]{1,4}\d|φ?\d+(?:\.\d+)?\s*(?:-|~|～|x|X|\*)\s*\d+|\d+\s*(?:mm|cm|m|kg|g|kV|KV|kv|℃|°C))",
        clean_spec,
    )
    if anchor and anchor.start() > 0:
        leading = clean_spec[: anchor.start()].strip(" -_/，,;；:：")
        trailing = clean_spec[anchor.start() :].strip(" -_/，,;；:：")
        if leading and trailing and re.search(r"[\u4e00-\u9fa5]", leading) and not re.search(r"\d", leading):
            overlap = 0
            max_overlap = min(len(clean_name), len(leading))
            for size in range(max_overlap, 0, -1):
                if clean_name.endswith(leading[:size]):
                    overlap = size
                    break
            merged = clean_name + leading[overlap:]
            clean_name = _normalize_item_name(merged)
            clean_spec = _normalize_spec_text(trailing)

    return clean_name, clean_spec


def _split_name_tail_to_spec(text: str) -> tuple[str, str]:
    value = _normalize_spaces(text).strip(" -_/，,;；:：")
    if not value:
        return "", ""

    tokens = value.split()
    if len(tokens) >= 2:
        candidates = [" ".join(tokens[-2:]), tokens[-1]]
        for candidate in candidates:
            candidate = _normalize_spec_text(candidate)
            if not candidate:
                continue
            if _is_spec_like(candidate) or candidate in SPEC_HINT_WORDS or any(word in candidate for word in SPEC_HINT_WORDS):
                name = value[: -len(candidate)].rstrip(" -_/，,;；:：")
                if name:
                    return _refine_name_spec_boundary(name, candidate)

    for word in sorted(SPEC_HINT_WORDS, key=len, reverse=True):
        if value.endswith(word) and len(value) > len(word) + 1:
            name = value[: -len(word)].rstrip(" -_/，,;；:：")
            if name:
                return _refine_name_spec_boundary(name, word)

    digit_split = re.match(r"^(?P<name>.*?)(?P<spec>(?:M\d.*|\d.*|[A-Za-z]{2,}\d*[A-Za-z0-9\-_.]*))$", value)
    if digit_split:
        name = _normalize_spaces(digit_split.group("name")).rstrip(" -_/，,;；:：")
        spec = _normalize_spec_text(digit_split.group("spec"))
        if name and spec and (_is_spec_like(spec) or spec in SPEC_HINT_WORDS):
            return _refine_name_spec_boundary(name, spec)

    return value, ""


def _split_long_prefix_item_spec(prefix: str) -> tuple[str, str]:
    text = _normalize_spaces(prefix).strip()
    if "*" in text:
        text = text[text.find("*") :]
    text = text.strip(" -_/，,;；:：")
    if not text:
        return "", ""

    if text.count("*") >= 2:
        first = text.find("*")
        second = text.find("*", first + 1)
        head = text[: second + 1]
        tail = text[second + 1 :].strip()
        if not tail:
            return _normalize_item_name(text), ""
        tail_name, tail_spec = _split_name_tail_to_spec(tail)
        item_name = f"{head}{tail_name}".strip()
        return _normalize_item_name(item_name), _normalize_spec_text(tail_spec)

    item_name, spec = _split_name_tail_to_spec(text)
    return _normalize_item_name(item_name), _normalize_spec_text(spec)


def _normalize_quantity(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    if number <= 0:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def _split_unit_price_and_quantity(raw_token: str, amount_no_tax: float | None) -> tuple[float | None, float | None]:
    token = re.sub(r"[^0-9.\s]", "", str(raw_token or "")).strip()
    if not token:
        return None, None

    parts = token.split()
    if len(parts) >= 2:
        unit_price = _to_float(parts[0])
        quantity = _to_float(parts[1])
        return unit_price, quantity

    if "." not in token:
        return _to_float(token), None

    best_price: float | None = None
    best_qty: float | None = None
    best_error = float("inf")

    # Some PDF texts merge "unit_price + quantity" into one token, e.g. "24.752475247524850"
    for suffix_digits in range(1, 5):
        if len(token) <= suffix_digits + 2:
            continue
        price_text = token[:-suffix_digits]
        qty_text = token[-suffix_digits:]
        if price_text.endswith("."):
            continue

        unit_price = _to_float(price_text)
        quantity = _to_float(qty_text)
        if unit_price is None or quantity is None or quantity <= 0:
            continue

        if amount_no_tax is not None and amount_no_tax > 0:
            error = abs(unit_price * quantity - amount_no_tax)
        else:
            error = abs(quantity - round(quantity))

        if error < best_error:
            best_error = error
            best_price = unit_price
            best_qty = quantity

    if best_price is not None:
        return best_price, best_qty

    return _to_float(token), None


def _split_lines(raw_text: str) -> list[str]:
    return [line for line in (_normalize_spaces(s) for s in raw_text.splitlines()) if line]


def _extract_invoice_number(raw_text: str, lines: list[str]) -> str | None:
    by_label = _first_match(
        [
            r"(?:发票号码|票据号码|No\.)[:：]?\s*([A-Z0-9]{8,20})",
            r"(?:发票代码)[:：]?\s*[A-Z0-9]{8,20}\s*(?:发票号码)[:：]?\s*([A-Z0-9]{8,20})",
        ],
        raw_text,
    )
    if by_label:
        return by_label

    for line in lines[:30]:
        match = re.search(r"\b(\d{20})\b", line)
        if match:
            return match.group(1)
    return None


def _extract_invoice_date(raw_text: str) -> str | None:
    return _first_match(
        [
            r"(?:开票日期|日期)[:：]?\s*([12]\d{3}年\d{1,2}月\d{1,2}日)",
            r"(?:开票日期|日期)[:：]?\s*([12]\d{3}[./-]\d{1,2}[./-]\d{1,2})",
            r"([12]\d{3}年\d{1,2}月\d{1,2}日)",
        ],
        raw_text,
    )


def _extract_amounts(raw_text: str, lines: list[str]) -> tuple[str | None, str | None]:
    amount_small = _normalize_amount(
        _first_match(
            [
                r"(?:价税合计\s*\(小写\)|\(小写\))[:：]?\s*([¥￥]?\s*[\d,]+(?:\.\d{1,2})?)",
            ],
            raw_text,
        )
    )

    # "合计 ¥17910.88 ¥179.12" => no_tax + tax
    no_tax_from_total_line = _normalize_amount(
        _first_match(
            [r"(?:^|[\s\r\n])合计[^\n\r]{0,40}?([¥￥]?\s*[\d,]+\.\d{2})\s+[¥￥]?\s*[\d,]+\.\d{2}"],
            raw_text,
        )
    )
    tax_from_total_line = _normalize_amount(
        _first_match(
            [r"(?:^|[\s\r\n])合计[^\n\r]{0,40}?[¥￥]?\s*[\d,]+\.\d{2}\s+([¥￥]?\s*[\d,]+\.\d{2})"],
            raw_text,
        )
    )

    derived_tax_from_totals = None
    amount_small_value = _to_float(amount_small)
    no_tax_total_value = _to_float(no_tax_from_total_line)
    if amount_small_value is not None and no_tax_total_value is not None and amount_small_value >= no_tax_total_value:
        derived_tax_from_totals = _format_amount(amount_small_value - no_tax_total_value)

    tax_amount = tax_from_total_line or derived_tax_from_totals
    if tax_amount is None:
        tax_amount = _normalize_amount(
            _first_match(
                [
                    r"(?:税额合计|税额)[:：]?\s*([¥￥]?\s*[\d,]+(?:\.\d{1,2})?)",
                ],
                raw_text,
            )
        )
    if tax_amount is None:
        # Last-resort fallback: pick the first row-level tax amount near a tax rate token.
        tax_amount = _normalize_amount(
            _first_match(
                [
                    r"\d{1,2}(?:\.\d+)?%\s*([\d,]+(?:\.\d{1,2})?)",
                ],
                raw_text,
            )
        )

    amount = amount_small
    if amount is None:
        # Fallback: derive total-with-tax from (no_tax + tax) when small total is not explicitly present.
        no_tax_value = _to_float(no_tax_from_total_line)
        tax_value = _to_float(tax_amount)
        if no_tax_value is not None and tax_value is not None:
            amount = _format_amount(no_tax_value + tax_value)

    if amount is None:
        amount = _normalize_amount(
            _first_match(
                [
                    r"(?:价税合计|金额合计|合计金额)[:：]?[^\n\r]{0,40}?([¥￥]?\s*[\d,]+(?:\.\d{1,2})?)",
                ],
                raw_text,
            )
        )

    if amount is not None and tax_amount is not None:
        return amount, tax_amount

    symbol_amounts: list[str] = []
    for line in lines:
        symbol_amounts.extend(re.findall(r"[¥￥]\s*([\d,]+\.\d{2})", line))
        symbol_amounts.extend(re.findall(r"([\d,]+\.\d{2})\s*[¥￥]", line))

    numeric = sorted(
        {float(v.replace(",", "")) for v in symbol_amounts if re.match(r"^\d[\d,]*\.\d{2}$", v)},
        reverse=True,
    )
    if amount is None and numeric:
        amount = f"{numeric[0]:.2f}"
    if tax_amount is None and len(numeric) >= 2:
        tax_amount = f"{numeric[-1]:.2f}"

    amount_value = _to_float(amount)
    tax_value = _to_float(tax_amount)
    if amount_value is not None and (tax_value is None or tax_value <= amount_value * 0.001):
        # Long invoices may contain page-level subtotals; infer tax from the nearest
        # no-tax total below the invoice total.
        all_money_values = sorted(
            {
                float(v.replace(",", ""))
                for v in re.findall(r"\d[\d,]*\.\d{2}", raw_text)
                if re.match(r"^\d[\d,]*\.\d{2}$", v)
            },
            reverse=True,
        )
        candidate_no_tax = [
            v
            for v in all_money_values
            if 0 < (amount_value - v) <= amount_value * 0.2
        ]
        if candidate_no_tax:
            inferred_tax = amount_value - max(candidate_no_tax)
            if 0 < inferred_tax <= amount_value * 0.2:
                tax_amount = _format_amount(inferred_tax)

    return amount, tax_amount


def _clean_company_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    cleaned = re.sub(r"^[A-Z0-9]{10,22}", "", cleaned)
    cleaned = re.sub(r"^名称[:：]?", "", cleaned)
    cleaned = cleaned.strip(" :：；，,")
    if len(cleaned) < 2:
        return None
    return cleaned


def _extract_buyer_seller(lines: list[str], raw_text: str) -> tuple[str | None, str | None]:
    buyer = _first_match(
        [
            r"(?:购买方信息|购买方|购方)[^\n\r]{0,40}?名称[:：]?\s*([^\n\r]{2,80})",
            r"购买方名称[:：]?\s*([^\n\r]{2,80})",
        ],
        raw_text,
    )
    seller = _first_match(
        [
            r"(?:销售方信息|销售方|销方)[^\n\r]{0,40}?名称[:：]?\s*([^\n\r]{2,80})",
            r"销售方名称[:：]?\s*([^\n\r]{2,80})",
            r"\d{15,20}\s*([^\d\n\r]{2,80}?(?:公司|经销处|门市|商行|中心|科技|器材|电子))\s*[A-Z0-9]{15,22}",
        ],
        raw_text,
    )

    if buyer is None:
        for line in lines:
            if any(hint in line for hint in ["大学", "学院", "研究院", "研究所"]):
                buyer = line
                break

    if seller is None:
        for line in lines:
            if "地址" in line or "开户行" in line or "账号" in line:
                continue
            if buyer and buyer in line:
                continue
            if any(hint in line for hint in ["公司", "经销处", "门市", "商行", "科技", "器材", "电子"]):
                seller = line
                break

    buyer_clean = _clean_company_name(buyer)
    seller_clean = _clean_company_name(seller)

    if seller_clean and buyer_clean and seller_clean == buyer_clean:
        seller_clean = None
        for line in lines:
            if buyer_clean in line or "地址" in line or "开户行" in line or "账号" in line:
                continue
            if any(hint in line for hint in ["公司", "经销处", "门市", "商行", "科技", "器材", "电子"]):
                seller_clean = _clean_company_name(line)
                if seller_clean:
                    break

    return buyer_clean, seller_clean


def _detect_bill_type(raw_text: str) -> str:
    mapping = [
        ("增值税专用发票", ["增值税专用发票", "专用发票"]),
        ("增值税普通发票", ["增值税普通发票", "普通发票", "电子发票(普通发票)"]),
        ("电子发票", ["电子发票", "全电发票"]),
        ("火车票", ["火车票", "高铁票", "动车票"]),
        ("机票", ["机票", "航空运输电子客票"]),
        ("出租车票", ["出租车", "打车"]),
        ("住宿发票", ["住宿", "酒店", "旅馆"]),
    ]
    for bill_type, keywords in mapping:
        if any(keyword in raw_text for keyword in keywords):
            return bill_type
    return "未知票据"


def _detect_item_content(raw_text: str, lines: list[str]) -> str | None:
    star_name = _first_match([r"(\*[^*\n\r]{2,40}\*)"], raw_text)
    if star_name:
        return star_name.replace("*", "")

    explicit = _first_match(
        [
            r"(?:项目名称|货物或应税劳务、服务名称)[:：]?\s*([^\n\r]{2,60})",
            r"(?:开票内容|备注)[:：]?\s*([^\n\r]{2,60})",
        ],
        raw_text,
    )
    if explicit and not any(token in explicit for token in ["规格型号", "税率", "数量", "金额"]):
        return explicit

    for idx, line in enumerate(lines):
        if "项目名称" in line and idx + 1 < len(lines):
            candidate = lines[idx + 1]
            if candidate and not any(token in candidate for token in ["规格型号", "税率", "税额"]):
                return candidate[:40]
    return None


def _extract_quantity_and_unit(text: str) -> tuple[str, str]:
    pattern = r"(?<!\d)(\d+(?:\.\d+)?)\s*(个|件|套|台|支|张|盒|瓶|米|公斤|kg|m)\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    pattern2 = r"\b(个|件|套|台|支|张|盒|瓶|米|公斤|kg|m)\s*(\d+(?:\.\d+)?)"
    match2 = re.search(pattern2, text, flags=re.IGNORECASE)
    if match2:
        return match2.group(2), match2.group(1)

    return "", ""


def _extract_structured_material_line_items(raw_text: str) -> list[dict[str, str]]:
    merged = re.sub(r"\s+", " ", raw_text or "")
    unit_pattern = "|".join(sorted((re.escape(token) for token in UNIT_TOKENS), key=len, reverse=True))
    pattern = re.compile(
        rf"(?P<name>\*[^*]{{1,80}}\*[^%]{{1,120}}?)\s*"
        rf"(?P<tax_rate>\d+(?:\.\d+)?%)\s*"
        rf"(?P<unit>{unit_pattern})\s*"
        rf"(?P<amount>\d+\.\d{{2}})\s*"
        rf"(?P<tax>\d+\.\d{{2}})\s*"
        rf"(?P<tail>[0-9.]+(?:\s+\d+(?:\.\d+)?)?)",
        flags=re.IGNORECASE,
    )

    items: list[dict[str, str]] = []
    for match in pattern.finditer(merged):
        amount_no_tax = _to_float(match.group("amount"))
        tax_amount = _to_float(match.group("tax"))
        unit_price, quantity = _split_unit_price_and_quantity(match.group("tail"), amount_no_tax)

        total_with_tax = None
        if amount_no_tax is not None:
            total_with_tax = amount_no_tax + (tax_amount or 0.0)

        items.append(
            {
                "item_name": _normalize_item_name(match.group("name")),
                "spec": "",
                "quantity": _normalize_quantity(quantity),
                "unit": match.group("unit"),
                "line_total_with_tax": _format_amount(total_with_tax),
                "amount_no_tax": _format_amount(amount_no_tax),
                "tax_amount": _format_amount(tax_amount),
                "unit_price": f"{unit_price:.10g}" if unit_price is not None else "",
                "tax_rate": match.group("tax_rate"),
            }
        )
    return items


def _extract_material_line_items(raw_text: str, amount: str | None, item_content: str | None) -> list[dict[str, str]]:
    structured_items = _extract_structured_material_line_items(raw_text)
    if structured_items:
        return structured_items

    merged = re.sub(r"[ \t\u3000]+", " ", raw_text.replace("\n", " "))
    blocks = re.findall(r"(\*[^*]{1,50}\*[^*]{0,220}?)(?=\*[^*]{1,50}\*|价税合计|合计|$)", merged)

    total_limit = None
    if amount:
        try:
            total_limit = float(amount)
        except ValueError:
            total_limit = None

    items: list[dict[str, str]] = []
    for block in blocks:
        title_match = re.search(r"(\*[^*]{1,50}\*)", block)
        if not title_match:
            continue

        item_name = title_match.group(1)
        tail = block[title_match.end() :].strip()
        if any(token in tail for token in ["项目名称", "规格型号", "税率", "税额"]):
            continue

        quantity, unit = _extract_quantity_and_unit(tail)
        quantity = _normalize_quantity(quantity)

        spec = ""
        tail_tokens = tail.split()
        if tail_tokens:
            first = tail_tokens[0]
            if not re.match(r"^[\d.]+$", first) and first not in UNIT_TOKENS and "*" not in first:
                spec = first[:24]

        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", tail)]
        filtered: list[float] = []
        for num in nums:
            if num <= 0 or num > 1e7:
                continue
            if total_limit is not None and num > total_limit * 2:
                continue
            filtered.append(num)

        line_total = ""
        if filtered:
            candidate = max(filtered)
            line_total = f"{candidate:.2f}"

        items.append(
            {
                "item_name": item_name,
                "spec": spec,
                "quantity": quantity,
                "unit": unit,
                "line_total_with_tax": line_total,
                "amount_no_tax": "",
                "tax_amount": "",
                "unit_price": "",
                "tax_rate": "",
            }
        )

    if items:
        return items

    fallback_name = item_content or "未识别项目"
    return [
        {
            "item_name": f"*{fallback_name}*" if "*" not in fallback_name else fallback_name,
            "spec": "",
            "quantity": "",
            "unit": "",
            "line_total_with_tax": amount or "",
            "amount_no_tax": "",
            "tax_amount": "",
            "unit_price": "",
            "tax_rate": "",
        }
    ]


def _rule_extract(raw_text: str) -> dict[str, object]:
    lines = _split_lines(raw_text)
    invoice_number = _extract_invoice_number(raw_text, lines)
    invoice_date = _extract_invoice_date(raw_text)
    amount, tax_amount = _extract_amounts(raw_text, lines)
    buyer, seller = _extract_buyer_seller(lines, raw_text)
    bill_type = _detect_bill_type(raw_text)
    item_content = _detect_item_content(raw_text, lines)
    line_items = _extract_material_line_items(raw_text, amount=amount, item_content=item_content)

    return {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "amount": amount,
        "tax_amount": tax_amount,
        "seller": seller,
        "buyer": buyer,
        "bill_type": bill_type,
        "item_content": item_content,
        "line_items": line_items,
    }


def _render_pdf_pages_to_base64_images(
    pdf_path: str | Path, max_pages: int = 2, render_scale: float = 1.2
) -> list[str]:
    path = Path(pdf_path)
    if not path.exists():
        return []

    images: list[str] = []
    doc = fitz.open(str(path))
    try:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
            images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    finally:
        doc.close()

    return images


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_line_item_row(row: dict[str, Any]) -> dict[str, str]:
    item_name = _normalize_item_name(str(_pick(row, ["item_name", "项目名称", "item"], "") or ""))
    spec = str(_pick(row, ["spec", "规格型号", "model"], "") or "").strip()
    item_name, spec = _split_item_name_and_spec(item_name, spec)
    quantity = _normalize_quantity(_pick(row, ["quantity", "数量"], ""))
    unit = str(_pick(row, ["unit", "单位"], "") or "").strip()

    amount_no_tax = _normalize_amount(str(_pick(row, ["amount_no_tax", "不含税金额", "金额", "line_amount"], "") or ""))
    tax_amount = _normalize_amount(str(_pick(row, ["tax_amount", "税额", "line_tax"], "") or ""))

    line_total_with_tax = _normalize_amount(
        str(_pick(row, ["line_total_with_tax", "每项含税总价", "含税总价", "价税合计", "line_total"], "") or "")
    )

    unit_price_raw = str(_pick(row, ["unit_price", "单价"], "") or "").strip()
    if not unit_price_raw and spec and re.fullmatch(r"\d+\.\d+", spec):
        # Some OCR/LLM outputs put unit price into "spec".
        unit_price_raw = spec
    unit_price_value = _to_float(re.sub(r"[^0-9.]", "", unit_price_raw))

    amount_no_tax_value = _to_float(amount_no_tax)
    tax_amount_value = _to_float(tax_amount)
    line_total_value = _to_float(line_total_with_tax)

    quantity_value = _to_float(quantity)
    if (
        (quantity_value is None or abs(quantity_value - round(quantity_value)) > 1e-6)
        and unit_price_value is not None
        and amount_no_tax_value is not None
        and unit_price_value > 0
    ):
        inferred_qty = amount_no_tax_value / unit_price_value
        if inferred_qty > 0 and abs(inferred_qty - round(inferred_qty)) < 0.05:
            quantity = str(int(round(inferred_qty)))
            quantity_value = float(quantity)

    if amount_no_tax_value is not None and tax_amount_value is not None:
        computed = amount_no_tax_value + tax_amount_value
        if line_total_value is None or abs(line_total_value - computed) > 0.02:
            line_total_value = computed
    elif line_total_value is None and amount_no_tax_value is not None:
        line_total_value = amount_no_tax_value

    if spec and re.fullmatch(r"\d+\.\d+", spec):
        spec = ""

    return {
        "item_name": item_name,
        "spec": spec,
        "quantity": quantity,
        "unit": unit,
        "line_total_with_tax": _format_amount(line_total_value) if line_total_value is not None else "",
        "amount_no_tax": amount_no_tax or "",
        "tax_amount": tax_amount or "",
        "unit_price": f"{unit_price_value:.10g}" if unit_price_value is not None else "",
        "tax_rate": str(_pick(row, ["tax_rate", "税率"], "") or ""),
    }


def _normalize_line_items(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    rows: list[dict[str, str]] = []
    for row in value:
        if isinstance(row, dict):
            rows.append(_normalize_line_item_row(row))
    return rows


def _chunk_list(values: list[Any], size: int) -> list[list[Any]]:
    step = max(1, size)
    return [values[i : i + step] for i in range(0, len(values), step)]


def _dedupe_line_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in items:
        key = tuple(str(row.get(field) or "").strip() for field in LINE_ITEM_BASE_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _normalize_tax_rate_number(rate_text: str) -> float:
    rate = _to_float(rate_text) or 1.0
    if rate <= 20:
        return rate

    # OCR often merges "... 80 1%" into "801%".
    compact = (rate_text or "").replace(".", "").strip()
    for tail in ("13", "9", "6", "3", "1"):
        if compact.endswith(tail):
            return float(tail)
    return 1.0


def _pick_amount_tax_pair(numbers: list[str], expected_rate: float) -> tuple[int, int, float, float] | None:
    if len(numbers) < 2:
        return None

    common_rates = [0.01, 0.03, 0.06, 0.09, 0.13]
    target_rates = [expected_rate] + [r for r in common_rates if abs(r - expected_rate) > 1e-9]

    best: tuple[tuple[float, ...], int, int, float, float] | None = None
    for amount_idx in range(len(numbers) - 1):
        amount_text = numbers[amount_idx]
        amount_decimal = amount_text.split(".")[-1] if "." in amount_text else ""
        if len(amount_decimal) != 2:
            continue

        amount = _to_float(amount_text)
        if amount is None or amount <= 0:
            continue

        for tax_idx in range(amount_idx + 1, len(numbers)):
            tax_text = numbers[tax_idx]
            tax_decimal = tax_text.split(".")[-1] if "." in tax_text else ""
            if len(tax_decimal) != 2:
                continue

            tax = _to_float(tax_text)
            if tax is None or tax < 0 or tax > amount:
                continue

            ratio = tax / amount
            score_any = min(abs(ratio - r) for r in target_rates)
            if score_any > 0.03:
                continue

            score_expected = abs(ratio - expected_rate)
            # Prefer earlier pairs first to avoid page subtotal/account numbers in tail text.
            score = (amount_idx, score_expected, score_any, tax_idx - amount_idx)
            if best is None or score < best[0]:
                best = (score, amount_idx, tax_idx, amount, tax)

    if not best:
        return None
    return best[1], best[2], best[3], best[4]


def _extract_long_mode_candidates(raw_text: str) -> list[dict[str, str]]:
    lines = [_normalize_spaces(line) for line in (raw_text or "").splitlines() if line.strip()]
    if not lines:
        return []

    merged_text = _normalize_spaces(" ".join(lines))
    row_start_re = re.compile(r"\*[\u4e00-\u9fa5]{2,10}\*")
    row_starts = list(row_start_re.finditer(merged_text))
    if not row_starts:
        return []

    long_units = sorted(
        {
            *UNIT_TOKENS,
            "包",
            "块",
            "只",
            "把",
            "条",
            "组",
            "卷",
            "袋",
            "箱",
            "双",
            "对",
            "批",
            "根",
            "片",
            "罐",
            "部",
            "平米",
            "平方",
            "平方米",
            "㎡",
            "m²",
            "千支",
            "份",
            "票",
            "次",
        },
        key=len,
        reverse=True,
    )
    unit_pattern = "(?:" + "|".join(re.escape(token) for token in long_units) + r"|[\u4e00-\u9fa5]{1,3})"
    row_head_re = re.compile(
        rf"^(?P<prefix>.+?)\s*(?P<tax_rate>\d+(?:\.\d+)?)%\s*(?P<unit>{unit_pattern})\s*(?P<numbers>.+)$",
        flags=re.IGNORECASE,
    )
    number_re = re.compile(r"\d+(?:\.\d+)?")

    candidates: list[dict[str, str]] = []
    for idx, start in enumerate(row_starts):
        end_pos = row_starts[idx + 1].start() if idx + 1 < len(row_starts) else len(merged_text)
        segment = merged_text[start.start() : end_pos].replace("¥", " ").replace("￥", " ").strip()
        if not segment:
            continue

        match = row_head_re.search(segment)
        if not match:
            continue

        prefix = match.group("prefix").strip()
        if "*" in prefix:
            prefix = prefix[prefix.find("*") :].strip()
        else:
            continue

        item_name, spec = _split_long_prefix_item_spec(prefix)
        if not item_name:
            continue

        tax_rate_value = _normalize_tax_rate_number(match.group("tax_rate"))
        tax_rate_text = f"{int(round(tax_rate_value))}%" if abs(tax_rate_value - round(tax_rate_value)) < 1e-6 else f"{tax_rate_value:g}%"

        numbers = [m.group(0) for m in number_re.finditer(match.group("numbers"))][:10]
        picked = _pick_amount_tax_pair(numbers, expected_rate=tax_rate_value / 100.0)
        if not picked:
            continue

        amount_idx, tax_idx, amount_value, tax_value = picked
        tail_tokens: list[str] = []
        if amount_idx > 0:
            tail_tokens.extend(numbers[:amount_idx])
            if tax_idx > amount_idx + 1:
                tail_tokens.extend(numbers[amount_idx + 1 : tax_idx])
        elif tax_idx + 1 < len(numbers):
            tail_tokens.append(numbers[tax_idx + 1])
        tail = " ".join(tail_tokens[:2]).strip()

        candidates.append(
            {
                "name": item_name,
                "spec": spec,
                "tax_rate": tax_rate_text,
                "unit": match.group("unit"),
                "amount": f"{amount_value:.2f}",
                "tax": f"{tax_value:.2f}",
                "tail": tail,
            }
        )

    return candidates


def _candidate_to_item_row(candidate: dict[str, str]) -> dict[str, str]:
    amount_no_tax = _to_float(candidate.get("amount"))
    tax_amount = _to_float(candidate.get("tax"))
    unit_price, quantity = _split_unit_price_and_quantity(candidate.get("tail", ""), amount_no_tax)

    line_total = None
    if amount_no_tax is not None:
        line_total = amount_no_tax + (tax_amount or 0.0)

    row = {
        "item_name": candidate.get("name") or "",
        "spec": candidate.get("spec") or "",
        "quantity": _normalize_quantity(quantity),
        "unit": candidate.get("unit") or "",
        "line_total_with_tax": _format_amount(line_total),
        "amount_no_tax": _format_amount(amount_no_tax),
        "tax_amount": _format_amount(tax_amount),
        "unit_price": f"{unit_price:.10g}" if unit_price is not None else "",
        "tax_rate": candidate.get("tax_rate") or "",
    }
    return _normalize_line_item_row(row)


def _extract_line_items_from_llm_payload(parsed: dict[str, Any]) -> list[dict[str, str]]:
    line_items = _pick(parsed, ["line_items", "items", "明细", "明细表"], [])
    return _normalize_line_items(line_items)


def _extract_chunk_with_ollama_text(chunk: list[dict[str, str]]) -> list[dict[str, str]]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    timeout = _env_int("LONG_MODE_CHUNK_TIMEOUT", 90)
    request_timeout = (10, timeout)

    payload_text = json.dumps(chunk, ensure_ascii=False, indent=2)
    prompt = (
        "你是财务发票明细整理助手。请根据候选数据输出严格 JSON。\n"
        "输出格式：{\"line_items\":[...]}\n"
        "line_items 每项必须包含：item_name,spec,quantity,unit,amount_no_tax,tax_amount,line_total_with_tax\n"
        "规则：\n"
        "1) 不允许合并或丢失行，输入几行就输出几行。\n"
        "2) line_total_with_tax = amount_no_tax + tax_amount（保留两位小数）。\n"
        "3) quantity 只填数字；无法判断时可留空字符串。\n"
        "4) spec 仅填规格/型号；若 item_name 尾部重复规格，需拆到 spec。\n"
        "只输出 JSON，不要代码块。\n"
        f"候选数据：{payload_text}"
    )

    payload_generate = {
        "model": model,
        "stream": False,
        "prompt": prompt,
        "options": {"temperature": 0},
    }

    errors: list[str] = []

    try:
        gen_resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=request_timeout)
        gen_resp.raise_for_status()
        parsed = _extract_json_from_text(gen_resp.json().get("response", ""))
        if parsed:
            rows = _extract_line_items_from_llm_payload(parsed)
            if rows:
                return rows
        errors.append("LLM returned empty/non-JSON line_items from /api/generate.")
    except requests.RequestException as exc:
        errors.append(f"/api/generate failed: {exc}")

    payload_chat = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0},
    }

    try:
        chat_resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=request_timeout)
        chat_resp.raise_for_status()
        chat_content = (chat_resp.json().get("message") or {}).get("content", "")
        parsed = _extract_json_from_text(chat_content)
        if parsed:
            rows = _extract_line_items_from_llm_payload(parsed)
            if rows:
                return rows
        errors.append("LLM returned empty/non-JSON line_items from /api/chat.")
    except requests.RequestException as exc:
        errors.append(f"/api/chat failed: {exc}")

    raise RuntimeError("; ".join(errors))


def _is_long_invoice_mode(raw_text: str, rule_fields: dict[str, object]) -> bool:
    if _env_flag_true("LONG_MODE_FORCE"):
        return True
    line_threshold = _env_int("LONG_MODE_LINE_THRESHOLD", 25)
    text_threshold = _env_int("LONG_MODE_TEXT_LEN_THRESHOLD", 3000)

    star_count = raw_text.count("*")
    parsed_rows = len((rule_fields.get("line_items") or [])) if isinstance(rule_fields.get("line_items"), list) else 0
    return star_count >= line_threshold or len(raw_text) >= text_threshold or parsed_rows >= line_threshold


def _extract_long_mode_fields(
    raw_text: str,
    rule_fields: dict[str, object],
    use_ollama: bool,
) -> dict[str, object]:
    chunk_size = _env_int("LONG_MODE_CHUNK_SIZE", 10)
    use_text_llm = use_ollama and _env_flag_true("LONG_MODE_USE_TEXT_LLM")
    candidates = _extract_long_mode_candidates(raw_text)

    result = dict(rule_fields)
    warnings: list[str] = []
    llm_chunks_ok = 0
    llm_chunks_failed = 0

    if not candidates:
        warnings.append("Long mode candidate extraction returned 0 rows, fallback to base line_items.")
        base_items = _normalize_line_items(result.get("line_items"))
        result["line_items"] = _dedupe_line_items(base_items)
        result["processing_mode"] = "long_chunked"
        result["long_mode_stats"] = {
            "candidate_rows": 0,
            "final_rows": len(result.get("line_items") or []),
            "chunk_size": chunk_size,
            "chunks_total": 0,
            "llm_enabled": use_text_llm,
            "llm_chunks_ok": 0,
            "llm_chunks_failed": 0,
        }
        result["warnings"] = warnings
        return _reconcile_extracted_fields(result)

    merged_rows: list[dict[str, str]] = []
    chunks = _chunk_list(candidates, chunk_size)

    for chunk in chunks:
        chunk_rows: list[dict[str, str]]
        if use_text_llm:
            try:
                chunk_rows = _extract_chunk_with_ollama_text(chunk)
                llm_chunks_ok += 1
            except Exception:
                llm_chunks_failed += 1
                chunk_rows = [_candidate_to_item_row(candidate) for candidate in chunk]
        else:
            chunk_rows = [_candidate_to_item_row(candidate) for candidate in chunk]
        merged_rows.extend(chunk_rows)

    final_rows = _dedupe_line_items(_normalize_line_items(merged_rows))
    if len(final_rows) < len(candidates):
        warnings.append(f"Some rows were dropped after normalization: candidates={len(candidates)}, final={len(final_rows)}")
    if use_text_llm and llm_chunks_failed > 0:
        warnings.append(f"{llm_chunks_failed} chunk(s) failed in LLM extraction and used rule fallback.")

    result["line_items"] = final_rows
    result["processing_mode"] = "long_chunked"
    result["long_mode_stats"] = {
        "candidate_rows": len(candidates),
        "final_rows": len(final_rows),
        "chunk_size": chunk_size,
        "chunks_total": len(chunks),
        "llm_enabled": use_text_llm,
        "llm_chunks_ok": llm_chunks_ok,
        "llm_chunks_failed": llm_chunks_failed,
    }
    if warnings:
        result["warnings"] = warnings

    return _reconcile_extracted_fields(result)


def _merge_line_items(rule_items: list[dict[str, str]], llm_items: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rule_items:
        return llm_items
    if not llm_items:
        return rule_items

    if len(rule_items) != len(llm_items):
        return rule_items if len(rule_items) >= len(llm_items) else llm_items

    merged_rows: list[dict[str, str]] = []
    for rule_row, llm_row in zip(rule_items, llm_items):
        row = dict(rule_row)

        llm_item_name = str(llm_row.get("item_name") or "").strip()
        rule_item_name = str(rule_row.get("item_name") or "").strip()
        if llm_item_name and len(llm_item_name.replace(" ", "")) >= len(rule_item_name.replace(" ", "")):
            row["item_name"] = llm_item_name

        llm_spec = str(llm_row.get("spec") or "").strip()
        if llm_spec and not re.fullmatch(r"\d+\.\d+", llm_spec):
            row["spec"] = llm_spec

        llm_quantity = _normalize_quantity(llm_row.get("quantity"))
        llm_quantity_value = _to_float(llm_quantity)
        if llm_quantity and llm_quantity_value is not None and abs(llm_quantity_value - round(llm_quantity_value)) < 1e-6:
            row["quantity"] = llm_quantity

        llm_unit = str(llm_row.get("unit") or "").strip()
        if llm_unit:
            row["unit"] = llm_unit

        for field in ["amount_no_tax", "tax_amount", "line_total_with_tax", "unit_price", "tax_rate"]:
            if not str(row.get(field) or "").strip() and str(llm_row.get(field) or "").strip():
                row[field] = str(llm_row.get(field) or "")

        merged_rows.append(_normalize_line_item_row(row))

    return merged_rows


def _reconcile_extracted_fields(data: dict[str, object]) -> dict[str, object]:
    result = dict(data)
    line_items = result.get("line_items")
    if isinstance(line_items, list):
        normalized_items = _normalize_line_items(line_items)
        result["line_items"] = normalized_items

        total_with_tax_sum = sum(_to_float(item.get("line_total_with_tax")) or 0.0 for item in normalized_items)
        total_no_tax_sum = sum(_to_float(item.get("amount_no_tax")) or 0.0 for item in normalized_items)

        amount_value = _to_float(result.get("amount"))
        tax_value = _to_float(result.get("tax_amount"))

        if total_with_tax_sum > 0:
            if amount_value is None:
                result["amount"] = _format_amount(total_with_tax_sum)
            elif tax_value is not None:
                # Typical OCR error: amount picked as no-tax total; if so, correct to tax-included total.
                if abs((amount_value + tax_value) - total_with_tax_sum) <= 0.1 and abs(amount_value - total_with_tax_sum) > 0.1:
                    result["amount"] = _format_amount(total_with_tax_sum)
            elif abs(amount_value - total_with_tax_sum) <= 0.1:
                result["amount"] = _format_amount(total_with_tax_sum)

        if total_with_tax_sum > 0 and total_no_tax_sum > 0:
            inferred_tax = total_with_tax_sum - total_no_tax_sum
            if inferred_tax >= 0:
                # Only trust inferred tax when tax is missing, or when line-item totals
                # are already consistent with invoice total (avoids overriding header tax
                # for long invoices with partial line extraction).
                if tax_value is None:
                    result["tax_amount"] = _format_amount(inferred_tax)
                elif amount_value is not None and abs(amount_value - total_with_tax_sum) <= 0.2 and abs(tax_value - inferred_tax) > 0.2:
                    result["tax_amount"] = _format_amount(inferred_tax)

    return result


def _normalize_llm_fields(raw: dict[str, Any]) -> dict[str, object]:
    buyer_block = _pick(raw, ["buyer_info", "购买方信息"], {}) or {}
    seller_block = _pick(raw, ["seller_info", "销售方信息"], {}) or {}

    buyer = _pick(raw, ["buyer", "buyer_name", "购买方", "购买方名称"])
    seller = _pick(raw, ["seller", "seller_name", "销售方", "销售方名称"])
    if not buyer and isinstance(buyer_block, dict):
        buyer = _pick(buyer_block, ["name", "名称"])
    if not seller and isinstance(seller_block, dict):
        seller = _pick(seller_block, ["name", "名称"])

    line_items = _pick(raw, ["line_items", "明细", "明细表"])
    if line_items in (None, [], ""):
        item_name = _pick(raw, ["item_content", "item_name", "项目内容", "项目名称"], "")
        if item_name:
            line_items = [
                {
                    "item_name": item_name,
                    "spec": _pick(raw, ["spec", "规格型号"], ""),
                    "quantity": _pick(raw, ["quantity", "数量"], ""),
                    "unit": _pick(raw, ["unit", "单位"], ""),
                    "line_total_with_tax": _pick(raw, ["line_total_with_tax", "含税总价"], ""),
                    "amount_no_tax": _pick(raw, ["amount_no_tax", "金额"], ""),
                    "tax_amount": _pick(raw, ["tax_amount", "税额"], ""),
                    "unit_price": _pick(raw, ["unit_price", "单价"], ""),
                    "tax_rate": _pick(raw, ["tax_rate", "税率"], ""),
                }
            ]

    amount_raw = _pick(raw, ["amount", "价税合计(小写)", "价税合计小写", "total_with_tax"])
    tax_raw = _pick(raw, ["tax_amount", "税额", "tax"])

    return {
        "invoice_number": _pick(raw, ["invoice_number", "发票号码"]),
        "invoice_date": _pick(raw, ["invoice_date", "开票日期"]),
        "amount": _normalize_amount(str(amount_raw)) if amount_raw else None,
        "tax_amount": _normalize_amount(str(tax_raw)) if tax_raw else None,
        "seller": seller,
        "buyer": buyer,
        "bill_type": _pick(raw, ["bill_type", "票据类型"]),
        "item_content": _pick(raw, ["item_content", "项目内容", "项目名称"]),
        "line_items": _normalize_line_items(line_items),
    }


def _merge_fields(rule_fields: dict[str, object], llm_fields: dict[str, object]) -> dict[str, object]:
    merged = dict(rule_fields)
    rule_line_items = _normalize_line_items(rule_fields.get("line_items"))
    llm_line_items = _normalize_line_items(llm_fields.get("line_items"))

    for key, value in llm_fields.items():
        if key == "line_items":
            merged[key] = _merge_line_items(rule_line_items, llm_line_items)
            continue
        if value not in (None, "", []):
            merged[key] = value
    return _reconcile_extracted_fields(merged)


def _extract_with_ollama_vl(raw_text: str, pdf_path: str | Path) -> dict[str, object]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    max_pages = int(os.getenv("OLLAMA_MAX_PAGES", "1"))
    render_scale = float(os.getenv("OLLAMA_RENDER_SCALE", "1.2"))

    request_timeout = (10, timeout)
    images = _render_pdf_pages_to_base64_images(pdf_path, max_pages=max_pages, render_scale=render_scale)
    if not images:
        raise RuntimeError("No image rendered from PDF.")

    prompt = (
        "你是财务票据抽取助手。请根据发票图片输出严格 JSON，不要输出其他文本。\n"
        "必须包含字段：invoice_number, invoice_date, amount, tax_amount, seller, buyer, bill_type, item_content, line_items。\n"
        "line_items 是数组，每项包含：item_name, spec, quantity, unit, amount_no_tax, tax_amount, unit_price, line_total_with_tax。\n"
        "注意：发票“金额”通常是不含税金额，line_total_with_tax 必须是 含税金额（= amount_no_tax + tax_amount）。\n"
        "如果某字段无法识别，请填 null 或空数组。金额统一保留两位小数。\n"
        f"OCR参考文本：{raw_text[:1500]}"
    )

    payload_generate = {
        "model": model,
        "stream": False,
        "prompt": prompt,
        "images": images,
        "options": {"temperature": 0},
    }

    errors: list[str] = []

    # Prefer /api/generate first: more stable across local Ollama versions.
    try:
        gen_resp = requests.post(f"{base_url}/api/generate", json=payload_generate, timeout=request_timeout)
        gen_resp.raise_for_status()
        gen_content = gen_resp.json().get("response", "")
        parsed = _extract_json_from_text(gen_content)
        if parsed:
            return _normalize_llm_fields(parsed)
        errors.append("LLM returned non-JSON content from /api/generate.")
    except requests.RequestException as exc:
        errors.append(f"/api/generate request failed: {exc}")

    payload_chat = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "options": {"temperature": 0},
    }

    try:
        chat_resp = requests.post(f"{base_url}/api/chat", json=payload_chat, timeout=request_timeout)
        chat_resp.raise_for_status()
        chat_content = (chat_resp.json().get("message") or {}).get("content", "")
        parsed = _extract_json_from_text(chat_content)
        if parsed:
            return _normalize_llm_fields(parsed)
        errors.append("LLM returned non-JSON content from /api/chat.")
    except requests.RequestException as exc:
        errors.append(f"/api/chat request failed: {exc}")

    raise RuntimeError("; ".join(errors))


def extract_invoice_fields(raw_text: str, pdf_path: str | Path | None = None) -> dict[str, object]:
    rule_fields = _reconcile_extracted_fields(_rule_extract(raw_text))
    use_ollama = _env_flag_true("USE_OLLAMA_VL")
    is_long_mode = _is_long_invoice_mode(raw_text, rule_fields)

    if is_long_mode:
        long_result = _extract_long_mode_fields(raw_text, rule_fields, use_ollama=use_ollama)
        if use_ollama and _env_flag_true("LONG_MODE_USE_TEXT_LLM"):
            llm_failed = ((long_result.get("long_mode_stats") or {}).get("llm_chunks_failed") or 0) > 0
            long_result["extraction_source"] = "long_chunk_mixed" if llm_failed else "long_chunk_llm"
        else:
            long_result["extraction_source"] = "long_rule_chunked"
        if "llm_error" not in long_result:
            long_result["llm_error"] = None
        return long_result

    if not use_ollama or not pdf_path:
        rule_fields["extraction_source"] = "rule_only"
        if not use_ollama:
            rule_fields["llm_error"] = "USE_OLLAMA_VL is false or .env not loaded."
        else:
            rule_fields["llm_error"] = "pdf_path missing."
        return _reconcile_extracted_fields(rule_fields)

    try:
        llm_fields = _extract_with_ollama_vl(raw_text, pdf_path)
        merged = _merge_fields(rule_fields, llm_fields)
        merged["extraction_source"] = "ollama_vl"
        merged["llm_error"] = None
        return merged
    except Exception as exc:
        rule_fields["extraction_source"] = "rule_fallback"
        rule_fields["llm_error"] = str(exc)
        return _reconcile_extracted_fields(rule_fields)
