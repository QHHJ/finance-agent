from __future__ import annotations

from collections import Counter
from typing import Any

from app.db.models import PolicyDocument


CATEGORY_RULES: list[dict[str, Any]] = [
    {
        "category": "差旅费",
        "keywords": ["差旅", "机票", "火车票", "高铁", "酒店", "住宿", "出差", "打车", "航班", "行程"],
        "required_materials": ["发票或电子票据", "行程单/车票信息", "支付凭证", "出差审批单"],
    },
    {
        "category": "材料费",
        "keywords": [
            "材料费",
            "材料",
            "实验室",
            "入库",
            "电子元件",
            "金属制品",
            "法兰",
            "安装工具",
            "变压器",
            "接地排",
            "器件",
            "耗材",
        ],
        "required_materials": ["发票", "入库明细Excel", "采购或领用说明", "支付凭证"],
    },
    {
        "category": "办公费",
        "keywords": ["办公", "文具", "打印", "办公用品", "电脑配件"],
        "required_materials": ["发票或收据", "采购清单", "签收或入库记录", "支付凭证"],
    },
    {
        "category": "业务招待费",
        "keywords": ["招待", "接待", "餐饮", "宴请", "商务餐"],
        "required_materials": ["发票", "接待审批单", "接待对象与事由说明", "支付凭证"],
    },
    {
        "category": "培训费",
        "keywords": ["培训", "课程", "研讨会", "会务"],
        "required_materials": ["发票", "培训通知或议程", "参训名单", "支付凭证"],
    },
    {
        "category": "软件服务费",
        "keywords": ["软件", "订阅", "云服务", "技术服务", "系统服务", "SaaS"],
        "required_materials": ["发票", "合同或订单", "服务验收记录", "支付凭证"],
    },
]


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _choose_rule(raw_text: str, extracted_data: dict[str, Any]) -> dict[str, Any]:
    combined = " ".join(
        [
            raw_text,
            extracted_data.get("bill_type") or "",
            extracted_data.get("item_content") or "",
            extracted_data.get("seller") or "",
            extracted_data.get("buyer") or "",
        ]
    )

    best_rule: dict[str, Any] | None = None
    best_score = 0
    for rule in CATEGORY_RULES:
        score = sum(1 for keyword in rule["keywords"] if keyword in combined)
        if score > best_score:
            best_score = score
            best_rule = rule

    if best_rule:
        return best_rule

    return {
        "category": "其他费用",
        "keywords": [],
        "required_materials": ["发票或有效票据", "费用说明", "支付凭证"],
    }


def _extract_policy_refs(policies: list[PolicyDocument], keywords: list[str]) -> list[str]:
    refs: list[str] = []
    for policy in policies:
        lines = [line.strip() for line in policy.raw_text.splitlines() if line.strip()]
        for line in lines:
            if len(line) < 6:
                continue
            if keywords and not any(keyword in line for keyword in keywords):
                continue
            refs.append(f"{policy.name}: {line[:80]}")
            if len(refs) >= 4:
                return refs

    if not refs and policies:
        refs.append(f"{policies[0].name}: 未命中明确条款，建议人工复核制度原文。")
    return refs


def _historical_hint(samples: list[dict[str, str]]) -> tuple[str | None, int, list[str]]:
    if not samples:
        return None, 0, []
    counter = Counter(sample["expense_category"] for sample in samples if sample.get("expense_category"))
    if not counter:
        return None, 0, []
    category, count = counter.most_common(1)[0]
    refs = [f"历史任务 {sample['task_id']} -> {sample['expense_category']}" for sample in samples[:3]]
    return category, count, refs


def suggest_processing(
    extracted_data: dict[str, Any],
    raw_text: str,
    policies: list[PolicyDocument],
    historical_samples: list[dict[str, str]],
) -> dict[str, Any]:
    rule = _choose_rule(raw_text, extracted_data)
    suggested_category = rule["category"]
    required_materials = list(rule["required_materials"])
    risk_points: list[str] = []

    policy_keywords = list(rule.get("keywords", [])) + [suggested_category]
    policy_references = _extract_policy_refs(policies, policy_keywords)

    amount = _safe_float(extracted_data.get("amount"))
    if not extracted_data.get("invoice_number"):
        risk_points.append("未识别发票号码，票据真实性校验存在风险。")
    if not extracted_data.get("invoice_date"):
        risk_points.append("未识别开票日期，可能影响报销时效校验。")
    if amount is None:
        risk_points.append("未识别金额，建议人工核对票面金额。")
    elif amount >= 5000:
        risk_points.append("金额较高，建议复核预算占用与审批权限。")

    if extracted_data.get("bill_type") == "未知票据":
        risk_points.append("票据类型未识别，建议补充票据说明或人工分类。")

    if suggested_category == "差旅费":
        has_itinerary = any(token in raw_text for token in ["行程单", "登机牌", "车票", "酒店订单", "携程"])
        if not has_itinerary:
            risk_points.append("疑似差旅单据但未识别到完整行程附件。")

    if suggested_category == "材料费":
        has_inventory_signal = any(token in raw_text for token in ["入库", "Excel", "规格型号", "数量", "单位"])
        if not has_inventory_signal:
            risk_points.append("材料费通常需补充入库明细Excel，当前未识别到相关信息。")
        else:
            risk_points.append("请核验入库明细Excel与发票明细、合计金额一致。")

    if not extracted_data.get("seller") or not extracted_data.get("buyer"):
        risk_points.append("购销方信息不完整，建议补充清晰票面。")

    historical_category, historical_count, similar_refs = _historical_hint(historical_samples)
    rationale = f"规则命中类别：{rule['category']}。"
    if historical_category and historical_count >= 2 and historical_category == suggested_category:
        rationale += f" 同类历史样例中“{historical_category}”出现 {historical_count} 次，按经验增强置信度。"
    elif historical_category:
        rationale += f" 发现少量历史样例倾向“{historical_category}”，仅作参考。"

    return {
        "expense_category": suggested_category,
        "required_materials": _dedupe(required_materials),
        "risk_points": _dedupe(risk_points),
        "policy_references": policy_references,
        "rationale": rationale,
        "similar_case_refs": similar_refs,
    }
