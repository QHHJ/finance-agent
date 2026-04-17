from __future__ import annotations

from collections import Counter
from typing import Any

from app.db.models import PolicyDocument
from app.services import rag_retriever


CATEGORY_RULES: list[dict[str, Any]] = [
    {
        "category": "差旅费",
        "keywords": ["差旅", "机票", "火车票", "高铁", "酒店", "住宿", "出差", "航班", "行程", "打车"],
        "required_materials": ["发票或电子票据", "行程单/车票信息", "支付凭证", "出差审批单"],
    },
    {
        "category": "材料费",
        "keywords": ["材料", "实验室", "入库", "电子元件", "金属制品", "器件", "耗材", "规格型号", "数量"],
        "required_materials": ["发票", "入库明细Excel", "采购/领用说明", "支付凭证"],
    },
    {
        "category": "办公费",
        "keywords": ["办公", "文具", "打印", "办公用品", "电脑配件"],
        "required_materials": ["发票", "采购清单", "签收或入库记录", "支付凭证"],
    },
    {
        "category": "业务招待费",
        "keywords": ["招待", "接待", "餐饮", "商务餐", "宴请"],
        "required_materials": ["发票", "接待审批单", "接待对象及事由说明", "支付凭证"],
    },
    {
        "category": "培训费",
        "keywords": ["培训", "课程", "研讨会", "会务"],
        "required_materials": ["发票", "培训通知/议程", "参训名单", "支付凭证"],
    },
    {
        "category": "软件服务费",
        "keywords": ["软件", "订阅", "云服务", "技术服务", "系统服务", "saas"],
        "required_materials": ["发票", "合同或订单", "服务验收记录", "支付凭证"],
    },
]


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        output.append(clean)
    return output


def _rule_for_category(category: str) -> dict[str, Any] | None:
    for rule in CATEGORY_RULES:
        if rule["category"] == category:
            return rule
    return None


def _choose_rule(raw_text: str, extracted_data: dict[str, Any]) -> dict[str, Any]:
    combined = " ".join(
        [
            str(raw_text or "").lower(),
            str(extracted_data.get("bill_type") or "").lower(),
            str(extracted_data.get("item_content") or "").lower(),
            str(extracted_data.get("seller") or "").lower(),
            str(extracted_data.get("buyer") or "").lower(),
        ]
    )

    best_rule: dict[str, Any] | None = None
    best_score = 0
    for rule in CATEGORY_RULES:
        score = sum(1 for keyword in rule["keywords"] if keyword.lower() in combined)
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


def _extract_policy_refs_from_text(policies: list[PolicyDocument], keywords: list[str]) -> list[str]:
    refs: list[str] = []
    for policy in policies:
        lines = [line.strip() for line in str(policy.raw_text or "").splitlines() if line.strip()]
        for line in lines:
            if len(line) < 6:
                continue
            if keywords and not any(keyword in line for keyword in keywords):
                continue
            refs.append(f"{policy.name}: {line[:80]}")
            if len(refs) >= 4:
                return refs

    if not refs and policies:
        refs.append(f"{policies[0].name}: 未命中精确条款，建议人工复核制度原文。")
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


def _rag_case_hint(case_hits: list[dict[str, Any]]) -> tuple[str | None, int, list[str], float]:
    categories: list[str] = []
    refs: list[str] = []
    best_score = 0.0

    for hit in case_hits:
        score = float(hit.get("score") or 0.0)
        if score > best_score:
            best_score = score
        metadata = dict(hit.get("metadata") or {})
        category = str(metadata.get("expense_category") or "").strip()
        if category:
            categories.append(category)
        if category:
            refs.append(f"RAG样例({score:.2f}): {category}")

    if not categories:
        return None, 0, refs[:3], best_score

    counter = Counter(categories)
    category, count = counter.most_common(1)[0]
    return category, count, refs[:3], best_score


def suggest_processing(
    extracted_data: dict[str, Any],
    raw_text: str,
    policies: list[PolicyDocument],
    historical_samples: list[dict[str, str]],
) -> dict[str, Any]:
    rule = _choose_rule(raw_text, extracted_data)
    suggested_category = str(rule["category"])
    required_materials = list(rule["required_materials"])
    risk_points: list[str] = []

    rag_bundle = rag_retriever.build_material_references(extracted_data, raw_text)
    rag_policy_refs = list(rag_bundle.get("policy_refs") or [])
    rag_case_hits = list(rag_bundle.get("case_hits") or [])

    policy_keywords = list(rule.get("keywords", [])) + [suggested_category]
    policy_references = rag_policy_refs or _extract_policy_refs_from_text(policies, policy_keywords)

    rag_category, rag_count, rag_case_refs, rag_best_score = _rag_case_hint(rag_case_hits)
    if rag_category and rag_category != suggested_category:
        strong_case_support = (rag_count >= 2 and rag_best_score >= 0.45) or rag_best_score >= 0.86
        if strong_case_support:
            suggested_category = rag_category
            target_rule = _rule_for_category(suggested_category)
            if target_rule:
                required_materials = list(target_rule["required_materials"])
            risk_points.append(f"命中高相似历史样例，已将类别修正为：{suggested_category}。")
        else:
            risk_points.append(f"历史样例倾向 {rag_category}，建议人工复核类别。")

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
            risk_points.append("疑似差旅票据但未识别到完整行程附件。")

    if suggested_category == "材料费":
        has_inventory_signal = any(token in raw_text for token in ["入库", "Excel", "规格型号", "数量", "单位"])
        if not has_inventory_signal:
            risk_points.append("材料费通常需补充入库明细Excel，当前未识别到相关信息。")
        else:
            risk_points.append("请核验入库明细Excel与发票明细、合计金额一致。")

    if not extracted_data.get("seller") or not extracted_data.get("buyer"):
        risk_points.append("购销方信息不完整，建议补充清晰票面。")

    historical_category, historical_count, historical_refs = _historical_hint(historical_samples)

    rationale_parts = [f"规则命中类别：{rule['category']}。"]
    if rag_category:
        rationale_parts.append(f"RAG历史样例主倾向：{rag_category}（命中 {rag_count} 条，高分 {rag_best_score:.2f}）。")
    if historical_category:
        rationale_parts.append(f"任务历史样例倾向：{historical_category}（{historical_count} 条）。")

    similar_case_refs = _dedupe(historical_refs + rag_case_refs)

    rag_trace = {
        "policy_hits": [
            {
                "doc_key": hit.get("doc_key"),
                "score": hit.get("score"),
                "title": hit.get("title"),
                "source_id": hit.get("source_id"),
            }
            for hit in rag_bundle.get("policy_hits", [])[:5]
        ],
        "case_hits": [
            {
                "doc_key": hit.get("doc_key"),
                "score": hit.get("score"),
                "title": hit.get("title"),
                "source_id": hit.get("source_id"),
                "expense_category": (hit.get("metadata") or {}).get("expense_category"),
            }
            for hit in rag_case_hits[:5]
        ],
    }

    return {
        "expense_category": suggested_category,
        "required_materials": _dedupe(required_materials),
        "risk_points": _dedupe(risk_points),
        "policy_references": _dedupe(policy_references),
        "rationale": " ".join(rationale_parts),
        "similar_case_refs": similar_case_refs,
        "rag_trace": rag_trace,
    }
