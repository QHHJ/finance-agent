from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any
from uuid import uuid4

GUIDE_FLOWS = {"travel", "material", "policy", "unknown"}


@dataclass(slots=True)
class GuideSessionState:
    session_id: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    user_goal: str = ""
    recommended_flow: str = "unknown"
    route_reason: str = ""
    uploaded_files: list[dict[str, Any]] = field(default_factory=list)
    precheck_result: dict[str, Any] = field(default_factory=dict)
    identified_doc_types: dict[str, int] = field(default_factory=dict)
    missing_items: list[str] = field(default_factory=list)
    is_ready_to_enter_flow: bool = False
    target_flow_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation_history": list(self.conversation_history),
            "user_goal": self.user_goal,
            "recommended_flow": self.recommended_flow,
            "route_reason": self.route_reason,
            "uploaded_files": list(self.uploaded_files),
            "precheck_result": dict(self.precheck_result),
            "identified_doc_types": dict(self.identified_doc_types),
            "missing_items": list(self.missing_items),
            "is_ready_to_enter_flow": bool(self.is_ready_to_enter_flow),
            "target_flow_payload": dict(self.target_flow_payload),
        }


def _welcome_text() -> str:
    return (
        "你好，我是首页引导助手。你可以先说要报销什么、手头有哪些材料，"
        "我会先帮你判断走差旅还是材料费，并告诉你对应要准备什么。"
    )


def new_guide_session() -> dict[str, Any]:
    state = GuideSessionState(session_id=uuid4().hex)
    state.conversation_history.append({"role": "assistant", "content": _welcome_text()})
    return state.to_dict()


def normalize_guide_session(state: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(state or {})
    session_id = str(source.get("session_id") or uuid4().hex)
    history = source.get("conversation_history")
    if not isinstance(history, list):
        history = []

    normalized_history: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            normalized_history.append({"role": role, "content": content})

    result = GuideSessionState(
        session_id=session_id,
        conversation_history=normalized_history,
        user_goal=str(source.get("user_goal") or "").strip(),
        recommended_flow=str(source.get("recommended_flow") or "unknown").strip(),
        route_reason=str(source.get("route_reason") or "").strip(),
        uploaded_files=[item for item in list(source.get("uploaded_files") or []) if isinstance(item, dict)],
        precheck_result=dict(source.get("precheck_result") or {}),
        identified_doc_types=dict(source.get("identified_doc_types") or {}),
        missing_items=[str(x) for x in list(source.get("missing_items") or []) if str(x).strip()],
        is_ready_to_enter_flow=bool(source.get("is_ready_to_enter_flow")),
        target_flow_payload=dict(source.get("target_flow_payload") or {}),
    )
    if result.recommended_flow not in GUIDE_FLOWS:
        result.recommended_flow = "unknown"
    if not result.conversation_history:
        result.conversation_history.append({"role": "assistant", "content": _welcome_text()})
    return result.to_dict()


def _keyword_score(text: str, keywords: list[str]) -> int:
    return sum(1 for key in keywords if key in text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _stable_pick(seed: str, options: list[str]) -> str:
    if not options:
        return ""
    value = sum(ord(ch) for ch in str(seed or ""))
    return options[value % len(options)]


def _infer_user_goal(existing_goal: str, user_message: str) -> str:
    text = _normalize_text(user_message)
    if not text:
        return str(existing_goal or "").strip()
    if _keyword_score(text, ["差旅", "出差", "机票", "酒店", "高铁", "火车", "打车", "travel", "trip"]) >= 1:
        return "差旅报销"
    if _keyword_score(text, ["材料费", "采购", "元器件", "入库", "清单", "发票明细", "material", "purchase"]) >= 1:
        return "材料费报销"
    if _keyword_score(text, ["制度", "政策", "规则", "报销标准", "财务规定", "能不能报", "policy", "rule"]) >= 1:
        return "制度咨询"
    return str(existing_goal or "").strip()


def _classify_uploaded_file(info: dict[str, Any]) -> tuple[str, str]:
    name = str(info.get("name") or "")
    suffix = str(info.get("suffix") or "").lower()
    text_preview = str(info.get("text_preview") or "")
    merged = _normalize_text(f"{name}\n{text_preview}")

    if not merged:
        return "unknown", "缺少文本线索"

    if _keyword_score(merged, ["制度", "规则", "办法", "报销标准", "流程说明", "faq"]) >= 2:
        return "policy_doc", "命中制度关键词"

    if any(key in merged for key in ["账单详情", "交易成功", "支付时间", "付款方式", "交易号", "支付凭证", "payment", "paid"]):
        if any(k in merged for k in ["酒店", "住宿", "携程酒店", "汉庭", "全季", "华住"]):
            return "hotel_payment", "命中酒店支付关键词"
        return "transport_payment", "命中交通支付关键词"

    if any(key in merged for key in ["价格明细", "机建", "燃油", "票价", "退改签", "行程单", "detail", "breakdown"]):
        return "flight_detail", "命中机票明细关键词"

    if any(key in merged for key in ["订单截图", "酒店订单", "几晚明细", "费用明细", "取消政策", "在线付", "hotelorder", "booking"]):
        return "hotel_order", "命中酒店订单关键词"

    if any(key in merged for key in ["电子发票", "发票号码", "价税合计", "税额", "购买方", "销售方", "invoice"]):
        if any(k in merged for k in ["航空", "机票", "客运服务", "代订机票费", "高铁", "铁路", "火车", "flight", "train"]):
            return "transport_ticket", "命中交通票据关键词"
        if any(k in merged for k in ["酒店", "住宿", "房费", "旅馆", "宾馆", "hotel"]):
            return "hotel_invoice", "命中酒店发票关键词"
        return "material_invoice", "命中发票关键词"

    if any(key in merged for key in ["元器件", "材料", "采购", "规格型号", "数量", "入库", "实验室", "material", "spec"]):
        return "material_support", "命中材料费关键词"

    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "unknown", "图片材料信息不足"
    if suffix == ".pdf":
        return "unknown", "PDF文本未命中关键词"
    return "unknown", "未识别到类型"


def _build_doc_type_stats(classified_files: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for item in classified_files:
        doc_type = str(item.get("doc_type") or "unknown")
        stats[doc_type] = stats.get(doc_type, 0) + 1
    return stats


def _recommend_flow(user_goal: str, user_message: str, stats: dict[str, int]) -> tuple[str, str]:
    goal_text = _normalize_text(user_goal)
    msg_text = _normalize_text(user_message)

    travel_score = 0
    material_score = 0
    policy_score = 0

    travel_score += _keyword_score(goal_text + msg_text, ["差旅", "出差", "机票", "酒店", "高铁", "打车", "travel", "trip"])
    material_score += _keyword_score(goal_text + msg_text, ["材料费", "采购", "元器件", "入库", "清单", "material", "purchase"])
    policy_score += _keyword_score(goal_text + msg_text, ["制度", "政策", "规则", "报销标准", "能不能报", "合规", "policy", "rule"])

    travel_score += (
        stats.get("transport_ticket", 0)
        + stats.get("transport_payment", 0)
        + stats.get("flight_detail", 0)
        + stats.get("hotel_invoice", 0)
        + stats.get("hotel_payment", 0)
        + stats.get("hotel_order", 0)
    )
    material_score += stats.get("material_invoice", 0) + stats.get("material_support", 0)
    policy_score += stats.get("policy_doc", 0)

    if policy_score >= max(travel_score, material_score) and policy_score > 0:
        return "policy", "对话和材料更偏制度咨询"
    if travel_score >= max(material_score, policy_score) and travel_score > 0:
        return "travel", "识别到较多差旅相关线索"
    if material_score >= max(travel_score, policy_score) and material_score > 0:
        return "material", "识别到较多材料费相关线索"
    return "unknown", "线索不足，暂时无法稳定判定流程"


def _missing_items_for_flow(flow: str, stats: dict[str, int]) -> list[str]:
    # Home guide only routes to flow type. Completeness checks belong to the formal flow page.
    _ = (flow, stats)
    return []


def _is_ready_for_flow(flow: str, stats: dict[str, int]) -> bool:
    # As long as flow is recognized, home guide can hand off.
    _ = stats
    return flow in {"travel", "material", "policy"}


def _build_target_payload(*, state: dict[str, Any], classified_files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "recommended_flow": state.get("recommended_flow"),
        "route_reason": state.get("route_reason"),
        "user_goal": state.get("user_goal"),
        "missing_items": list(state.get("missing_items") or []),
        "identified_doc_types": dict(state.get("identified_doc_types") or {}),
        "precheck_result": dict(state.get("precheck_result") or {}),
        "guide_summary": {
            "uploaded_count": len(classified_files),
            "ready": bool(state.get("is_ready_to_enter_flow")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def _flow_label(flow: str) -> str:
    return {
        "travel": "差旅",
        "material": "材料费",
        "policy": "制度咨询",
        "unknown": "未确定",
    }.get(str(flow or "unknown"), "未确定")


def _focus_type(user_message: str) -> str:
    text = _normalize_text(user_message)
    if not text:
        return "upload_only"
    if any(k in text for k in ["需要什么材料", "要什么材料", "准备什么材料", "材料清单", "报销要求", "要准备什么"]):
        return "requirements"
    if any(k in text for k in ["进入流程", "直接进入", "开始处理", "开始报销", "走差旅", "走材料费"]):
        return "enter_flow"
    if any(k in text for k in ["为什么", "依据", "怎么判断", "凭什么"]):
        return "reason"
    if any(k in text for k in ["再看", "看看这个", "不太对", "你看一下"]):
        return "review"
    return "general"


def _top_doc_type_hint(stats: dict[str, int]) -> str:
    labels = {
        "transport_ticket": "交通票据",
        "transport_payment": "交通支付记录",
        "flight_detail": "机票明细",
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "material_invoice": "材料费发票",
        "material_support": "材料费附件",
        "policy_doc": "制度文档",
    }
    pairs: list[tuple[str, int]] = []
    for key, count in stats.items():
        if key == "unknown":
            continue
        count_num = int(count or 0)
        if count_num <= 0:
            continue
        pairs.append((labels.get(key, key), count_num))
    pairs.sort(key=lambda item: (-item[1], item[0]))
    if not pairs:
        return "暂未识别出稳定材料类型"
    return "、".join(f"{name}{count}份" for name, count in pairs[:3])


def _infer_target_flow_for_requirements(user_message: str, fallback_flow: str) -> str:
    text = _normalize_text(user_message)
    if any(k in text for k in ["差旅", "出差", "机票", "酒店", "高铁", "火车"]):
        return "travel"
    if any(k in text for k in ["材料费", "采购", "元器件", "入库", "材料"]):
        return "material"
    if fallback_flow in {"travel", "material"}:
        return fallback_flow
    return "unknown"


def _travel_requirements_text() -> str:
    return (
        "差旅常用材料清单：\n"
        "- 去程交通：交通票据（机票发票/高铁报销凭证）+ 支付记录；若飞机还要机票明细\n"
        "- 返程交通：同去程\n"
        "- 酒店：酒店发票 + 酒店支付记录 + 酒店平台订单截图\n"
        "- 特殊情况：在出差地不住宿需补“情况说明”\n"
        "你可以先上传一部分材料，我先做分流，再带你进入正式流程继续整理。"
    )


def _material_requirements_text() -> str:
    return (
        "材料费常用材料清单：\n"
        "- 材料费发票（抬头、税号、金额等可识别）\n"
        "- 材料明细/入库相关信息（项目名称、规格型号、数量、单位、每项含税总价）\n"
        "- 支付记录（微信/支付宝/银行）\n"
        "- 采购单/合同/送货单（如有）\n"
        "你先上传发票也可以，我会先抽取并生成明细，后续再补材料。"
    )


def _flow_requirements_hint(flow: str) -> str:
    if flow == "travel":
        return "差旅关键材料：去/返程票据与支付记录，飞机场景补机票明细，酒店补发票+支付+订单截图。"
    if flow == "material":
        return "材料费关键材料：发票 + 明细/入库信息 + 支付记录（有采购单据可一并上传）。"
    if flow == "policy":
        return "制度咨询场景建议上传制度PDF或直接提问具体条款。"
    return "你可以先告诉我是差旅还是材料费，我会给对应清单。"


def _compose_reply(state: dict[str, Any], user_message: str) -> str:
    flow = str(state.get("recommended_flow") or "unknown")
    reason = str(state.get("route_reason") or "已完成初步分流判断")
    uploaded_count = len(list(state.get("uploaded_files") or []))
    stats = dict(state.get("identified_doc_types") or {})
    focus = _focus_type(user_message)
    seed = f"{user_message}|{flow}|{uploaded_count}|{focus}"
    asks_enter_flow = focus == "enter_flow"

    if focus == "requirements":
        target_flow = _infer_target_flow_for_requirements(user_message, flow)
        if target_flow == "travel":
            first = _stable_pick(seed, ["可以，我先给你差旅的材料要求。", "好，我先把差旅报销要准备的材料列给你。", "明白，先看差旅材料清单。"])
            second = _travel_requirements_text()
            third = (
                f"当前已纳入 {uploaded_count} 份材料，识别线索：{_top_doc_type_hint(stats)}。"
                "你准备好后我会帮你自动进入差旅正式流程。"
            )
            return f"{first}\n\n{second}\n\n{third}"
        if target_flow == "material":
            first = _stable_pick(seed, ["可以，我先给你材料费的准备清单。", "好，我先把材料费报销要准备的材料列给你。", "明白，先看材料费清单。"])
            second = _material_requirements_text()
            third = (
                f"当前已纳入 {uploaded_count} 份材料，识别线索：{_top_doc_type_hint(stats)}。"
                "你也可以直接上传发票，我先帮你进入材料费流程。"
            )
            return f"{first}\n\n{second}\n\n{third}"
        first = _stable_pick(seed, ["我先给你一版通用清单。", "我先按两个高频场景给你材料清单。", "先给你差旅和材料费两类常用要求。"])
        second = _travel_requirements_text() + "\n\n" + _material_requirements_text()
        third = "你告诉我“这次是差旅”或“这次是材料费”，我会立刻收敛到对应流程。"
        return f"{first}\n\n{second}\n\n{third}"

    if flow == "unknown":
        first = _stable_pick(
            seed,
            [
                "我先看过你当前给的信息了。",
                "收到，我先做了一轮首页分流判断。",
                "我先把这批材料做了轻量预判。",
            ],
        )
        if uploaded_count <= 0:
            second = (
                "目前还没有材料可供预检查，但我可以先把流程和材料要求讲清楚。"
                "你可以直接说“我要差旅报销”或“我要材料费报销”。"
            )
            third = "如果你愿意，我现在就可以先给你对应清单；也可以先上传 1-2 份票据让我判断。"
        else:
            second = (
                f"目前还不能稳定判定是差旅还是材料费（{reason}）。"
                f"现在已纳入 {uploaded_count} 份材料，识别线索：{_top_doc_type_hint(stats)}。"
            )
            third = "你补一句“这是差旅报销”或“这是材料费报销”，我就能直接给出分流并进入正式流程。"
        return f"{first}\n\n{second}\n\n{third}"

    if focus in {"reason", "review"}:
        first = _stable_pick(seed, ["可以，我说下判断依据。", "我来解释这次分流为什么这样判。", "我把当前分流依据展开给你。"])
        second = f"当前推荐流程：{_flow_label(flow)}。判断依据：{reason}。已识别线索：{_top_doc_type_hint(stats)}。"
        third = f"如果你愿意，我可以继续把该流程材料清单列出来。{_flow_requirements_hint(flow)}"
        return f"{first}\n\n{second}\n\n{third}"

    if flow == "travel":
        first = _stable_pick(seed, ["明白，这批更像差旅报销。", "看起来这是差旅场景。", "我先把它归到差旅方向了。"])
        second = f"已纳入 {uploaded_count} 份材料，判断依据：{reason}。"
        if uploaded_count <= 0:
            third = _travel_requirements_text()
        elif asks_enter_flow:
            third = "可以，信息已足够分流。我会把当前材料一起带入差旅正式流程。"
        else:
            third = "你可以继续问“差旅还要准备什么材料”，或说“直接进入差旅流程”。"
        return f"{first}\n\n{second}\n\n{third}"

    if flow == "material":
        first = _stable_pick(seed, ["明白，这批更像材料费报销。", "看起来这是材料费场景。", "我先把它归到材料费方向了。"])
        second = f"已纳入 {uploaded_count} 份材料，判断依据：{reason}。"
        if uploaded_count <= 0:
            third = _material_requirements_text()
        elif asks_enter_flow:
            third = "可以，信息已足够分流。我会把当前材料一起带入材料费正式流程。"
        else:
            third = "你可以继续问“材料费还需要什么”，或说“直接进入材料费流程”。"
        return f"{first}\n\n{second}\n\n{third}"

    first = _stable_pick(seed, ["明白，你更偏向制度咨询。", "这个场景更像制度问答。", "我先按制度咨询来处理。"])
    second = f"当前判断依据：{reason}。"
    third = "如果你要开始票据处理，告诉我“走差旅”或“走材料费”，我会立刻分流。"
    return f"{first}\n\n{second}\n\n{third}"


def _dedupe_reply(state: dict[str, Any], reply: str) -> str:
    history = list(state.get("conversation_history") or [])
    last_assistant = ""
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") == "assistant":
            last_assistant = str(item.get("content") or "").strip()
            if last_assistant:
                break

    if not last_assistant:
        return reply

    now_norm = re.sub(r"\s+", "", str(reply or ""))
    last_norm = re.sub(r"\s+", "", last_assistant)
    if not now_norm or now_norm != last_norm:
        return reply

    flow = _flow_label(str(state.get("recommended_flow") or "unknown"))
    seed = f"{flow}|{len(list(state.get('uploaded_files') or []))}|{len(list(state.get('conversation_history') or []))}"
    tail = _stable_pick(
        seed,
        [
            "你可以继续告诉我这次报销目标，我会按这个方向接着整理。",
            "如果你愿意，我可以现在直接给出该流程的材料清单。",
            "你也可以直接说“进入差旅流程”或“进入材料费流程”。",
        ],
    )
    return (
        f"我这边复核后结论不变：当前仍推荐“{flow}”流程。\n\n" +
        tail
    )


def _build_precheck_result(files: list[dict[str, Any]]) -> dict[str, Any]:
    classified_files: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        doc_type, reason = _classify_uploaded_file(item)
        classified_files.append(
            {
                "name": str(item.get("name") or ""),
                "size": int(item.get("size") or 0),
                "suffix": str(item.get("suffix") or ""),
                "doc_type": doc_type,
                "reason": reason,
            }
        )
    return {
        "classified_files": classified_files,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def process_guide_turn(
    state: dict[str, Any] | None,
    *,
    user_message: str,
    uploaded_files: list[dict[str, Any]] | None = None,
    record_history: bool = True,
) -> tuple[dict[str, Any], str]:
    current = normalize_guide_session(state)
    text = str(user_message or "").strip()
    files = [item for item in list(uploaded_files or []) if isinstance(item, dict)]

    current["uploaded_files"] = files
    current["user_goal"] = _infer_user_goal(current.get("user_goal", ""), text)
    current["precheck_result"] = _build_precheck_result(files)
    classified_files = list(current["precheck_result"].get("classified_files") or [])
    stats = _build_doc_type_stats(classified_files)
    current["identified_doc_types"] = stats

    recommended_flow, route_reason = _recommend_flow(current.get("user_goal", ""), text, stats)
    current["recommended_flow"] = recommended_flow
    current["route_reason"] = route_reason

    missing_items = _missing_items_for_flow(recommended_flow, stats)
    current["missing_items"] = missing_items
    current["is_ready_to_enter_flow"] = _is_ready_for_flow(recommended_flow, stats)

    current["target_flow_payload"] = _build_target_payload(state=current, classified_files=classified_files)

    reply = _compose_reply(current, text)
    reply = _dedupe_reply(current, reply)

    if record_history:
        if text:
            current["conversation_history"].append({"role": "user", "content": text})
        current["conversation_history"].append({"role": "assistant", "content": reply})
    return current, reply
