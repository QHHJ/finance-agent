from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TravelChatIntent(str, Enum):
    ASK_MISSING = "ask_missing"
    ASK_FILE_LIST = "ask_file_list"
    ASK_FILE_COUNT = "ask_file_count"
    ASK_MISMATCH = "ask_mismatch"
    ASK_NEXT_STEP = "ask_next_step"
    ASK_REQUIREMENT = "ask_requirement"
    RECLASSIFY_FILE = "reclassify_file"
    BIND_SLOT = "bind_slot"
    ASK_REASON = "ask_reason"
    CLARIFY = "clarify"


class TravelChatQuery(BaseModel):
    intent: TravelChatIntent
    target_slot: Optional[str] = None
    group_name: Optional[str] = None
    filename: Optional[str] = None
    need_clarify: bool = False
    clarify_question: Optional[str] = None


def parse_travel_chat_query(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> TravelChatQuery:
    """
    第一版先走 rule-first，后面再接 LLM fallback。
    这里不要直接生成最终回答，只做问句归一化。
    """
    text = (user_message or "").strip().lower()

    # ===== 缺失项 =====
    if any(x in user_message for x in ["还缺什么", "缺什么", "少什么", "差什么"]):
        return TravelChatQuery(intent=TravelChatIntent.ASK_MISSING)

    # ===== 下一步 =====
    if any(x in user_message for x in ["下一步", "现在该做什么", "接下来怎么办", "怎么弄"]):
        return TravelChatQuery(intent=TravelChatIntent.ASK_NEXT_STEP)

    # ===== 金额/核对问题 =====
    if any(x in user_message for x in ["对不上", "不一致", "核对问题", "金额问题", "哪个和哪个"]):
        return TravelChatQuery(intent=TravelChatIntent.ASK_MISMATCH)

    # ===== 酒店材料要求 =====
    if "酒店" in user_message and any(x in user_message for x in ["材料", "需要什么", "要什么"]):
        return TravelChatQuery(
            intent=TravelChatIntent.ASK_REQUIREMENT,
            group_name="hotel",
        )

    # ===== 文件数量 / 文件列表 =====
    hotel_words = ["酒店发票", "酒店票据", "住宿发票"]
    hotel_payment_words = ["酒店支付", "支付记录", "付款记录"]
    hotel_order_words = ["酒店订单", "订单截图", "预订截图"]

    if any(w in user_message for w in hotel_words):
        if any(x in user_message for x in ["几个", "几份", "几张", "数量"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_COUNT,
                target_slot="hotel_invoice",
            )
        if any(x in user_message for x in ["哪些", "哪几个", "列出来", "文件名", "具体是"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_LIST,
                target_slot="hotel_invoice",
            )

    if any(w in user_message for w in hotel_payment_words):
        if any(x in user_message for x in ["几个", "几份", "几张", "数量"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_COUNT,
                target_slot="hotel_payment",
            )
        if any(x in user_message for x in ["哪些", "哪几个", "列出来", "文件名", "具体是"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_LIST,
                target_slot="hotel_payment",
            )

    if any(w in user_message for w in hotel_order_words):
        if any(x in user_message for x in ["几个", "几份", "几张", "数量"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_COUNT,
                target_slot="hotel_order",
            )
        if any(x in user_message for x in ["哪些", "哪几个", "列出来", "文件名", "具体是"]):
            return TravelChatQuery(
                intent=TravelChatIntent.ASK_FILE_LIST,
                target_slot="hotel_order",
            )

    # ===== 兜底澄清 =====
    return TravelChatQuery(
        intent=TravelChatIntent.CLARIFY,
        need_clarify=True,
        clarify_question="你是想看缺失项、金额核对问题，还是某类材料对应的文件列表？",
    )


def execute_travel_chat_query(
    query: TravelChatQuery,
    assignment: Dict[str, Any],
    status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    这里完全不要让模型猜，直接查当前任务状态。
    """
    if query.intent == TravelChatIntent.ASK_MISSING:
        missing = status.get("missing", []) or []
        return {
            "answer_type": "missing_list",
            "items": missing,
        }

    if query.intent == TravelChatIntent.ASK_FILE_LIST:
        files = _get_slot_files(assignment, query.target_slot)
        return {
            "answer_type": "file_list",
            "label": _slot_label(query.target_slot),
            "count": len(files),
            "files": files,
        }

    if query.intent == TravelChatIntent.ASK_FILE_COUNT:
        files = _get_slot_files(assignment, query.target_slot)
        return {
            "answer_type": "file_count",
            "label": _slot_label(query.target_slot),
            "count": len(files),
        }

    if query.intent == TravelChatIntent.ASK_MISMATCH:
        issues = status.get("issues", []) or []
        return {
            "answer_type": "issue_list",
            "issues": issues,
        }

    if query.intent == TravelChatIntent.ASK_NEXT_STEP:
        missing = status.get("missing", []) or []
        issues = status.get("issues", []) or []
        return build_next_step_payload(missing, issues)

    if query.intent == TravelChatIntent.ASK_REQUIREMENT:
        if query.group_name == "hotel":
            return {
                "answer_type": "requirement_list",
                "title": "酒店相关材料",
                "items": [
                    "酒店发票",
                    "酒店支付记录",
                    "酒店订单截图",
                ],
            }

    if query.intent == TravelChatIntent.CLARIFY:
        return {
            "answer_type": "clarify",
            "text": query.clarify_question or "我先确认一下你的问题。",
        }

    return {
        "answer_type": "clarify",
        "text": "我先确认一下，你是想看材料列表、缺失项还是金额核对问题？",
    }


def render_travel_chat_answer(payload: Dict[str, Any]) -> str:
    answer_type = payload.get("answer_type")

    if answer_type == "missing_list":
        items = payload.get("items", []) or []
        if not items:
            return "当前必需材料已经补齐。"
        return "当前还缺这些材料：\n" + "\n".join(f"- {x}" for x in items)

    if answer_type == "file_count":
        return f'{payload.get("label", "该类材料")}目前识别到 {payload.get("count", 0)} 份。'

    if answer_type == "file_list":
        files = payload.get("files", []) or []
        label = payload.get("label", "该类材料")
        if not files:
            return f"当前还没有识别到{label}。"
        return f"{label}目前识别到 {len(files)} 份：\n" + "\n".join(f"- {x}" for x in files)

    if answer_type == "issue_list":
        issues = payload.get("issues", []) or []
        if not issues:
            return "当前没有发现明确的金额核对问题。"
        return "目前发现这些核对问题：\n" + "\n".join(f"- {x}" for x in issues)

    if answer_type == "requirement_list":
        items = payload.get("items", []) or []
        title = payload.get("title", "所需材料")
        return f"{title}一般包括：\n" + "\n".join(f"- {x}" for x in items)

    if answer_type == "next_step":
        return payload.get("text", "我先帮你继续整理。")

    return payload.get("text", "我先帮你继续整理。")


def build_next_step_payload(missing: List[str], issues: List[str]) -> Dict[str, Any]:
    if missing:
        return {
            "answer_type": "next_step",
            "text": "你现在优先补齐这些材料：\n"
            + "\n".join(f"- {x}" for x in missing)
        }
    if issues:
        return {
            "answer_type": "next_step",
            "text": "你现在优先处理这些金额核对问题：\n"
            + "\n".join(f"- {x}" for x in issues)
        }
    return {
        "answer_type": "next_step",
        "text": "当前材料基本齐了，可以继续检查细项，或者直接进入确认/导出。",
    }


def _get_slot_files(assignment: Dict[str, Any], slot: Optional[str]) -> List[str]:
    """
    这里要按你项目里 assignment 的真实结构微调。
    先兼容 list[dict] / list[str] 两种情况。
    """
    if not slot:
        return []

    raw = assignment.get(slot) or []
    result: List[str] = []

    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("filename")
                or item.get("file_name")
                or item.get("path")
                or "未命名文件"
            )
            result.append(name)

    return result


def _slot_label(slot: Optional[str]) -> str:
    mapping = {
        "hotel_invoice": "酒店发票",
        "hotel_payment": "酒店支付记录",
        "hotel_order": "酒店订单截图",
        "go_ticket": "去程票据",
        "return_ticket": "返程票据",
    }
    return mapping.get(slot or "", "该类材料")