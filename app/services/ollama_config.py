from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag_true(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in TRUE_VALUES


def env_int_value(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def env_float_value(name: str, default: float) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def vl_model() -> str:
    return os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")


def text_model() -> str:
    return os.getenv("OLLAMA_TEXT_MODEL") or os.getenv("OLLAMA_CHAT_MODEL") or vl_model()


def travel_doc_text_model() -> str:
    return (
        os.getenv("OLLAMA_TRAVEL_DOC_TEXT_MODEL")
        or os.getenv("OLLAMA_TEXT_MODEL")
        or os.getenv("OLLAMA_CHAT_MODEL")
        or "qwen2.5:7b-instruct"
    )


@st.cache_data(show_spinner=False, ttl=60)
def list_ollama_model_names(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{str(base_url or '').rstrip('/')}/api/tags", timeout=(4, 10))
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def select_available_model(preferred: str, fallbacks: list[str], available: list[str]) -> str:
    preferred_name = str(preferred or "").strip()
    if not available:
        return preferred_name
    if preferred_name in available:
        return preferred_name
    for candidate in fallbacks:
        name = str(candidate or "").strip()
        if name and name in available:
            return name
    return available[0]


def chat_model() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    preferred = os.getenv("OLLAMA_CHAT_MODEL") or text_model()
    available = list_ollama_model_names(base_url)
    fallback_order = [
        os.getenv("OLLAMA_TEXT_MODEL"),
        "qwen2.5:7b-instruct",
        "qwen2.5vl:7b",
        vl_model(),
    ]
    return select_available_model(preferred, fallback_order, available)


def current_model_config() -> dict[str, Any]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    chat_model_env = os.getenv("OLLAMA_CHAT_MODEL") or ""
    return {
        "use_ollama_vl": env_flag_true("USE_OLLAMA_VL"),
        "base_url": base_url,
        "vl_model": vl_model(),
        "text_model": text_model(),
        "travel_doc_text_model": travel_doc_text_model(),
        "chat_model_env": chat_model_env,
        "chat_model": chat_model(),
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    }


def get_ollama_runtime_rows(base_url: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=(4, 12))
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return [], str(exc)

    rows: list[dict[str, Any]] = []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return rows, None

    for item in models:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "模型": str(item.get("name") or ""),
                "处理器占用": str(item.get("details", {}).get("processor") or item.get("processor") or ""),
                "上下文": str(item.get("details", {}).get("context_length") or item.get("context") or ""),
                "驻留到": str(item.get("expires_at") or item.get("until") or ""),
                "大小": str(item.get("size") or ""),
            }
        )
    return rows, None


def render_model_runtime_panel() -> None:
    cfg = current_model_config()
    with st.expander("当前模型与运行状态", expanded=False):
        st.markdown(f"- 视觉抽取模型：`{cfg['vl_model']}`")
        st.markdown(f"- 文本抽取/修复模型：`{cfg['text_model']}`")
        st.markdown(f"- 差旅 doc_type 分类模型：`{cfg['travel_doc_text_model']}`")
        st.markdown(f"- 对话模型（已解析）：`{cfg['chat_model']}`")
        if str(cfg.get("chat_model_env") or "").strip():
            st.markdown(f"- 对话模型（环境变量）：`{cfg['chat_model_env']}`")
        st.markdown(f"- 向量模型：`{cfg['embed_model']}`")
        st.markdown(f"- Ollama 地址：`{cfg['base_url']}`")
        st.markdown(f"- 启用视觉抽取：`{cfg['use_ollama_vl']}`")

        rows, err = get_ollama_runtime_rows(cfg["base_url"])
        if err:
            st.caption(f"未能获取运行中模型状态：{err}")
            return
        if not rows:
            st.caption("当前没有模型驻留（或 Ollama 暂无活跃会话）。")
            return
        st.dataframe(rows, use_container_width=True, hide_index=True)
