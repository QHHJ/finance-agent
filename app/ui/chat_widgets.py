from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import streamlit as st


DEFAULT_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
TRUE_VALUES = {"1", "true", "yes", "on"}


def compose_three_stage_reply(understand: str, status_change: str, next_step: str) -> str:
    part1 = str(understand or "好的，我理解了。").strip()
    part2 = str(status_change or "当前状态保持不变。").strip()
    part3 = str(next_step or "你可以继续告诉我希望我怎么处理。").strip()
    return f"{part1}\n\n{part2}\n\n{part3}"


def _as_uploaded_list(uploaded_value: Any) -> list[Any]:
    if not uploaded_value:
        return []
    if isinstance(uploaded_value, (list, tuple)):
        return list(uploaded_value)
    return [uploaded_value]


def _chat_typewriter_enabled() -> bool:
    # Default on for conversational UX; can disable with CHAT_TYPEWRITER=0.
    raw = str(os.getenv("CHAT_TYPEWRITER", "1")).strip().lower()
    return raw in TRUE_VALUES


def _iter_typewriter_chunks(text: str):
    source = str(text or "")
    if not source:
        return
    length = len(source)
    if length >= 1000:
        chunk_size, pause = 18, 0.002
    elif length >= 600:
        chunk_size, pause = 12, 0.004
    elif length >= 260:
        chunk_size, pause = 6, 0.008
    else:
        chunk_size, pause = 2, 0.012

    for idx in range(0, length, chunk_size):
        yield source[idx : idx + chunk_size]
        if pause > 0:
            time.sleep(pause)


def _render_typewriter_markdown(content: str) -> None:
    holder = st.empty()
    rendered = ""
    for chunk in _iter_typewriter_chunks(content):
        rendered += chunk
        holder.markdown(rendered + "▌")
    holder.markdown(rendered)


def render_chat_messages(messages: list[dict[str, Any]], *, stream_state_key: str) -> None:
    latest_assistant_idx: int | None = None
    latest_assistant_text = ""
    for idx in range(len(messages) - 1, -1, -1):
        item = messages[idx]
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "assistant") == "assistant":
            latest_assistant_idx = idx
            latest_assistant_text = str(item.get("content") or "")
            break

    signature = ""
    if latest_assistant_idx is not None:
        digest = hashlib.sha1(latest_assistant_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        signature = f"{len(messages)}:{latest_assistant_idx}:{digest}"

    last_streamed = str(st.session_state.get(stream_state_key) or "")
    should_stream = (
        _chat_typewriter_enabled()
        and latest_assistant_idx is not None
        and signature
        and signature != last_streamed
        and any(str((m or {}).get("role") or "") == "user" for m in messages)
    )

    for idx, message in enumerate(messages):
        role = str((message or {}).get("role") or "assistant")
        content = str((message or {}).get("content") or "")
        with st.chat_message(role):
            if should_stream and idx == latest_assistant_idx and content:
                _render_typewriter_markdown(content)
                st.session_state[stream_state_key] = signature
            else:
                st.markdown(content)


def _extract_chat_composer_submission(raw_value: Any) -> tuple[str, list[Any]]:
    if raw_value is None:
        return "", []
    text_value = getattr(raw_value, "text", raw_value)
    file_value = getattr(raw_value, "files", [])
    return str(text_value or "").strip(), _as_uploaded_list(file_value)


def travel_chat_input_with_files(*, key: str, upload_types: list[str] | None = None) -> tuple[str, list[Any]]:
    try:
        raw_value = st.chat_input(
            "直接提问，或把差旅材料拖到这里（支持多文件）",
            key=key,
            accept_file="multiple",
            file_type=upload_types or DEFAULT_UPLOAD_TYPES,
            max_upload_size=200,
        )
        return _extract_chat_composer_submission(raw_value)
    except TypeError:
        # Older Streamlit versions do not support chat_input file attachments.
        raw_text = st.chat_input("例如：我现在还缺什么？", key=key)
        return str(raw_text or "").strip(), []


def inject_ui_styles() -> None:
    st.markdown(
        """
<style>
/* Chat input: make boundary obvious and consistent */
div[data-testid="stChatInput"] {
  border: 1.5px solid #c7d2fe !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06) !important;
  padding: 4px 8px !important;
  margin-top: 6px !important;
}

div[data-testid="stChatInput"]:focus-within {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16) !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
  background: #ffffff !important;
  color: #0f172a !important;
}

div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] input::placeholder {
  color: #6b7280 !important;
  opacity: 1 !important;
}

/* Generic input field visibility (for fallback text boxes) */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  background: #ffffff !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
}

.travel-workbench-anchor,
.travel-side-panel-anchor {
  display: none;
}

body:has(.travel-workbench-anchor) .block-container {
  padding-bottom: 120px !important;
}

body:has(.travel-workbench-anchor) div[data-testid="stChatInput"] > div {
  background: #ffffff !important;
}

div[data-testid="column"]:has(.travel-side-panel-anchor) {
  position: sticky !important;
  top: 14px !important;
  align-self: flex-start !important;
  max-height: calc(100vh - 28px) !important;
  overflow-y: auto !important;
  padding-right: 4px !important;
}

.travel-preview-table {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
}

.travel-preview-head {
  color: #64748b;
  font-size: 0.82rem;
  font-weight: 650;
  padding: 8px 0;
}

.travel-preview-row {
  border-top: 1px solid #eef2f7;
  padding: 6px 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )
