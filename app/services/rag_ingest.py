from __future__ import annotations

import hashlib
import re
from typing import Any

from . import rag_store


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> list[str]:
    normalized = str(text or "").replace("\r", "\n")
    paragraphs = [segment.strip() for segment in normalized.split("\n") if segment.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= chunk_chars:
            current = paragraph
            continue

        # Long paragraph hard split.
        step = max(64, chunk_chars - overlap_chars)
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_chars, len(paragraph))
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start += step
        current = ""

    if current:
        chunks.append(current)

    return [_clean_text(chunk) for chunk in chunks if _clean_text(chunk)]


def sync_policy_document(policy: Any) -> int:
    policy_id = str(getattr(policy, "id"))
    policy_name = str(getattr(policy, "name", "") or "policy").strip()
    raw_text = str(getattr(policy, "raw_text", "") or "")

    rag_store.delete_documents(source_type="policy", source_id=policy_id)

    chunks = chunk_text(raw_text)
    documents: list[dict[str, Any]] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
        doc_key = f"policy:{policy_id}:{idx}:{digest}"
        documents.append(
            {
                "doc_key": doc_key,
                "title": f"{policy_name}#{idx + 1}",
                "content": chunk,
                "metadata": {
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "chunk_index": idx,
                    "chunk_total": total,
                },
            }
        )

    if not documents:
        return 0
    return rag_store.upsert_documents(source_type="policy", source_id=policy_id, documents=documents)


def delete_policy_document(policy_id: int | str) -> int:
    return rag_store.delete_documents(source_type="policy", source_id=str(policy_id))
