from __future__ import annotations

import os
import re
from typing import Any

from . import rag_store


def _top_k(default_value: int, env_key: str) -> int:
    raw = os.getenv(env_key)
    if raw is None:
        return default_value
    try:
        return max(1, int(raw))
    except ValueError:
        return default_value


def _build_material_query(extracted_data: dict[str, Any], raw_text: str) -> str:
    parts = [
        str(extracted_data.get("bill_type") or ""),
        str(extracted_data.get("item_content") or ""),
        str(extracted_data.get("seller") or ""),
        str(extracted_data.get("buyer") or ""),
        str(extracted_data.get("amount") or ""),
        str(raw_text or "")[:1200],
    ]
    return "\n".join(part for part in parts if part).strip()


def _shorten(text: str, max_len: int = 220) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def retrieve_policy_hits(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    return rag_store.query_documents(
        query=query,
        source_types=["policy"],
        top_k=top_k or _top_k(4, "RAG_POLICY_TOP_K"),
        min_score=float(os.getenv("RAG_POLICY_MIN_SCORE", "0.12")),
    )


def retrieve_material_case_hits(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    return rag_store.query_documents(
        query=query,
        source_types=["material_case"],
        top_k=top_k or _top_k(4, "RAG_MATERIAL_CASE_TOP_K"),
        min_score=float(os.getenv("RAG_CASE_MIN_SCORE", "0.16")),
    )


def retrieve_material_fix_case_hits(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    k = top_k or _top_k(4, "RAG_MATERIAL_FIX_CASE_TOP_K")
    hits = rag_store.query_documents(
        query=query,
        source_types=["material_fix_case"],
        top_k=k,
        min_score=float(os.getenv("RAG_CASE_MIN_SCORE", "0.16")),
    )
    if hits:
        return hits
    # Fallback to coarse material cases when row-level fix samples are still sparse.
    return retrieve_material_case_hits(query=query, top_k=k)


def retrieve_travel_case_hits(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    return rag_store.query_documents(
        query=query,
        source_types=["travel_case"],
        top_k=top_k or _top_k(4, "RAG_TRAVEL_CASE_TOP_K"),
        min_score=float(os.getenv("RAG_CASE_MIN_SCORE", "0.16")),
    )


def build_material_references(
    extracted_data: dict[str, Any],
    raw_text: str,
) -> dict[str, Any]:
    query = _build_material_query(extracted_data, raw_text)
    policy_hits = retrieve_policy_hits(query)
    case_hits = retrieve_material_case_hits(query)

    policy_refs: list[str] = []
    for hit in policy_hits:
        meta = dict(hit.get("metadata") or {})
        title = str(meta.get("policy_name") or hit.get("title") or "policy")
        policy_refs.append(f"{title}: {_shorten(hit.get('content', ''), max_len=96)}")

    return {
        "query": query,
        "policy_hits": policy_hits,
        "case_hits": case_hits,
        "policy_refs": policy_refs,
    }


def build_travel_policy_context(raw_text: str, top_k: int = 3) -> str:
    query = str(raw_text or "").strip()
    if not query:
        return ""

    policy_hits = retrieve_policy_hits(query, top_k=top_k)
    travel_hits = retrieve_travel_case_hits(query, top_k=max(2, top_k))

    blocks: list[str] = []
    for idx, hit in enumerate(policy_hits, start=1):
        meta = dict(hit.get("metadata") or {})
        name = str(meta.get("policy_name") or hit.get("title") or "policy")
        blocks.append(f"[Policy#{idx} score={hit['score']:.3f} name={name}] {_shorten(hit.get('content', ''), max_len=260)}")

    for idx, hit in enumerate(travel_hits, start=1):
        blocks.append(f"[TravelCase#{idx} score={hit['score']:.3f}] {_shorten(hit.get('content', ''), max_len=220)}")

    return "\n".join(blocks)
