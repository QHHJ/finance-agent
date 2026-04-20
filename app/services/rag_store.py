from __future__ import annotations

from typing import Any

from app.retrieval.factory import get_retriever


def upsert_documents(
    *,
    source_type: str,
    source_id: str,
    documents: list[dict[str, Any]],
) -> int:
    return get_retriever().upsert_documents(
        source_type=source_type,
        source_id=source_id,
        documents=documents,
    )


def delete_documents(
    *,
    source_type: str | None = None,
    source_id: str | None = None,
    doc_key_prefix: str | None = None,
) -> int:
    return get_retriever().delete_documents(
        source_type=source_type,
        source_id=source_id,
        doc_key_prefix=doc_key_prefix,
    )


def query_documents(
    *,
    query: str,
    source_types: list[str],
    top_k: int = 6,
    min_score: float = 0.15,
    limit_scan: int = 5000,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return get_retriever().query_documents(
        query=query,
        source_types=source_types,
        top_k=top_k,
        min_score=min_score,
        limit_scan=limit_scan,
        metadata_filter=metadata_filter,
    )

