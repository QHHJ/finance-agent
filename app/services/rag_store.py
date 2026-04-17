from __future__ import annotations

from typing import Any

from app.db import repo
from app.db.session import SessionLocal

from . import rag_embedder


def upsert_documents(
    *,
    source_type: str,
    source_id: str,
    documents: list[dict[str, Any]],
) -> int:
    if not documents:
        return 0

    contents = [str(doc.get("content") or "").strip() for doc in documents]
    vectors = rag_embedder.embed_texts(contents)
    if len(vectors) != len(documents):
        return 0

    db = SessionLocal()
    try:
        count = 0
        for doc, embedding in zip(documents, vectors):
            content = str(doc.get("content") or "").strip()
            if not content:
                continue
            doc_key = str(doc.get("doc_key") or "").strip()
            if not doc_key:
                continue
            doc_source_type = str(doc.get("source_type") or source_type)
            doc_source_id = str(doc.get("source_id") or source_id)

            repo.upsert_rag_document(
                db,
                source_type=doc_source_type,
                source_id=doc_source_id,
                doc_key=doc_key,
                title=str(doc.get("title") or "").strip() or None,
                content=content,
                metadata_json=dict(doc.get("metadata") or {}),
                embedding=embedding,
            )
            count += 1
        db.commit()
        return count
    finally:
        db.close()


def delete_documents(
    *,
    source_type: str | None = None,
    source_id: str | None = None,
    doc_key_prefix: str | None = None,
) -> int:
    db = SessionLocal()
    try:
        count = repo.delete_rag_documents(
            db,
            source_type=source_type,
            source_id=source_id,
            doc_key_prefix=doc_key_prefix,
        )
        db.commit()
        return count
    finally:
        db.close()


def _metadata_match(metadata_json: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if metadata_json.get(key) != expected:
            return False
    return True


def query_documents(
    *,
    query: str,
    source_types: list[str],
    top_k: int = 6,
    min_score: float = 0.15,
    limit_scan: int = 5000,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query_text = str(query or "").strip()
    if not query_text:
        return []

    db = SessionLocal()
    try:
        docs = repo.list_rag_documents(db, source_types=source_types, limit=limit_scan)
    finally:
        db.close()

    query_vec = rag_embedder.embed_text(query_text)
    if not query_vec:
        return []

    hits: list[dict[str, Any]] = []
    fallback_hits: list[dict[str, Any]] = []
    for doc in docs:
        metadata_json = dict(getattr(doc, "metadata_json", {}) or {})
        if not _metadata_match(metadata_json, metadata_filter):
            continue
        embedding = list(getattr(doc, "embedding", []) or [])
        score = rag_embedder.cosine_similarity(query_vec, embedding)
        item = {
            "id": doc.id,
            "score": score,
            "source_type": doc.source_type,
            "source_id": doc.source_id,
            "doc_key": doc.doc_key,
            "title": doc.title or "",
            "content": doc.content or "",
            "metadata": metadata_json,
        }
        fallback_hits.append(item)
        if score < min_score:
            continue
        hits.append(item)

    hits.sort(key=lambda item: item["score"], reverse=True)
    if hits:
        return hits[: max(1, top_k)]

    fallback_hits.sort(key=lambda item: item["score"], reverse=True)
    return fallback_hits[: max(1, top_k)]
