from __future__ import annotations

from typing import Any, Protocol


class Retriever(Protocol):
    def upsert_documents(
        self,
        *,
        source_type: str,
        source_id: str,
        documents: list[dict[str, Any]],
    ) -> int:
        ...

    def delete_documents(
        self,
        *,
        source_type: str | None = None,
        source_id: str | None = None,
        doc_key_prefix: str | None = None,
    ) -> int:
        ...

    def query_documents(
        self,
        *,
        query: str,
        source_types: list[str],
        top_k: int = 6,
        min_score: float = 0.15,
        limit_scan: int = 5000,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...

