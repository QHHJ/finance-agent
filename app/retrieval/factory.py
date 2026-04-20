from __future__ import annotations

import os

from app.retrieval.base import Retriever
from app.retrieval.faiss_retriever import FaissRetriever
from app.retrieval.sqlite_retriever import SQLiteRetriever

_CACHE: dict[str, Retriever] = {}


def get_retriever(backend: str | None = None) -> Retriever:
    selected = str(backend or os.getenv("RAG_BACKEND", "sqlite")).strip().lower()
    if selected not in {"sqlite", "faiss"}:
        selected = "sqlite"

    cached = _CACHE.get(selected)
    if cached is not None:
        return cached

    if selected == "faiss":
        retriever: Retriever = FaissRetriever()
    else:
        retriever = SQLiteRetriever()
    _CACHE[selected] = retriever
    return retriever


def rebuild_faiss_index() -> int:
    retriever = FaissRetriever()
    return retriever.rebuild_index()


def clear_retriever_cache() -> None:
    _CACHE.clear()

