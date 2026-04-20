from __future__ import annotations

from app.retrieval.factory import rebuild_faiss_index


def rebuild_faiss() -> int:
    """Rebuild FAISS index from SQLite-stored RAG documents."""
    return rebuild_faiss_index()


if __name__ == "__main__":
    count = rebuild_faiss()
    print(f"FAISS index rebuilt, indexed documents: {count}")
