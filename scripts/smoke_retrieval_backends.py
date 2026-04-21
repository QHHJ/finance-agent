from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _smoke_backend(backend: str) -> dict:
    from app.retrieval.factory import clear_retriever_cache, get_retriever, rebuild_faiss_index

    clear_retriever_cache()
    if backend == "faiss":
        rebuild_faiss_index()

    retriever = get_retriever(backend)
    source_id = f"smoke_{backend}_{uuid4().hex[:8]}"
    doc_key = f"travel_case:{source_id}:1"

    inserted = retriever.upsert_documents(
        source_type="travel_case",
        source_id=source_id,
        documents=[
            {
                "doc_key": doc_key,
                "title": "smoke_case",
                "content": "机票 报销 酒店 支付记录",
                "metadata": {"smoke": True, "backend": backend},
            }
        ],
    )
    if inserted < 1:
        raise RuntimeError(f"{backend} upsert failed")

    hits = retriever.query_documents(
        query="机票 支付记录",
        source_types=["travel_case"],
        top_k=5,
        min_score=-1.0,
        metadata_filter={"smoke": True, "backend": backend},
    )
    if len(hits) < 1:
        raise RuntimeError(f"{backend} query returned no hits")

    deleted = retriever.delete_documents(source_type="travel_case", source_id=source_id)

    return {
        "ok": True,
        "backend": backend,
        "inserted": inserted,
        "hits": len(hits),
        "deleted": deleted,
    }


def main() -> int:
    _bootstrap_path()
    results: dict[str, dict] = {}
    for backend in ("sqlite", "faiss"):
        try:
            results[backend] = _smoke_backend(backend)
        except Exception as exc:
            results[backend] = {
                "ok": False,
                "backend": backend,
                "error": str(exc),
            }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item.get("ok") for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
