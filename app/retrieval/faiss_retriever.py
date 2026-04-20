from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import repo
from app.db.models import RagVectorDocument
from app.db.session import SessionLocal
from app.retrieval.base import Retriever
from app.retrieval.sqlite_retriever import SQLiteRetriever
from app.runtime import BASE_DIR
from app.services import rag_embedder


class FaissRetriever(Retriever):
    def __init__(self, index_dir: Path | None = None) -> None:
        self._sqlite = SQLiteRetriever()
        self.index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR", BASE_DIR / "data" / "faiss_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "rag_index.faiss"
        self.mapping_path = self.index_dir / "rag_index_mapping.json"
        self._index = None
        self._doc_ids: list[int] = []

    def upsert_documents(
        self,
        *,
        source_type: str,
        source_id: str,
        documents: list[dict[str, Any]],
    ) -> int:
        count = self._sqlite.upsert_documents(
            source_type=source_type,
            source_id=source_id,
            documents=documents,
        )
        if count > 0:
            self.rebuild_index()
        return count

    def delete_documents(
        self,
        *,
        source_type: str | None = None,
        source_id: str | None = None,
        doc_key_prefix: str | None = None,
    ) -> int:
        count = self._sqlite.delete_documents(
            source_type=source_type,
            source_id=source_id,
            doc_key_prefix=doc_key_prefix,
        )
        if count > 0:
            self.rebuild_index()
        return count

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
        query_text = str(query or "").strip()
        if not query_text:
            return []

        if not self._ensure_index_loaded():
            return self._sqlite.query_documents(
                query=query_text,
                source_types=source_types,
                top_k=top_k,
                min_score=min_score,
                limit_scan=limit_scan,
                metadata_filter=metadata_filter,
            )

        query_vec = rag_embedder.embed_text(query_text)
        if not query_vec:
            return []

        try:
            import numpy as np
        except Exception:
            return self._sqlite.query_documents(
                query=query_text,
                source_types=source_types,
                top_k=top_k,
                min_score=min_score,
                limit_scan=limit_scan,
                metadata_filter=metadata_filter,
            )

        if self._index is None or not self._doc_ids:
            return self._sqlite.query_documents(
                query=query_text,
                source_types=source_types,
                top_k=top_k,
                min_score=min_score,
                limit_scan=limit_scan,
                metadata_filter=metadata_filter,
            )

        candidate_k = min(max(20, top_k * 8), len(self._doc_ids))
        query_arr = np.asarray([query_vec], dtype="float32")
        scores, positions = self._index.search(query_arr, candidate_k)
        score_row = scores[0].tolist() if len(scores) else []
        pos_row = positions[0].tolist() if len(positions) else []

        id_to_score: dict[int, float] = {}
        for pos, score in zip(pos_row, score_row):
            if pos < 0 or pos >= len(self._doc_ids):
                continue
            doc_id = int(self._doc_ids[pos])
            prev = id_to_score.get(doc_id)
            if prev is None or score > prev:
                id_to_score[doc_id] = float(score)

        if not id_to_score:
            return []

        db = SessionLocal()
        try:
            stmt = select(RagVectorDocument).where(RagVectorDocument.id.in_(list(id_to_score.keys())))
            docs = list(db.scalars(stmt))
        finally:
            db.close()

        hits: list[dict[str, Any]] = []
        fallback_hits: list[dict[str, Any]] = []
        source_type_set = set(source_types or [])
        for doc in docs:
            if source_type_set and str(doc.source_type) not in source_type_set:
                continue
            metadata_json = dict(getattr(doc, "metadata_json", {}) or {})
            if not self._metadata_match(metadata_json, metadata_filter):
                continue

            score = float(id_to_score.get(int(doc.id), 0.0))
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
            if score >= min_score:
                hits.append(item)

        hits.sort(key=lambda item: item["score"], reverse=True)
        if hits:
            return hits[: max(1, top_k)]

        fallback_hits.sort(key=lambda item: item["score"], reverse=True)
        return fallback_hits[: max(1, top_k)]

    def rebuild_index(self) -> int:
        faiss = self._import_faiss()
        if faiss is None:
            return 0

        try:
            import numpy as np
        except Exception:
            return 0

        limit = max(1000, int(os.getenv("FAISS_REBUILD_SCAN_LIMIT", "200000")))
        db = SessionLocal()
        try:
            docs = repo.list_rag_documents(db, source_types=None, limit=limit)
        finally:
            db.close()

        vectors: list[list[float]] = []
        doc_ids: list[int] = []
        dim = 0
        for doc in docs:
            emb = list(getattr(doc, "embedding", []) or [])
            if not emb:
                continue
            emb = [float(v) for v in emb]
            if not emb:
                continue
            if dim == 0:
                dim = len(emb)
            if len(emb) != dim:
                continue
            vectors.append(emb)
            doc_ids.append(int(doc.id))

        if not vectors:
            self._index = None
            self._doc_ids = []
            if self.index_path.exists():
                self.index_path.unlink()
            self.mapping_path.write_text(json.dumps({"doc_ids": []}, ensure_ascii=False), encoding="utf-8")
            return 0

        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))
        self.mapping_path.write_text(json.dumps({"doc_ids": doc_ids}, ensure_ascii=False), encoding="utf-8")

        self._index = index
        self._doc_ids = doc_ids
        return len(doc_ids)

    def _ensure_index_loaded(self) -> bool:
        faiss = self._import_faiss()
        if faiss is None:
            return False

        if self._index is not None and self._doc_ids:
            return True

        if not self.index_path.exists() or not self.mapping_path.exists():
            return self.rebuild_index() > 0

        try:
            payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            doc_ids = [int(x) for x in list(payload.get("doc_ids") or [])]
            if not doc_ids:
                self._index = None
                self._doc_ids = []
                return False
            index = faiss.read_index(str(self.index_path))
            if int(index.ntotal) != len(doc_ids):
                return self.rebuild_index() > 0
            self._index = index
            self._doc_ids = doc_ids
            return True
        except Exception:
            return self.rebuild_index() > 0

    @staticmethod
    def _metadata_match(metadata_json: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata_json.get(key) != expected:
                return False
        return True

    @staticmethod
    def _import_faiss():
        try:
            import faiss

            return faiss
        except Exception:
            return None

