from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

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
        self._faiss = self._import_faiss_strict()
        self._np = self._import_numpy_strict()

        self.index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR", BASE_DIR / "data" / "faiss_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "rag_index.faiss"
        self.mapping_path = self.index_dir / "rag_index_mapping.json"

        self._index = None
        self._doc_ids: list[int] = []
        self._db_snapshot: dict[str, Any] | None = None

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
            # Keep availability if index is temporarily unavailable.
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
        if self._index is None or not self._doc_ids:
            return []

        total_docs = len(self._doc_ids)
        query_arr = self._np.asarray([query_vec], dtype="float32")
        candidate_k = min(max(top_k * 20, 200), total_docs)

        # For source_types / metadata_filter, do one expanded search if needed.
        for pass_id in range(2):
            id_to_score = self._collect_candidates(query_arr, candidate_k)
            if not id_to_score:
                return []

            hits, fallback_hits = self._load_ranked_hits(
                id_to_score=id_to_score,
                source_types=source_types,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )
            if hits:
                return hits[: max(1, top_k)]
            if fallback_hits:
                return fallback_hits[: max(1, top_k)]

            if pass_id == 0 and candidate_k < total_docs:
                candidate_k = total_docs
                continue
            break

        return []

    def rebuild_index(self) -> int:
        limit = max(1000, int(os.getenv("FAISS_REBUILD_SCAN_LIMIT", "200000")))
        db = SessionLocal()
        try:
            docs = repo.list_rag_documents(db, source_types=None, limit=limit)
            db_snapshot = self._build_db_snapshot(db)
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
            self._db_snapshot = db_snapshot
            if self.index_path.exists():
                self.index_path.unlink()
            self._write_mapping(
                {
                    "doc_ids": [],
                    "db_snapshot": db_snapshot,
                    "dim": 0,
                    "index_type": "IndexFlatIP",
                    "built_at": datetime.utcnow().isoformat(),
                }
            )
            return 0

        matrix = self._np.asarray(vectors, dtype="float32")
        index = self._faiss.IndexFlatIP(dim)
        index.add(matrix)
        self._faiss.write_index(index, str(self.index_path))
        self._write_mapping(
            {
                "doc_ids": doc_ids,
                "db_snapshot": db_snapshot,
                "dim": dim,
                "index_type": "IndexFlatIP",
                "built_at": datetime.utcnow().isoformat(),
            }
        )

        self._index = index
        self._doc_ids = doc_ids
        self._db_snapshot = db_snapshot
        return len(doc_ids)

    def _ensure_index_loaded(self) -> bool:
        current_snapshot = self._current_db_snapshot()

        if self._index is not None and self._doc_ids and self._db_snapshot == current_snapshot:
            return True

        if not self.index_path.exists() or not self.mapping_path.exists():
            if int(current_snapshot.get("doc_count", 0)) <= 0:
                self._index = None
                self._doc_ids = []
                self._db_snapshot = current_snapshot
                return False
            return self.rebuild_index() > 0

        payload = self._read_mapping()
        mapped_doc_ids = [int(x) for x in list(payload.get("doc_ids") or [])]
        mapped_snapshot = dict(payload.get("db_snapshot") or {})

        if mapped_snapshot != current_snapshot:
            return self.rebuild_index() > 0

        if not mapped_doc_ids:
            self._index = None
            self._doc_ids = []
            self._db_snapshot = current_snapshot
            return False

        try:
            index = self._faiss.read_index(str(self.index_path))
        except Exception:
            return self.rebuild_index() > 0

        if int(index.ntotal) != len(mapped_doc_ids):
            return self.rebuild_index() > 0

        self._index = index
        self._doc_ids = mapped_doc_ids
        self._db_snapshot = current_snapshot
        return True

    def _collect_candidates(self, query_arr, top_k: int) -> dict[int, float]:
        if self._index is None or not self._doc_ids or top_k <= 0:
            return {}
        scores, positions = self._index.search(query_arr, top_k)
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
        return id_to_score

    def _load_ranked_hits(
        self,
        *,
        id_to_score: dict[int, float],
        source_types: list[str],
        min_score: float,
        metadata_filter: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not id_to_score:
            return [], []

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
        fallback_hits.sort(key=lambda item: item["score"], reverse=True)
        return hits, fallback_hits

    def _current_db_snapshot(self) -> dict[str, Any]:
        db = SessionLocal()
        try:
            return self._build_db_snapshot(db)
        finally:
            db.close()

    @staticmethod
    def _build_db_snapshot(db) -> dict[str, Any]:
        row = db.execute(
            select(
                func.count(RagVectorDocument.id),
                func.max(RagVectorDocument.id),
                func.max(RagVectorDocument.updated_at),
            )
        ).one()
        count = int(row[0] or 0)
        max_id = int(row[1] or 0) if row[1] is not None else 0
        max_updated = row[2].isoformat() if row[2] is not None else ""
        return {"doc_count": count, "max_doc_id": max_id, "max_updated_at": max_updated}

    def _read_mapping(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _write_mapping(self, payload: dict[str, Any]) -> None:
        self.mapping_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _metadata_match(metadata_json: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata_json.get(key) != expected:
                return False
        return True

    @staticmethod
    def _import_faiss_strict():
        try:
            import faiss

            return faiss
        except Exception as exc:
            raise RuntimeError(
                "FAISS backend is enabled but faiss is not available. "
                "Install dependency: pip install faiss-cpu"
            ) from exc

    @staticmethod
    def _import_numpy_strict():
        try:
            import numpy as np

            return np
        except Exception as exc:
            raise RuntimeError(
                "FAISS backend requires numpy. Install dependency: pip install numpy"
            ) from exc
