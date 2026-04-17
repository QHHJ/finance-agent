from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

import requests


def _normalize_vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    vector: list[float] = []
    for value in values:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            continue
    return vector


def _l2_normalize(vector: list[float]) -> list[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    return [v / norm for v in vector]


def _hash_fallback_embedding(text: str, dim: int = 256) -> list[float]:
    tokens = re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]", (text or "").lower())
    if not tokens:
        return [0.0] * dim

    vector = [0.0] * dim
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        vector[index] += sign
    return _l2_normalize(vector)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _embedding_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _request_timeout() -> tuple[int, int]:
    connect_timeout = int(os.getenv("RAG_EMBED_CONNECT_TIMEOUT", "5"))
    read_timeout = int(os.getenv("RAG_EMBED_READ_TIMEOUT", "30"))
    return connect_timeout, read_timeout


def _embed_with_ollama_api_embed(texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return []
    payload = {"model": _embedding_model(), "input": texts}
    resp = requests.post(
        f"{_ollama_base_url()}/api/embed",
        json=payload,
        timeout=_request_timeout(),
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data.get("embeddings"), list):
        vectors = [_normalize_vector(item) for item in data["embeddings"]]
        if vectors:
            return vectors

    # Some versions may return a single embedding field.
    single = _normalize_vector(data.get("embedding"))
    if single:
        return [single]
    return None


def _embed_with_ollama_api_embeddings(texts: list[str]) -> list[list[float]] | None:
    vectors: list[list[float]] = []
    for text in texts:
        payload = {"model": _embedding_model(), "prompt": text}
        resp = requests.post(
            f"{_ollama_base_url()}/api/embeddings",
            json=payload,
            timeout=_request_timeout(),
        )
        resp.raise_for_status()
        vector = _normalize_vector(resp.json().get("embedding"))
        if not vector:
            return None
        vectors.append(vector)
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    clean_texts = [str(text or "").strip() for text in texts]
    if not clean_texts:
        return []

    try:
        vectors = _embed_with_ollama_api_embed(clean_texts)
        if vectors and len(vectors) == len(clean_texts):
            return [_l2_normalize(vector) for vector in vectors]
    except Exception:
        pass

    try:
        vectors = _embed_with_ollama_api_embeddings(clean_texts)
        if vectors and len(vectors) == len(clean_texts):
            return [_l2_normalize(vector) for vector in vectors]
    except Exception:
        pass

    return [_hash_fallback_embedding(text) for text in clean_texts]


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = sum(a[idx] * b[idx] for idx in range(length))
    if dot > 1.0:
        return 1.0
    if dot < -1.0:
        return -1.0
    return dot
