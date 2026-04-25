from __future__ import annotations

import base64
import hashlib
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore[import-not-found]
except Exception:
    fitz = None
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
import requests

try:
    from PIL import Image
except Exception:
    Image = None

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _base_url() -> str:
    return str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")


def _vl_model() -> str:
    return str(os.getenv("OLLAMA_VL_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5vl:3b")


def _normalize_suffix(suffix: str) -> str:
    value = str(suffix or "").strip().lower()
    if not value:
        return ""
    return value if value.startswith(".") else f".{value}"


def _clean_ocr_text(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^```[a-zA-Z]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _render_pdf_pages_to_base64_images(file_bytes: bytes, *, max_pages: int, render_scale: float) -> list[str]:
    if not file_bytes or fitz is None:
        return []
    pages: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page_count = len(doc)
        if page_count <= 0:
            return []
        limit = page_count if max_pages <= 0 else min(page_count, max_pages)
        for page_idx in range(limit):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
            png_bytes = pix.tobytes("png")
            pages.append(base64.b64encode(png_bytes).decode("utf-8"))
    return pages


def _encode_file_bytes_to_images(file_bytes: bytes, suffix: str, *, max_pages: int, render_scale: float) -> list[str]:
    normalized = _normalize_suffix(suffix)
    if normalized in IMAGE_SUFFIXES:
        return [base64.b64encode(file_bytes).decode("utf-8")] if file_bytes else []
    if normalized == ".pdf":
        return _render_pdf_pages_to_base64_images(file_bytes, max_pages=max_pages, render_scale=render_scale)
    return []


def _rotate_image_to_base64(file_bytes: bytes, degrees: int) -> str:
    if not file_bytes or Image is None:
        return ""
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            rotated = image.rotate(degrees, expand=True)
            output = BytesIO()
            rotated.convert("RGB").save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _fallback_pdf_text_from_bytes(file_bytes: bytes) -> str:
    if not file_bytes or PdfReader is None:
        return ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        return ""
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(text)
    return "\n".join(chunk for chunk in pages if chunk)


def _ollama_chat_with_images(
    *,
    base_url: str,
    model: str,
    images: list[str],
    timeout_sec: int,
    fallback_timeout_sec: int,
) -> str:
    system_prompt = "你是OCR助手。只提取可见文本内容，尽量保持阅读顺序。"
    user_prompt = (
        "请做OCR，只输出文本本身。不要解释，不要总结，不要输出JSON。"
        "如果票据是横向、倒置或旋转的，请先按正确阅读方向理解后再提取。"
        "发票请尽量保留发票号码、开票日期、购买方、销售方、项目名称/服务名称、备注、价税合计等关键字段。"
    )
    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": images},
            ],
            "options": {"temperature": 0.2},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=(8, timeout_sec))
        resp.raise_for_status()
        content = str((resp.json().get("message") or {}).get("content") or "").strip()
        if content:
            return content
    except Exception:
        pass

    fallback_prompt = f"[system] {system_prompt}\n\n[user] {user_prompt}"
    payload = {
        "model": model,
        "stream": False,
        "prompt": fallback_prompt,
        "images": images,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=(8, fallback_timeout_sec))
    resp.raise_for_status()
    return str(resp.json().get("response") or "").strip()


def parse_file_bytes(file_bytes: bytes, suffix: str, *, max_pages: int | None = None) -> str:
    normalized = _normalize_suffix(suffix)
    if not file_bytes:
        return ""

    if normalized in TEXT_SUFFIXES:
        try:
            return file_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if normalized == ".pdf" and fitz is None:
        return _fallback_pdf_text_from_bytes(file_bytes)

    if max_pages is None:
        resolved_max_pages = _env_int("OLLAMA_OCR_MAX_PAGES", 2)
    else:
        try:
            resolved_max_pages = int(max_pages)
        except (TypeError, ValueError):
            resolved_max_pages = 2
    if resolved_max_pages < 0:
        resolved_max_pages = 2
    render_scale = _env_float("OLLAMA_OCR_RENDER_SCALE", 2.0)
    timeout_sec = _env_int("OLLAMA_OCR_TIMEOUT", 14)
    fallback_timeout_sec = _env_int("OLLAMA_OCR_FALLBACK_TIMEOUT", 6)
    if timeout_sec < 8:
        timeout_sec = 8
    if fallback_timeout_sec < 6:
        fallback_timeout_sec = 6
    images = _encode_file_bytes_to_images(
        file_bytes,
        normalized,
        max_pages=resolved_max_pages,
        render_scale=render_scale,
    )
    if not images:
        return ""

    try:
        text = _ollama_chat_with_images(
            base_url=_base_url(),
            model=_vl_model(),
            images=images,
            timeout_sec=timeout_sec,
            fallback_timeout_sec=fallback_timeout_sec,
        )
    except Exception:
        text = ""

    compact_text_len = len(re.sub(r"\s+", "", _clean_ocr_text(text)))
    rotation_min_chars = _env_int("OLLAMA_OCR_ROTATE_MIN_CHARS", 50)
    rotation_enabled = str(os.getenv("OLLAMA_OCR_ROTATE_FALLBACK", "1")).strip().lower() in {"1", "true", "yes", "on"}
    best_text = _clean_ocr_text(text)
    invoice_like = any(token in best_text for token in ["发票", "价税合计", "税额", "金额(小写)", "开票日期", "购买方", "销售方"])
    missing_invoice_item = invoice_like and not any(
        token in best_text for token in ["项目名称", "服务名称", "住宿", "房费", "机票", "客运", "代订"]
    )
    should_try_rotation = compact_text_len < rotation_min_chars or missing_invoice_item
    if normalized in IMAGE_SUFFIXES and rotation_enabled and should_try_rotation:
        best_len = compact_text_len
        rotate_timeout_sec = max(timeout_sec, _env_int("OLLAMA_OCR_ROTATE_TIMEOUT", 30))
        rotate_fallback_timeout_sec = max(fallback_timeout_sec, _env_int("OLLAMA_OCR_ROTATE_FALLBACK_TIMEOUT", 12))
        for degrees in (270, 90, 180):
            rotated_image = _rotate_image_to_base64(file_bytes, degrees)
            if not rotated_image:
                continue
            try:
                candidate = _ollama_chat_with_images(
                    base_url=_base_url(),
                    model=_vl_model(),
                    images=[rotated_image],
                    timeout_sec=rotate_timeout_sec,
                    fallback_timeout_sec=rotate_fallback_timeout_sec,
                )
            except Exception:
                continue
            candidate = _clean_ocr_text(candidate)
            candidate_len = len(re.sub(r"\s+", "", candidate))
            if candidate_len > best_len:
                best_text = candidate
                best_len = candidate_len
            if best_len >= rotation_min_chars:
                break
        return best_text
    return _clean_ocr_text(text)


def parse_file_text(file_path: str | Path, *, max_pages: int | None = None) -> str:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return ""
    try:
        file_bytes = path.read_bytes()
    except Exception:
        return ""
    return parse_file_bytes(file_bytes, path.suffix, max_pages=max_pages)


def parse_pdf_text(pdf_path: str | Path) -> str:
    # Backward-compatible entry point; now unified to LLM OCR.
    return parse_file_text(pdf_path)


def compute_file_sha256(file_path: str | Path) -> str:
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
