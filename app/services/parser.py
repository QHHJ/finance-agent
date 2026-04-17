from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader


def parse_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join(chunk for chunk in pages if chunk)


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
