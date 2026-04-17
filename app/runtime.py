from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from app.db.session import Base, engine

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
POLICY_DIR = UPLOAD_DIR / "policies"
EXPORT_DIR = BASE_DIR / "data" / "exports"
DOTENV_PATH = BASE_DIR / ".env"


def init_runtime() -> None:
    # Always load project-level .env no matter where streamlit is started.
    load_dotenv(dotenv_path=DOTENV_PATH, override=True, encoding="utf-8-sig")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
