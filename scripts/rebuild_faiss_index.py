from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> int:
    _bootstrap_path()
    from app.retrieval.rebuild import rebuild_faiss

    count = rebuild_faiss()
    print(f"[OK] FAISS index rebuilt. indexed={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
