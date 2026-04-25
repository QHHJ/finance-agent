from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object_loose(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        parsed = json.loads(source)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", source)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
