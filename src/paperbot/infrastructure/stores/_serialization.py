"""Shared JSON serialization helpers for Store / Repository classes.

These eliminate the repeated ``json.dumps(x or {}, ensure_ascii=False)``
and ``json.loads(raw or "{}")`` patterns scattered across stores.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def dump_json(data: Any, *, fallback: Any = None) -> str:
    """Serialize *data* to a JSON string (``ensure_ascii=False``).

    If *data* is falsy, *fallback* (default ``{}``) is serialized instead.
    """
    if fallback is None:
        fallback = {}
    return json.dumps(data or fallback, ensure_ascii=False)


def load_json_dict(raw: str | None) -> Dict[str, Any]:
    """Deserialize a JSON string that is expected to be a dict.

    Returns ``{}`` when *raw* is ``None`` or empty.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def load_json_list(raw: str | None) -> List[Any]:
    """Deserialize a JSON string that is expected to be a list.

    Returns ``[]`` when *raw* is ``None`` or empty.
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
