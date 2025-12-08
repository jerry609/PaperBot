"""
鲁棒 JSON 解析：优先标准 json，失败再尝试 json_repair。
"""

from __future__ import annotations

import json
from typing import Any

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None


class JSONParseError(ValueError):
    pass


class RobustJSONParser:
    def parse(self, text: str) -> Any:
        if not text or not isinstance(text, str):
            raise JSONParseError("空文本")
        # 去除围栏
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        try:
            return json.loads(cleaned)
        except Exception:
            if repair_json:
                try:
                    fixed = repair_json(cleaned)
                    return json.loads(fixed)
                except Exception:
                    pass
        raise JSONParseError("JSON 解析失败")

