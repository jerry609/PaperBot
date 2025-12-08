"""
通用 JSON 解析 Mixin，支持 json_repair 回退。
"""

import json
from typing import Any

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None


class JSONParseError(ValueError):
    pass


class JSONParserMixin:
    def parse_json(self, text: str, allow_repair: bool = True) -> Any:
        if not text or not isinstance(text, str):
            raise JSONParseError("空响应或类型错误")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        try:
            return json.loads(cleaned)
        except Exception:
            if allow_repair and repair_json:
                try:
                    fixed = repair_json(cleaned)
                    return json.loads(fixed)
                except Exception:
                    pass
        raise JSONParseError("JSON 解析失败")

