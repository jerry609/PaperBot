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
    """JSON 解析错误。"""
    pass


class RobustJSONParser:
    """
    鲁棒的 JSON 解析器。
    
    先尝试标准 json 解析，失败后尝试使用 json_repair 修复。
    """
    
    def parse(self, text: str) -> Any:
        """
        解析 JSON 文本。
        
        Args:
            text: 待解析的文本
            
        Returns:
            解析后的 Python 对象
            
        Raises:
            JSONParseError: 解析失败时抛出
        """
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

