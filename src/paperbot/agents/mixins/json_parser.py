# src/paperbot/agents/mixins/json_parser.py
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
    """JSON 解析错误。"""
    pass


class JSONParserMixin:
    """
    JSON 解析 Mixin。
    
    提供鲁棒的 JSON 解析能力，支持：
    - 标准 JSON 解析
    - 代码块标记清理
    - json_repair 回退修复
    """
    
    def parse_json(self, text: str, allow_repair: bool = True) -> Any:
        """
        解析 JSON 文本。
        
        Args:
            text: 待解析的文本
            allow_repair: 是否允许使用 json_repair 修复
            
        Returns:
            解析后的 Python 对象
            
        Raises:
            JSONParseError: 解析失败时抛出
        """
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
