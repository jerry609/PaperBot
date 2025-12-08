"""
模板选择节点：从模板目录选择最佳模板；若 LLM 不可用则选第一个。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .base_node import BaseNode
from ..utils.json_parser import JSONParseError


SYSTEM_PROMPT_TEMPLATE = """你是报告模板选择助手。根据用户主题与已有素材，选出最合适的模板。
只返回 JSON: {"template": "<文件名>", "reason": "<简要理由>"}"""


class TemplateSelectionNode(BaseNode):
    def __init__(self, llm_client, template_dir: Path):
        super().__init__(llm_client, "TemplateSelection")
        self.template_dir = template_dir

    def _list_templates(self) -> List[Path]:
        if not self.template_dir.exists():
            return []
        return [p for p in self.template_dir.glob("*.md")]

    def run(self, query: str, summary: str = "") -> Dict[str, Any]:
        candidates = self._list_templates()
        if not candidates:
            raise FileNotFoundError("未找到模板文件")

        if not self.llm:
            logger.info("LLM 不可用，使用默认模板")
            return self._fallback(candidates[0])

        names = [p.name for p in candidates]
        user_prompt = f"主题: {query}\n摘要: {summary[:800]}\n可选模板: {names}"
        try:
            resp = self.llm.invoke(SYSTEM_PROMPT_TEMPLATE, user_prompt, temperature=0.1)
            data = self.parser.parse(resp)
            chosen = data.get("template")
            if chosen and chosen in names:
                path = self.template_dir / chosen
                return {"path": path, "reason": data.get("reason", ""), "content": path.read_text(encoding="utf-8")}
        except JSONParseError:
            self.warn("LLM 解析失败，使用回退模板")
        except Exception as exc:
            self.warn(f"模板选择异常: {exc}")
        return self._fallback(candidates[0])

    def _fallback(self, path: Path) -> Dict[str, Any]:
        return {"path": path, "reason": "fallback", "content": path.read_text(encoding='utf-8')}

