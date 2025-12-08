"""
布局节点：生成标题、目录、主题要素。
"""

from __future__ import annotations

import json
from typing import Dict, Any, List

from .base_node import BaseNode
from ..utils.json_parser import JSONParseError

SYSTEM_PROMPT_LAYOUT = """你是报告布局设计师。根据模板章节与素材，输出 JSON:
{"title": "...", "subtitle": "...", "toc": [{"title": "...", "slug": "..."}], "theme": {"tone": "..."}}
保持简洁。"""


class DocumentLayoutNode(BaseNode):
    def run(self, template_sections: List[Dict[str, Any]], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return self._fallback(template_sections, query)

        sections_text = "\n".join([f"- {s.get('title')}" for s in template_sections])
        summary = context.get("summary", "")[:800]
        user_prompt = f"主题: {query}\n章节: {sections_text}\n摘要: {summary}"
        try:
            resp = self.llm.invoke(SYSTEM_PROMPT_LAYOUT, user_prompt, temperature=0.2, top_p=0.9)
            data = self.parser.parse(resp)
            return data
        except JSONParseError:
            self.warn("布局 JSON 解析失败，使用回退")
        except Exception as exc:
            self.warn(f"布局生成异常: {exc}")
        return self._fallback(template_sections, query)

    def _fallback(self, template_sections: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        toc = [{"title": s.get("title", ""), "slug": s.get("slug", f"sec-{i}")} for i, s in enumerate(template_sections)]
        return {
            "title": query[:60] or "Report",
            "subtitle": "Auto-generated report",
            "toc": toc,
            "theme": {"tone": "neutral"},
        }

