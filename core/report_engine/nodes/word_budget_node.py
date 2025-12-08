"""
篇幅规划节点：给出总字数和逐章 target/min/max。
"""

from __future__ import annotations

from typing import List, Dict, Any

from .base_node import BaseNode
from ..utils.json_parser import JSONParseError

SYSTEM_PROMPT_BUDGET = """你是篇幅规划助手。输入章节目录，输出 JSON:
{"totalWords": 6000, "chapters": [{"slug": "...", "target": 800, "min": 400, "max": 1200}]}"""


class WordBudgetNode(BaseNode):
    def __init__(self, llm_client, default_total: int = 6000):
        super().__init__(llm_client, "WordBudget")
        self.default_total = default_total

    def run(self, toc: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        if not self.llm:
            return self._fallback(toc)
        toc_text = "\n".join([f"- {t.get('title')}" for t in toc])
        user_prompt = f"主题: {query}\n目录:\n{toc_text}"
        try:
            resp = self.llm.invoke(SYSTEM_PROMPT_BUDGET, user_prompt, temperature=0.2, top_p=0.8)
            data = self.parser.parse(resp)
            return data
        except JSONParseError:
            self.warn("篇幅规划 JSON 解析失败，使用回退")
        except Exception as exc:
            self.warn(f"篇幅规划异常: {exc}")
        return self._fallback(toc)

    def _fallback(self, toc: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not toc:
            return {"totalWords": self.default_total, "chapters": []}
        per = max(int(self.default_total / max(len(toc), 1)), 400)
        return {
            "totalWords": self.default_total,
            "chapters": [
                {"slug": t.get("slug", f"sec-{i}"), "target": per, "min": int(per * 0.6), "max": int(per * 1.4)}
                for i, t in enumerate(toc)
            ],
        }

