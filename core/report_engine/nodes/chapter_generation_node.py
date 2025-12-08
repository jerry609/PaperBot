"""
章节生成节点：LLM 生成 JSON；若不可用则使用 fallback 文本。
"""

from __future__ import annotations

import json
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger

from .base_node import BaseNode
from ..utils.json_parser import JSONParseError
from ..ir.schema import default_chapter_template
from ..ir.validator import IRValidator, IRValidationError


SYSTEM_PROMPT_CHAPTER = """你是报告撰写助手。根据章节题目与素材，输出 JSON:
{"slug": "...", "title": "...", "blocks": [{"type": "paragraph", "content": "..."}]}
仅输出 JSON，不要其他文本。"""


class ChapterGenerationNode(BaseNode):
    def __init__(self, llm_client, validator: IRValidator, min_body_chars: int = 400):
        super().__init__(llm_client, "ChapterGeneration")
        self.validator = validator
        self.min_body_chars = min_body_chars

    def run(
        self,
        chapter_meta: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        slug = chapter_meta.get("slug")
        title = chapter_meta.get("title")
        user_prompt = self._build_prompt(chapter_meta, context)

        if not self.llm:
            return self._fallback(slug, title, context)

        try:
            resp = self.llm.invoke(SYSTEM_PROMPT_CHAPTER, user_prompt, temperature=0.35, top_p=0.9)
            data = self.parser.parse(resp)
            self._check_density(data)
            self.validator.validate_chapter(data)
            return data
        except (JSONParseError, IRValidationError) as exc:
            self.warn(f"章节结构解析失败，使用回退: {exc}")
        except Exception as exc:
            self.warn(f"章节生成异常，使用回退: {exc}")
        return self._fallback(slug, title, context)

    def _build_prompt(self, chapter_meta: Dict[str, Any], context: Dict[str, Any]) -> str:
        info = {
            "chapter": chapter_meta,
            "context": context,
        }
        return json.dumps(info, ensure_ascii=False)

    def _fallback(self, slug: str, title: str, context: Dict[str, Any]) -> Dict[str, Any]:
        body_parts = []
        for key, val in context.items():
            if isinstance(val, dict):
                body_parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)[:400]}")
            else:
                body_parts.append(f"{key}: {str(val)[:400]}")
        body = "\n".join(body_parts) or "This section summarizes the findings."
        return default_chapter_template(slug=slug or "chapter", title=title or "章节", body=body)

    def _check_density(self, chapter: Dict[str, Any]) -> None:
        text = ""
        for b in chapter.get("blocks", []):
            content = b.get("content", "")
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                text += " ".join(content)
        if len(text) < self.min_body_chars:
            raise IRValidationError("章节正文过短")

