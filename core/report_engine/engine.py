"""
ReportEngine Facade：串联模板选择、布局、篇幅、章节、渲染。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from .config import ReportEngineConfig, ReportResult
from .ir.validator import IRValidator
from .nodes.template_selection_node import TemplateSelectionNode
from .nodes.document_layout_node import DocumentLayoutNode
from .nodes.word_budget_node import WordBudgetNode
from .nodes.chapter_generation_node import ChapterGenerationNode
from .renderers.html_renderer import HTMLRenderer
from .renderers.pdf_renderer import PDFRenderer
from .renderers.base import RenderContext
from ..llm_client import LLMClient
from .llm_strategy import LLMStrategy
from .compare import normalize_compare_items
from .view_model import ViewModelBuilder
import json


class ReportEngine:
    def __init__(self, config: ReportEngineConfig):
        self.config = config
        self.validator = IRValidator()
        self.html_renderer = HTMLRenderer()
        self.pdf_renderer = PDFRenderer()
        self.llm = None
        # 分层模型策略
        self.llm_strategy = LLMStrategy(
            default_model=config.model,
            task_models=config.model_tiers or {},
        )
        if config.enabled and config.api_key:
            try:
                self.llm = LLMClient(
                    api_key=config.api_key,
                    model_name=self.llm_strategy.default_model,
                    base_url=config.base_url,
                    timeout=1200,
                )
            except Exception as exc:
                logger.warning(f"ReportEngine LLM 初始化失败，自动禁用: {exc}")
                self.llm = None
                self.config.enabled = False

        scenario_cfg = self._scenario_overrides(config.scenario)
        template_dir = scenario_cfg.get("template_dir", config.template_dir)
        max_words = scenario_cfg.get("max_words", config.max_words)

        self.template_node = TemplateSelectionNode(self._llm_for("template"), template_dir)
        self.layout_node = DocumentLayoutNode(self._llm_for("layout"))
        self.budget_node = WordBudgetNode(self._llm_for("budget"), default_total=max_words)
        self.chapter_node = ChapterGenerationNode(self._llm_for("chapter"), self.validator)
        self.vm_builder = ViewModelBuilder()

    def generate(
        self,
        topic: str,
        sections_context: Dict[str, Any],
        summary: str = "",
        task_id: Optional[str] = None,
        enable_pdf: bool = True,
        compare_items: Optional[List[Dict[str, Any]]] = None,
        scenario: Optional[str] = None,
    ) -> ReportResult:
        if not self.config.enabled:
            logger.info("ReportEngine 未启用，跳过生成")
            return ReportResult()

        result = ReportResult()
        try:
            tpl = self.template_node.run(query=topic, summary=summary)
            tpl_content = tpl["content"]
            tpl_sections = self._extract_sections(tpl_content)
            compare_list = normalize_compare_items(compare_items)

            layout = self.layout_node.run(tpl_sections, topic, {"summary": summary})
            word_budget = self.budget_node.run(layout.get("toc", []), topic)

            chapters = []
            for sec in tpl_sections:
                chapter = self.chapter_node.run(
                    chapter_meta={"slug": sec.get("slug"), "title": sec.get("title")},
                    context=sections_context,
                )
                chapters.append(chapter)

            self.validator.validate_document(chapters)

            vm = self.vm_builder.build(layout, chapters, compare_list, sections_context)
            render_ctx = RenderContext(
                title=vm.title or topic,
                subtitle=vm.subtitle,
                toc=vm.toc,
                chapters=vm.chapters,
                model_info=f"{self.config.model}",
                data_time=vm.data_time,
                env_info=vm.env_info,
            )

            html = self.html_renderer.render(render_ctx)

            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = task_id or topic[:30] or "report"
            html_path = output_dir / f"{base_name}.html"
            ir_path = output_dir / f"{base_name}.json"
            self.html_renderer.persist(html, html_path)
            ir_path.write_text(json.dumps({"chapters": chapters}, ensure_ascii=False, indent=2), encoding="utf-8")

            # 结构化摘要（决策/证据/复现等）
            summary_payload = self._build_summary(
                sections_context, layout, word_budget, chapters, compare_list
            )
            summary_path = output_dir / f"{base_name}_summary.json"
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            pdf_path = None
            if enable_pdf and self.config.pdf_enabled:
                pdf_path = output_dir / f"{base_name}.pdf"
                pdf_written = self.pdf_renderer.render(render_ctx, html_content=html, output_path=pdf_path)
                if not pdf_written:
                    pdf_path = None

            result.html_path = html_path
            result.pdf_path = pdf_path
            result.ir_path = ir_path
            result.summary_path = summary_path
            result.html_content = html
            result.title = layout.get("title", topic)
            return result
        except Exception as exc:
            logger.exception(f"ReportEngine 生成失败: {exc}")
            return result

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        简易模板切片：按 markdown 二级标题拆分。
        """
        sections: List[Dict[str, Any]] = []
        current = None
        for line in content.splitlines():
            if line.startswith("## "):
                if current:
                    sections.append(current)
                title = line.lstrip("# ").strip()
                slug = title.lower().replace(" ", "-")
                current = {"title": title, "slug": slug}
        if current:
            sections.append(current)
        if not sections:
            sections = [{"title": "内容", "slug": "content"}]
        return sections

    def _scenario_overrides(self, scenario: str) -> Dict[str, Any]:
        """
        根据场景调整模板目录、篇幅等参数。
        """
        base_dir = self.config.template_dir
        overrides = {
            "review": {
                "template_dir": base_dir / "",
                "max_words": 5000,
            },
            "verification": {
                "template_dir": base_dir / "",
                "max_words": 4500,
            },
            "compare": {
                "template_dir": base_dir / "",
                "max_words": 5500,
            },
            "scholar": {
                "template_dir": base_dir / "",
                "max_words": 6000,
            },
        }
        return overrides.get(scenario, {})

    def _llm_for(self, task_type: str) -> Optional[LLMClient]:
        """
        按任务类型选择模型，若与默认一致则复用同一 client。
        """
        model_name = self.llm_strategy.pick(task_type)
        if not self.config.enabled or not self.config.api_key:
            return None
        # 若与默认一致，复用
        if self.llm and model_name == self.llm_strategy.default_model:
            return self.llm
        try:
            return LLMClient(
                api_key=self.config.api_key,
                model_name=model_name,
                base_url=self.config.base_url,
                timeout=1200,
            )
        except Exception as exc:
            logger.warning(f"LLM 初始化失败 task={task_type}, model={model_name}: {exc}")
            return self.llm

    def _build_summary(
        self,
        ctx: Dict[str, Any],
        layout: Dict[str, Any],
        budget: Dict[str, Any],
        chapters: List[Dict[str, Any]],
        compare_items: List[Any],
    ) -> Dict[str, Any]:
        """提取结构化摘要，便于下游集成。"""
        return {
            "title": layout.get("title"),
            "subtitle": layout.get("subtitle"),
            "toc": layout.get("toc", []),
            "word_budget": budget,
            "paper": ctx.get("paper"),
            "review": ctx.get("review"),
            "verification": ctx.get("verification"),
            "influence": ctx.get("influence"),
            "repro": ctx.get("repro"),
            "compare_items": [ci.__dict__ for ci in compare_items],
            "chapters": chapters,
            "env_info": ctx.get("env_info"),
            "data_time": ctx.get("data_time"),
        }

