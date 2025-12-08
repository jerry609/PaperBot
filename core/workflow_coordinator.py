# core/workflow_coordinator.py
"""
MVP 版工作流协调器
用于串联多 Agent 完成学者追踪论文分析流水线
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from agents import (
    ResearchAgent,
    CodeAnalysisAgent,
    QualityAgent,
    DocumentationAgent,
)
from scholar_tracking.models import PaperMeta, CodeMeta
from scholar_tracking.models.influence import InfluenceResult
from influence import InfluenceCalculator
from reports.writer import ReportWriter
from core.collaboration import (
    CollaborationBus,
    HostOrchestrator,
    HostConfig,
    AgentMessage,
    MessageType,
)
from core.report_engine import ReportEngine, ReportEngineConfig

logger = logging.getLogger(__name__)


class ScholarWorkflowCoordinator:
    """
    学者追踪工作流协调器 (MVP 版)
    
    顺序执行流水线:
    1. ResearchAgent → 扩展摘要 + 代码仓库链接
    2. CodeAnalysisAgent → 代码质量分析
    3. QualityAgent → 综合质量评价
    4. InfluenceCalculator → PIS 评分
    5. DocumentationAgent → 生成 Markdown 报告
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        report_writer: Optional[ReportWriter] = None,
    ):
        """
        初始化协调器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collab_settings = self.config.get("collab", {})
        
        # 初始化 Agents
        self.research_agent = ResearchAgent(config)
        self.code_analysis_agent = CodeAnalysisAgent(config)
        self.quality_agent = QualityAgent(config)
        self.documentation_agent = DocumentationAgent(config)
        
        # 初始化影响力计算器
        self.influence_calculator = InfluenceCalculator(config)

        # 报告渲染器
        output_dir = None
        if self.config.get("output_dir"):
            output_dir = Path(self.config["output_dir"])
        template_name = self.config.get("report_template", "paper_report.md.j2")
        self.report_writer = report_writer or ReportWriter(
            output_dir=output_dir,
            template_name=template_name,
        )

        # 协作总线与主持人
        self.collab_bus = CollaborationBus()
        self.host = HostOrchestrator(self._build_host_config())
        self.collab_enabled = bool(self.collab_settings.get("enabled", True))

        # Report Engine
        self.report_engine_cfg = self._build_report_engine_config()
        self.report_engine = ReportEngine(self.report_engine_cfg)

        # 复现结果占位
        self._latest_repro = None
        self._env_info = self._build_env_info()
    
    async def run_paper_pipeline(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str] = None,
        persist_report: bool = True,
    ) -> Tuple[Optional[Path], InfluenceResult, Dict[str, Any]]:
        """
        运行论文分析流水线
        
        Args:
            paper: 论文元数据
            scholar_name: 学者名称（用于报告生成）
            persist_report: 是否写入 Markdown 文件
            
        Returns:
            (report_path, influence_result, pipeline_data)
        """
        self._validate_paper_meta(paper)
        self._current_paper_ctx = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
        }
        pipeline_data = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "scholar_name": scholar_name,
            "started_at": datetime.now().isoformat(),
            "stages": {},
            "errors": [],
            "collab_log_path": None,
        }
        
        code_meta = None
        research_result = {}
        code_analysis_result = {}
        quality_result = {}
        report_path: Optional[Path] = None
        report_markdown: Optional[str] = None
        
        try:
            # Stage 1: Research Agent - 扩展摘要和代码仓库发现
            self.logger.info(f"[1/5] Running ResearchAgent for: {paper.title[:50]}...")
            try:
                research_result = await self._run_research_stage(paper)
                pipeline_data["stages"]["research"] = {
                    "status": "success",
                    "result": research_result,
                }
                self._emit_stage_message(
                    stage="research",
                    content=f"ResearchAgent 完成: {paper.title}",
                    payload=research_result,
                )
                
                # 从研究结果中提取代码仓库信息
                if research_result.get("github_url"):
                    paper.github_url = research_result["github_url"]
                    paper.has_code = True
            except Exception as e:
                self.logger.warning(f"ResearchAgent failed: {e}")
                pipeline_data["stages"]["research"] = {"status": "failed", "error": str(e)}
                pipeline_data["errors"].append(f"Research: {e}")
            
            # Stage 2: Code Analysis Agent - 代码分析
            self.logger.info(f"[2/5] Running CodeAnalysisAgent...")
            if paper.github_url or paper.has_code:
                try:
                    code_analysis_result = await self._run_code_analysis_stage(paper)
                    pipeline_data["stages"]["code_analysis"] = {
                        "status": "success",
                        "result": code_analysis_result,
                    }
                    self._emit_stage_message(
                        stage="code_analysis",
                        content="CodeAnalysisAgent 完成",
                        payload=code_analysis_result,
                    )
                    
                    # 构建 CodeMeta
                    code_meta = self._build_code_meta(paper, code_analysis_result)
                except Exception as e:
                    self.logger.warning(f"CodeAnalysisAgent failed: {e}")
                    pipeline_data["stages"]["code_analysis"] = {"status": "failed", "error": str(e)}
                    pipeline_data["errors"].append(f"CodeAnalysis: {e}")
                    self._emit_stage_message(
                        stage="code_analysis",
                        content=f"CodeAnalysisAgent 失败: {e}",
                        payload={"error": str(e)},
                        message_type=MessageType.ERROR,
                    )
            else:
                pipeline_data["stages"]["code_analysis"] = {"status": "skipped", "reason": "no code"}
                self._emit_stage_message(
                    stage="code_analysis",
                    content="CodeAnalysisAgent 跳过（无代码仓库）",
                    payload={"reason": "no code"},
                    message_type=MessageType.RESULT,
                )
            
            # Stage 3: Quality Agent - 质量评估
            self.logger.info(f"[3/5] Running QualityAgent...")
            try:
                quality_result = await self._run_quality_stage(
                    paper, research_result, code_analysis_result
                )
                pipeline_data["stages"]["quality"] = {
                    "status": "success",
                    "result": quality_result,
                }
                # 捕获复现/可运行性结果（如果下游质量阶段已有）
                if quality_result.get("repro"):
                    self._latest_repro = quality_result.get("repro")
                self._emit_stage_message(
                    stage="quality",
                    content="QualityAgent 完成",
                    payload=quality_result,
                )
            except Exception as e:
                self.logger.warning(f"QualityAgent failed: {e}")
                pipeline_data["stages"]["quality"] = {"status": "failed", "error": str(e)}
                pipeline_data["errors"].append(f"Quality: {e}")
                self._emit_stage_message(
                    stage="quality",
                    content=f"QualityAgent 失败: {e}",
                    payload={"error": str(e)},
                    message_type=MessageType.ERROR,
                )
            
            # Stage 4: Influence Calculator - 影响力评分
            self.logger.info(f"[4/5] Calculating influence score...")
            influence_result = self.influence_calculator.calculate(paper, code_meta)
            pipeline_data["stages"]["influence"] = {
                "status": "success",
                "result": influence_result.to_dict(),
            }
            self._emit_stage_message(
                stage="influence",
                content="InfluenceCalculator 完成",
                payload=influence_result.to_dict(),
            )
            
            # Stage 5: Report Rendering
            self.logger.info(f"[5/5] Generating report...")
            try:
                report_markdown = await self._generate_report(
                    paper=paper,
                    scholar_name=scholar_name,
                    research_result=self._ensure_defaults(research_result, default={}),
                    code_analysis_result=self._ensure_defaults(
                        code_analysis_result,
                        default={"repo_url": paper.github_url, "repo_name": None},
                    ),
                    quality_result=self._ensure_defaults(quality_result, default={}),
                    influence_result=influence_result,
                    env_info=self._env_info,
                )
                pipeline_data["stages"]["documentation"] = {"status": "success"}
            except Exception as e:
                self.logger.warning(f"Documentation stage failed: {e}")
                pipeline_data["stages"]["documentation"] = {
                    "status": "failed",
                    "error": str(e),
                }
                pipeline_data["errors"].append(f"Documentation: {e}")
                report_markdown = self._generate_fallback_report(
                    paper, influence_result, pipeline_data
                )
                self._emit_stage_message(
                    stage="documentation",
                    content="DocumentationAgent 完成",
                    payload={"has_report": bool(report_markdown)},
                )

            # 新版 Report Engine 输出
            if self.report_engine_cfg.enabled:
                try:
                    re_result = self._run_report_engine(
                        topic=paper.title,
                        summary=research_result.get("summary", ""),
                        sections_context={
                            "paper": paper.to_dict(),
                            "research": research_result,
                            "code_analysis": code_analysis_result,
                            "quality": quality_result,
                            "influence": influence_result.to_dict(),
                            "repro": self._latest_repro,
                            "env_info": self._env_info,
                            "data_time": pipeline_data.get("started_at"),
                        },
                        task_id=paper.paper_id or paper.title,
                    )
                    pipeline_data["stages"]["report_engine"] = {
                        "status": "success",
                        "html": str(re_result.html_path) if re_result.html_path else None,
                        "pdf": str(re_result.pdf_path) if re_result.pdf_path else None,
                        "ir": str(re_result.ir_path) if re_result.ir_path else None,
                    }
                except Exception as exc:
                    self.logger.warning(f"ReportEngine 生成失败: {exc}")
                    pipeline_data["stages"]["report_engine"] = {"status": "failed", "error": str(exc)}
                    pipeline_data["errors"].append(f"ReportEngine: {exc}")
            
            # 持久化报告
            if report_markdown:
                if persist_report:
                    try:
                        report_path = self.report_writer.write_report(
                            report_markdown,
                            paper,
                            scholar_name,
                        )
                        pipeline_data["report_path"] = str(report_path)
                    except Exception as e:
                        self.logger.error(f"Failed to write report: {e}")
                        pipeline_data["errors"].append(f"ReportWrite: {e}")
                        pipeline_data["report_content"] = report_markdown
                else:
                    pipeline_data["report_content"] = report_markdown
            
            pipeline_data["completed_at"] = datetime.now().isoformat()
            pipeline_data["status"] = "success" if not pipeline_data["errors"] else "partial"

            # 持久化协作日志
            pipeline_data["collab_log_path"] = str(self._persist_collab_log(paper))
            
            return report_path, influence_result, pipeline_data
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {paper.title}: {e}")
            pipeline_data["status"] = "failed"
            pipeline_data["errors"].append(str(e))
            
            # 即使失败也计算影响力分数
            influence_result = self.influence_calculator.calculate(paper, None)
            report_markdown = self._generate_fallback_report(paper, influence_result, pipeline_data)
            
            if report_markdown and persist_report:
                try:
                    report_path = self.report_writer.write_report(
                        report_markdown,
                        paper,
                        scholar_name,
                    )
                    pipeline_data["report_path"] = str(report_path)
                except Exception as write_error:
                    self.logger.error(f"Failed to write fallback report: {write_error}")
                    pipeline_data["errors"].append(f"ReportWrite: {write_error}")
                    pipeline_data["report_content"] = report_markdown
            else:
                pipeline_data["report_content"] = report_markdown
            pipeline_data["collab_log_path"] = str(self._persist_collab_log(paper))
            return report_path, influence_result, pipeline_data

    # =========================================================
    # 协作与主持人辅助函数
    # =========================================================

    def _emit_stage_message(
        self,
        stage: str,
        content: str,
        payload: Optional[dict] = None,
        message_type: MessageType = MessageType.RESULT,
    ):
        """写入协作总线并尝试触发主持人引导。"""
        if not self.collab_enabled:
            return
        msg = AgentMessage(
            sender=stage,
            message_type=message_type,
            content=content,
            metadata=payload or {},
            stage=stage,
        )
        self.collab_bus.add_message(msg)
        self._maybe_host_guidance(stage)

    def _maybe_host_guidance(self, stage: str):
        """主持人根据最近消息生成引导，失败自动降级。"""
        if not self.collab_enabled or not self.host.is_available():
            return
        recent = self.collab_bus.latest_messages(limit=20)
        guidance = self.host.generate_guidance(
            messages=recent,
            context={**(self._current_paper_ctx or {}), "stage": stage},
        )
        if guidance:
            self.collab_bus.add_host_message(guidance, stage=stage)
            self.collab_bus.next_round()

    def _persist_collab_log(self, paper: PaperMeta) -> Path:
        """持久化协作日志到 output/collab_logs 下。"""
        base_dir = self.config.get("output_dir") or "./output"
        log_dir = Path(base_dir) / "collab_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{paper.paper_id or 'paper'}_{timestamp}.jsonl"
        return self.collab_bus.persist(log_dir / filename)

    def _build_host_config(self) -> HostConfig:
        """从配置构造主持人配置，缺省使用通用 OpenAI Key。"""
        host_cfg = self.collab_settings.get("host", {})
        api_key = host_cfg.get("api_key") or self.config.get("openai_api_key") or self.config.get("api_key")
        model = host_cfg.get("model") or self.config.get("host_model") or "gpt-4o-mini"
        base_url = host_cfg.get("base_url") or self.config.get("host_base_url")
        enabled = bool(host_cfg.get("enabled", False))
        return HostConfig(
            enabled=enabled,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=host_cfg.get("temperature", 0.3),
            top_p=host_cfg.get("top_p", 0.9),
        )

    def _build_report_engine_config(self) -> ReportEngineConfig:
        cfg = self.config.get("report_engine", {})
        return ReportEngineConfig(
            enabled=cfg.get("enabled", False),
            api_key=cfg.get("api_key") or self.config.get("openai_api_key"),
            model=cfg.get("model", "gpt-4o-mini"),
            base_url=cfg.get("base_url"),
            output_dir=Path(cfg.get("output_dir", "output/reports")),
            template_dir=Path(cfg.get("template_dir", "core/report_engine/templates")),
            pdf_enabled=cfg.get("pdf_enabled", True),
            max_words=cfg.get("max_words", 6000),
        )

    def _run_report_engine(
        self,
        topic: str,
        summary: str,
        sections_context: Dict[str, Any],
        task_id: str,
    ):
        return self.report_engine.generate(
            topic=topic,
            summary=summary,
            sections_context=sections_context,
            task_id=task_id,
            enable_pdf=self.report_engine_cfg.pdf_enabled,
        )

    def _build_env_info(self) -> str:
        """构造环境信息摘要（模型/镜像/资源限制）。"""
        parts = []
        if self.report_engine_cfg.enabled:
            parts.append(f"ReportEngineModel={self.report_engine_cfg.model}")
        repro_cfg = self.config.get("repro", {})
        if repro_cfg:
            parts.append(f"DockerImage={repro_cfg.get('docker_image','')}")
            parts.append(f"CPU={repro_cfg.get('cpu_shares','')}")
            parts.append(f"Mem={repro_cfg.get('mem_limit','')}")
            parts.append(f"Network={repro_cfg.get('network')}")
        return "; ".join([p for p in parts if p])
    
    async def _run_research_stage(self, paper: PaperMeta) -> Dict[str, Any]:
        """运行研究阶段"""
        # 调用 ResearchAgent 获取更多论文信息
        result = await self.research_agent.process(
            paper_title=paper.title,
            paper_id=paper.paper_id,
            abstract=paper.abstract,
        )
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    async def _run_code_analysis_stage(self, paper: PaperMeta) -> Dict[str, Any]:
        """运行代码分析阶段"""
        if not paper.github_url:
            return {"status": "no_code_url"}
        
        result = await self.code_analysis_agent.process(
            repo_url=paper.github_url,
        )
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    async def _run_quality_stage(
        self,
        paper: PaperMeta,
        research_result: Dict[str, Any],
        code_analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行质量评估阶段"""
        # 合并上下文信息
        context = {
            "paper": paper.to_dict(),
            "research": research_result,
            "code_analysis": code_analysis_result,
        }
        
        result = await self.quality_agent.process(context)
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    def _build_code_meta(
        self,
        paper: PaperMeta,
        code_analysis_result: Dict[str, Any],
    ) -> Optional[CodeMeta]:
        """从代码分析结果构建 CodeMeta"""
        if not paper.github_url:
            return None
        
        try:
            return CodeMeta(
                repo_url=paper.github_url,
                repo_name=code_analysis_result.get("repo_name"),
                stars=code_analysis_result.get("stars", 0),
                forks=code_analysis_result.get("forks", 0),
                language=code_analysis_result.get("language"),
                updated_at=code_analysis_result.get("updated_at"),
                has_readme=code_analysis_result.get("has_readme", False),
                reproducibility_score=code_analysis_result.get("reproducibility_score"),
            )
        except Exception as e:
            self.logger.warning(f"Failed to build CodeMeta: {e}")
            return CodeMeta(repo_url=paper.github_url)
    
    async def _generate_report(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str],
        research_result: Dict[str, Any],
        code_analysis_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        influence_result: InfluenceResult,
    ) -> str:
        """生成完整的 Markdown 报告"""
        # 调用 Jinja 模板生成报告，可选地使用 DocumentationAgent 丰富内容
        report_data = {
            "paper": paper.to_dict(),
            "scholar_name": scholar_name,
            "research": research_result,
            "code_analysis": code_analysis_result,
            "quality": quality_result,
            "influence": influence_result.to_dict(),
        }

        if self.config.get("use_documentation_agent"):
            try:
                doc_result = await self.documentation_agent.process(report_data)
                report_data["documentation_agent"] = doc_result
            except Exception as e:
                self.logger.warning(f"DocumentationAgent enrichment failed: {e}")
        
        return self.report_writer.render_template(
            paper=paper,
            influence=influence_result,
            research_result=research_result,
            code_analysis_result=code_analysis_result,
            quality_result=quality_result,
            scholar_name=scholar_name,
        )
    
    def _generate_fallback_report(
        self,
        paper: PaperMeta,
        influence_result: InfluenceResult,
        pipeline_data: Dict[str, Any],
    ) -> str:
        """生成备用报告（当 DocumentationAgent 失败时）"""
        authors = ", ".join(paper.authors) if paper.authors else "未知"
        
        report = f"""# {paper.title}

## 📋 元信息

| 属性 | 值 |
|------|-----|
| 作者 | {authors} |
| 年份 | {paper.year or '未知'} |
| 发表于 | {paper.venue or '未知'} |
| 引用数 | {paper.citation_count} |
| Semantic Scholar ID | {paper.paper_id} |

## 📝 摘要

{paper.abstract or paper.tldr or '暂无摘要'}

## 💻 代码信息

"""
        if paper.github_url:
            report += f"- **仓库地址**: [{paper.github_url}]({paper.github_url})\n"
        else:
            report += "未发现公开代码仓库\n"
        
        report += f"""
## 📊 影响力评分 (PIS)

| 指标 | 分数 |
|------|------|
| **总分** | {influence_result.total_score:.1f}/100 |
| 学术影响力 | {influence_result.academic_score:.1f}/100 |
| 工程影响力 | {influence_result.engineering_score:.1f}/100 |

{influence_result.explanation}

## 🎯 推荐级别

**{influence_result.recommendation.value}**

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*由 PaperBot 自动生成*
"""
        return report

    def _validate_paper_meta(self, paper: PaperMeta) -> None:
        """最小校验，提前发现必填字段缺失"""
        missing = []
        if not paper.paper_id:
            missing.append("paper_id")
        if not paper.title:
            missing.append("title")
        if missing:
            raise ValueError(f"PaperMeta missing required fields: {', '.join(missing)}")

    def _ensure_defaults(self, value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """确保传递给模板的数据包含默认字段，避免 KeyError"""
        if not isinstance(value, dict):
            return default or {}
        merged = dict(default or {})
        merged.update({k: v for k, v in value.items() if v is not None})
        return merged
    
    async def run_batch_pipeline(
        self,
        papers: List[PaperMeta],
        scholar_name: Optional[str] = None,
    ) -> List[Tuple[str, InfluenceResult, Dict[str, Any]]]:
        """
        批量运行论文分析流水线
        
        Args:
            papers: 论文列表
            scholar_name: 学者名称
            
        Returns:
            结果列表
        """
        results = []
        total = len(papers)
        
        for i, paper in enumerate(papers, 1):
            self.logger.info(f"Processing paper {i}/{total}: {paper.title[:50]}...")
            
            try:
                result = await self.run_paper_pipeline(paper, scholar_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process paper: {e}")
                # 创建失败结果
                influence = self.influence_calculator.calculate(paper, None)
                fallback_report = self._generate_fallback_report(
                    paper, influence, {"error": str(e)}
                )
                results.append(
                    (
                        None,
                        influence,
                        {
                            "status": "failed",
                            "error": str(e),
                            "report_content": fallback_report,
                        },
                    )
                )
        
        return results
