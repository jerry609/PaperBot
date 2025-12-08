# src/paperbot/agents/review/agent.py
"""
ReviewerAgent: Simulates human-like peer review of scientific papers.
Inspired by DeepReview (Zhu et al., 2024) - Deep Thinking for Paper Review.

This agent performs:
1. Preliminary Screening: Quick assessment of scope and quality.
2. Deep Critique: Detailed evaluation of contributions, methodology, and reproducibility.
3. Final Decision: Structured verdict with confidence score.

Uses Mixin pattern:
- TextParsingMixin: Text extraction utilities
"""

from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from ..base import BaseAgent
from ..mixins import TextParsingMixin
from ..mixins.json_parser import JSONParseError
from paperbot.core.report_engine import ReportEngine, ReportEngineConfig, ReportResult


class ReviewerAgent(BaseAgent, TextParsingMixin):
    """
    Performs deep peer-review analysis of a paper.
    
    Outputs a structured review containing:
    - Summary
    - Strengths (list)
    - Weaknesses (list)
    - Novelty Assessment (with ScholarEval-style comparison)
    - Reproducibility Assessment
    - Overall Score (1-10) and Decision (Accept/Reject/Borderline)
    """

    REVIEW_SYSTEM_PROMPT = """You are an expert peer reviewer for top-tier AI/ML conferences (NeurIPS, ICML, ACL, USENIX Security).
Your reviews are known for being rigorous, fair, and constructive.
Always provide specific evidence from the paper to support your claims.
Be direct but respectful. Prioritize scientific correctness and reproducibility."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._report_engine: Optional[ReportEngine] = None

    def _validate_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Validate that title and abstract are provided."""
        if not kwargs.get("title") or not kwargs.get("abstract"):
            return {"status": "error", "error": "ReviewerAgent requires 'title' and 'abstract' arguments."}
        return None

    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Core execution: perform paper review."""
        title = kwargs.get("title", "")
        abstract = kwargs.get("abstract", "")
        full_text = kwargs.get("full_text")  # Optional
        related_work = kwargs.get("related_work", [])  # Optional list of similar papers
        
        # Phase 1: Preliminary Screening
        preliminary = await self._preliminary_screening(title, abstract)
        
        # Phase 2: Deep Critique (if passes initial screen)
        if preliminary.get("proceed_to_full_review", True):
            deep_critique = await self._deep_critique(title, abstract, full_text)
        else:
            deep_critique = {
                "strengths": [],
                "weaknesses": ["Did not pass preliminary screening."],
                "methodology_score": 0,
                "reproducibility_score": 0
            }
        
        # Phase 3: Novelty Assessment (ScholarEval style)
        novelty = await self._assess_novelty(title, abstract, related_work)
        
        # Phase 4: Final Decision
        try:
            structured = await self._generate_final_review(
                title, abstract, preliminary, deep_critique, novelty
            )
            return self._parse_structured(structured)
        except Exception as exc:
            return self._on_failure(exc)

    def _post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result = super()._post_process(result)
        rendered: Dict[str, Any] = {}
        engine = self._get_report_engine()
        if engine:
            try:
                ctx = {
                    "review": result,
                }
                compare_items = result.get("compare_items")
                summary = result.get("summary", "")
                topic = result.get("paper_title", "Review")
                re_res: ReportResult = engine.generate(
                    topic=topic,
                    summary=summary,
                    sections_context=ctx,
                    task_id=topic,
                    enable_pdf=engine.config.pdf_enabled,
                    compare_items=compare_items,
                )
                rendered = {
                    "html": str(re_res.html_path) if re_res.html_path else None,
                    "pdf": str(re_res.pdf_path) if re_res.pdf_path else None,
                    "ir": str(re_res.ir_path) if re_res.ir_path else None,
                }
            except Exception as exc:  # pragma: no cover - 渲染失败时降级
                self.logger.warning(f"ReportEngine 渲染失败: {exc}")
                rendered = {"error": str(exc)}
        result["report_engine"] = rendered
        return result

    def _parse_structured(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """可覆写的结构化解析钩子，当前直接返回."""
        return raw

    # ==================== Helpers ====================
    def _get_report_engine(self) -> Optional[ReportEngine]:
        if hasattr(self, "_report_engine") and self._report_engine:
            return self._report_engine
        cfg_dict = self.config.get("report_engine", {})
        if not cfg_dict or not cfg_dict.get("enabled"):
            return None
        api_key = cfg_dict.get("api_key") or self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        self._report_engine = ReportEngine(
            ReportEngineConfig(
                enabled=True,
                api_key=api_key,
                model=cfg_dict.get("model", "gpt-4o-mini"),
                base_url=cfg_dict.get("base_url"),
                output_dir=Path(cfg_dict.get("output_dir", "output/reports")),
                template_dir=Path(cfg_dict.get("template_dir", "core/report_engine/templates")),
                pdf_enabled=cfg_dict.get("pdf_enabled", True),
                max_words=cfg_dict.get("max_words", 4000),
            )
        )
        return self._report_engine

    async def _preliminary_screening(self, title: str, abstract: str) -> Dict[str, Any]:
        """Quick initial assessment of paper quality and scope."""
        prompt = f"""Perform a preliminary screening of this paper.

**Title:** {title}

**Abstract:** {abstract}

Provide:
1. Is this paper within scope for a top-tier venue? (yes/no)
2. Does the abstract clearly state the problem, contribution, and results? (yes/no)
3. Any immediate red flags (e.g., overclaiming, unclear methodology)? (list or "None")
4. Should this paper proceed to full review? (yes/no)

Respond in JSON format with keys: in_scope, clear_abstract, red_flags, proceed_to_full_review"""

        response = await self.ask_claude(prompt, system=self.REVIEW_SYSTEM_PROMPT, max_tokens=500)
        
        import json
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
        
        return {
            "in_scope": True,
            "clear_abstract": True,
            "red_flags": [],
            "proceed_to_full_review": True,
            "raw_response": response
        }

    async def _deep_critique(
        self, title: str, abstract: str, full_text: Optional[str]
    ) -> Dict[str, Any]:
        """In-depth analysis of methodology, contributions, and reproducibility."""
        context = f"**Title:** {title}\n\n**Abstract:** {abstract}"
        if full_text:
            context += f"\n\n**Key Sections:**\n{full_text[:4000]}"
        
        prompt = f"""Analyze this paper in depth.

{context}

Provide:
1. **Strengths** (3-5 bullet points): What does this paper do well? Be specific.
2. **Weaknesses** (3-5 bullet points): What are the limitations or concerns? Be specific.
3. **Methodology Score** (1-10): Is the methodology sound and well-executed?
4. **Reproducibility Score** (1-10): Is there sufficient detail to reproduce the work? Is code available?
5. **Key Questions for Authors** (2-3): What clarifications would improve the paper?

Be rigorous but fair. Cite specific aspects of the paper in your critique."""

        response = await self.ask_claude(prompt, system=self.REVIEW_SYSTEM_PROMPT, max_tokens=1500)
        
        return {
            "strengths": self.extract_bullet_points(response, "Strengths"),
            "weaknesses": self.extract_bullet_points(response, "Weaknesses"),
            "methodology_score": self.extract_score(response, "Methodology"),
            "reproducibility_score": self.extract_score(response, "Reproducibility"),
            "author_questions": self.extract_bullet_points(response, "Questions"),
            "raw_response": response
        }

    async def _assess_novelty(
        self, title: str, abstract: str, related_work: List[str]
    ) -> Dict[str, Any]:
        """Assess the novelty of the paper (ScholarEval inspired)."""
        related_str = "\n".join([f"- {w}" for w in related_work]) if related_work else "No related work provided."
        
        prompt = f"""Assess the novelty of this paper.

**Paper Title:** {title}
**Abstract:** {abstract}

**Potentially Related Work:**
{related_str}

Evaluate:
1. **Novelty Type**: Is this (a) entirely new problem, (b) new method for existing problem, (c) incremental improvement, or (d) application/engineering work?
2. **Novelty Score** (1-10): How novel is the contribution?
3. **Comparison to Related Work**: How does this paper differentiate from the listed related work?
4. **Potential Prior Art Concerns**: Any work that might anticipate this contribution?

Be critical but recognize that incremental work can still be valuable if well-executed."""

        response = await self.ask_claude(prompt, system=self.REVIEW_SYSTEM_PROMPT, max_tokens=800)
        
        return {
            "novelty_score": self.extract_score(response, "Novelty"),
            "novelty_type": self.extract_field(response, "Novelty Type"),
            "comparison": self.extract_field(response, "Comparison"),
            "prior_art_concerns": self.extract_field(response, "Prior Art"),
            "raw_response": response
        }

    async def _generate_final_review(
        self,
        title: str,
        abstract: str,
        preliminary: Dict[str, Any],
        deep_critique: Dict[str, Any],
        novelty: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize all analyses into a final structured review."""
        methodology = deep_critique.get("methodology_score", 5)
        reproducibility = deep_critique.get("reproducibility_score", 5)
        novelty_score = novelty.get("novelty_score", 5)
        
        # Weighted average (methodology 40%, novelty 35%, reproducibility 25%)
        overall_score = (methodology * 0.4) + (novelty_score * 0.35) + (reproducibility * 0.25)
        
        # Decision thresholds
        if overall_score >= 7:
            decision = "Accept"
        elif overall_score >= 5:
            decision = "Borderline"
        else:
            decision = "Reject"
        
        # Generate summary
        summary_prompt = f"""Write a 2-3 sentence summary of this paper for a meta-reviewer.

**Title:** {title}
**Abstract:** {abstract}
**Key Strengths:** {', '.join(deep_critique.get('strengths', [])[:2])}
**Key Weaknesses:** {', '.join(deep_critique.get('weaknesses', [])[:2])}

Be concise and objective."""

        summary = await self.ask_claude(summary_prompt, max_tokens=200)
        
        return {
            "paper_title": title,
            "summary": summary.strip(),
            "preliminary_screening": preliminary,
            "strengths": deep_critique.get("strengths", []),
            "weaknesses": deep_critique.get("weaknesses", []),
            "author_questions": deep_critique.get("author_questions", []),
            "scores": {
                "methodology": methodology,
                "reproducibility": reproducibility,
                "novelty": novelty_score,
                "overall": round(overall_score, 1)
            },
            "novelty_assessment": novelty,
            "decision": decision,
            "confidence": "High" if len(deep_critique.get("strengths", [])) > 2 else "Medium"
        }

