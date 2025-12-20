# repro/agents/coding_agent.py
"""
Coding Agent for Paper2Code pipeline.

Responsible for:
- Code generation with CodeMemory context
- RAG-enhanced generation with code patterns
- Dependency-aware file ordering
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from .base_agent import BaseAgent, AgentResult, AgentStatus
from ..models import PaperContext, Blueprint, ReproductionPlan, ImplementationSpec
from ..nodes import GenerationNode, AnalysisNode
from ..memory import CodeMemory
from ..rag import CodeKnowledgeBase

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """
    Agent responsible for code generation.

    Pipeline:
    1. Analyze paper for implementation details
    2. Generate code files in dependency order
    3. Use CodeMemory for cross-file context
    4. Use CodeRAG for pattern injection

    Context Input:
        - paper_context: PaperContext
        - blueprint: Blueprint
        - plan: ReproductionPlan

    Context Output:
        - generated_files: Dict[str, str]
        - code_memory: CodeMemory
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        max_context_tokens: int = 8000,
        use_rag: bool = True,
        **kwargs
    ):
        super().__init__(name="CodingAgent", **kwargs)
        self.output_dir = output_dir
        self.analysis_node = AnalysisNode()
        self.generation_node = GenerationNode(
            max_context_tokens=max_context_tokens,
            use_rag=use_rag
        )

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute code generation pipeline."""
        paper_context = context.get("paper_context")
        blueprint = context.get("blueprint")
        plan = context.get("plan")

        if not paper_context:
            return AgentResult.failure("Missing paper_context in context")
        if not plan:
            return AgentResult.failure("Missing plan in context - run PlanningAgent first")

        try:
            # Step 1: Analyze for implementation spec
            self.log("Analyzing paper for implementation details...")
            analysis_result = await self.analysis_node.run(paper_context)

            if not analysis_result.success:
                self.log(f"Analysis warning: {analysis_result.error}, using defaults")
                spec = ImplementationSpec()
            else:
                spec = analysis_result.data

            self.log(f"Spec: {spec.model_type} model, lr={spec.learning_rate}, batch_size={spec.batch_size}")

            # Step 2: Generate code files
            self.log(f"Generating {len(plan.file_structure)} code files...")

            if blueprint:
                gen_input = (paper_context, plan, spec, blueprint)
            else:
                gen_input = (paper_context, plan, spec)

            gen_result = await self.generation_node.run(gen_input)

            if not gen_result.success:
                return AgentResult.failure(
                    f"Code generation failed: {gen_result.error}"
                )

            generated_files = gen_result.data
            self.log(f"Generated {len(generated_files)} files")

            # Step 3: Write files to disk if output_dir specified
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                for filepath, content in generated_files.items():
                    file_path = self.output_dir / filepath
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
                self.log(f"Wrote files to {self.output_dir}")

            # Store in context
            context["generated_files"] = generated_files
            context["spec"] = spec
            context["code_memory"] = self.generation_node.memory

            return AgentResult.success(
                data={
                    "generated_files": generated_files,
                    "spec": spec,
                },
                metadata={
                    "num_files": len(generated_files),
                    "total_chars": sum(len(c) for c in generated_files.values()),
                    "files": list(generated_files.keys()),
                }
            )

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return AgentResult.failure(str(e))
