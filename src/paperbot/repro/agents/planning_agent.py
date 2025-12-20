# repro/agents/planning_agent.py
"""
Planning Agent for Paper2Code pipeline.

Responsible for:
- Blueprint distillation from paper context
- Reproduction plan generation
- File structure planning
"""

import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResult, AgentStatus
from ..models import PaperContext, Blueprint, ReproductionPlan
from ..nodes import BlueprintDistillationNode, PlanningNode

logger = logging.getLogger(__name__)


class PlanningAgent(BaseAgent):
    """
    Agent responsible for planning the code reproduction.

    Pipeline:
    1. Distill paper context into Blueprint
    2. Generate reproduction plan from Blueprint
    3. Output file structure and dependencies

    Context Input:
        - paper_context: PaperContext

    Context Output:
        - blueprint: Blueprint
        - plan: ReproductionPlan
    """

    def __init__(self, **kwargs):
        super().__init__(name="PlanningAgent", **kwargs)
        self.blueprint_node = BlueprintDistillationNode()
        self.planning_node = PlanningNode()

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute planning pipeline."""
        paper_context = context.get("paper_context")
        if not paper_context:
            return AgentResult.failure("Missing paper_context in context")

        try:
            # Step 1: Distill Blueprint
            self.log("Distilling paper into Blueprint...")
            blueprint_result = await self.blueprint_node.run(paper_context)

            if not blueprint_result.success:
                return AgentResult.failure(
                    f"Blueprint distillation failed: {blueprint_result.error}"
                )

            blueprint = blueprint_result.data
            self.log(f"Blueprint created: {blueprint.architecture_type} architecture, {blueprint.domain} domain")

            # Step 2: Generate Plan
            self.log("Generating reproduction plan...")
            plan_result = await self.planning_node.run((paper_context, blueprint))

            if not plan_result.success:
                return AgentResult.failure(
                    f"Plan generation failed: {plan_result.error}"
                )

            plan = plan_result.data
            self.log(f"Plan created: {len(plan.file_structure)} files, {len(plan.dependencies)} dependencies")

            # Store in context for other agents
            context["blueprint"] = blueprint
            context["plan"] = plan

            return AgentResult.success(
                data={
                    "blueprint": blueprint,
                    "plan": plan,
                },
                metadata={
                    "architecture": blueprint.architecture_type,
                    "domain": blueprint.domain,
                    "num_files": len(plan.file_structure),
                    "num_dependencies": len(plan.dependencies),
                }
            )

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return AgentResult.failure(str(e))
