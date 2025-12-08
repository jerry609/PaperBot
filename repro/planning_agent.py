# repro/planning_agent.py
"""
Planning Agent for Paper2Code-style reproduction.
Uses Claude Agent SDK to generate structured reproduction plans.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from repro.models import PaperContext, ReproductionPlan, ImplementationSpec

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None
    logger.warning("claude-agent-sdk not installed; PlanningAgent will use fallback.")


class PlanningAgent:
    """
    Phase 1 & 2 of Paper2Code pipeline:
    - Planning: Create file structure and component overview
    - Analysis: Extract implementation specifications
    """

    PLANNING_SYSTEM_PROMPT = """You are an expert ML engineer tasked with planning code reproduction from research papers.
Given a paper's abstract and method section, create a detailed plan for implementing the code.
Be specific about file structure, dependencies, and key components.
Output should be valid JSON that can be parsed programmatically."""

    ANALYSIS_SYSTEM_PROMPT = """You are an expert ML researcher analyzing paper implementation details.
Extract precise hyperparameters, model architecture details, and training configurations.
Be specific about shapes, layer types, optimizer settings, etc.
Output should be valid JSON that can be parsed programmatically."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get("model", "claude-3-5-sonnet-latest")

    async def generate_plan(self, paper_context: PaperContext) -> ReproductionPlan:
        """
        Phase 1: Generate high-level reproduction plan from paper context.
        """
        if query is None or ClaudeAgentOptions is None:
            return self._fallback_plan(paper_context)

        try:
            prompt = f"""Analyze this paper and create a reproduction plan.

{paper_context.to_prompt_context()}

Create a JSON plan with this structure:
{{
    "project_name": "short_name_for_project",
    "description": "Brief description of what the code will do",
    "file_structure": {{
        "main.py": "Entry point for training/inference",
        "model.py": "Model architecture definition",
        "data.py": "Data loading and preprocessing",
        "config.py": "Configuration and hyperparameters",
        "utils.py": "Utility functions",
        "requirements.txt": "Dependencies"
    }},
    "entry_point": "main.py",
    "dependencies": ["torch", "numpy", "..."],
    "key_components": ["Model", "DataLoader", "Trainer"],
    "estimated_complexity": "low|medium|high"
}}

Return ONLY the JSON, no explanation."""

            opts = ClaudeAgentOptions(
                system_prompt=self.PLANNING_SYSTEM_PROMPT,
                model=self.model,
            )

            response_text = ""
            async for msg in query(prompt=prompt, options=opts):
                content = getattr(msg, "content", None)
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None)
                        if text:
                            response_text += text
                elif isinstance(content, str):
                    response_text += content

            return self._parse_plan(response_text, paper_context)

        except Exception as e:
            logger.warning(f"Planning failed, using fallback: {e}")
            return self._fallback_plan(paper_context)

    async def generate_spec(
        self, paper_context: PaperContext, plan: ReproductionPlan
    ) -> ImplementationSpec:
        """
        Phase 2: Generate detailed implementation specifications.
        """
        if query is None or ClaudeAgentOptions is None:
            return self._fallback_spec()

        try:
            prompt = f"""Analyze this paper for implementation details.

{paper_context.to_prompt_context()}

{plan.to_prompt_context()}

Extract implementation specifications as JSON:
{{
    "model_type": "transformer|cnn|mlp|rnn|etc",
    "layers": [
        {{"type": "Linear", "in_features": 768, "out_features": 256}},
        {{"type": "ReLU"}},
        ...
    ],
    "optimizer": "adam|sgd|adamw",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "input_shape": [batch, seq_len, hidden],
    "output_shape": [batch, num_classes],
    "data_format": "csv|json|numpy|pytorch",
    "extra_params": {{}}
}}

Infer values from the paper. If not specified, use reasonable defaults.
Return ONLY the JSON."""

            opts = ClaudeAgentOptions(
                system_prompt=self.ANALYSIS_SYSTEM_PROMPT,
                model=self.model,
            )

            response_text = ""
            async for msg in query(prompt=prompt, options=opts):
                content = getattr(msg, "content", None)
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None)
                        if text:
                            response_text += text
                elif isinstance(content, str):
                    response_text += content

            return self._parse_spec(response_text)

        except Exception as e:
            logger.warning(f"Spec generation failed, using fallback: {e}")
            return self._fallback_spec()

    def _parse_plan(self, response: str, paper_context: PaperContext) -> ReproductionPlan:
        """Parse LLM response into ReproductionPlan."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return ReproductionPlan(
                    project_name=data.get("project_name", "repro_project"),
                    description=data.get("description", ""),
                    file_structure=data.get("file_structure", {}),
                    entry_point=data.get("entry_point", "main.py"),
                    dependencies=data.get("dependencies", []),
                    key_components=data.get("key_components", []),
                    estimated_complexity=data.get("estimated_complexity", "medium")
                )
        except json.JSONDecodeError:
            pass
        return self._fallback_plan(paper_context)

    def _parse_spec(self, response: str) -> ImplementationSpec:
        """Parse LLM response into ImplementationSpec."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return ImplementationSpec(
                    model_type=data.get("model_type", ""),
                    layers=data.get("layers", []),
                    optimizer=data.get("optimizer", "adam"),
                    learning_rate=data.get("learning_rate", 1e-4),
                    batch_size=data.get("batch_size", 32),
                    epochs=data.get("epochs", 10),
                    input_shape=tuple(data["input_shape"]) if data.get("input_shape") else None,
                    output_shape=tuple(data["output_shape"]) if data.get("output_shape") else None,
                    data_format=data.get("data_format", ""),
                    extra_params=data.get("extra_params", {})
                )
        except json.JSONDecodeError:
            pass
        return self._fallback_spec()

    def _fallback_plan(self, paper_context: PaperContext) -> ReproductionPlan:
        """Fallback plan when LLM is unavailable."""
        return ReproductionPlan(
            project_name=paper_context.title[:30].lower().replace(" ", "_"),
            description=f"Reproduction of: {paper_context.title}",
            file_structure={
                "main.py": "Entry point",
                "model.py": "Model definition",
                "data.py": "Data handling",
                "config.py": "Configuration",
                "requirements.txt": "Dependencies"
            },
            entry_point="main.py",
            dependencies=["torch", "numpy", "tqdm"],
            key_components=["Model", "DataLoader", "Trainer"],
            estimated_complexity="medium"
        )

    def _fallback_spec(self) -> ImplementationSpec:
        """Fallback spec when LLM is unavailable."""
        return ImplementationSpec(
            model_type="unknown",
            optimizer="adam",
            learning_rate=1e-4,
            batch_size=32,
            epochs=10
        )
