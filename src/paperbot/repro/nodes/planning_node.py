# repro/nodes/planning_node.py
"""
Planning Node for Paper2Code pipeline.
Phase 1: Generate high-level reproduction plan from paper context.
"""

import logging
import json
from typing import Dict, Any, Optional
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, ReproductionPlan

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class PlanningNode(BaseNode[ReproductionPlan]):
    """
    Generate high-level reproduction plan from paper context.
    
    Input: PaperContext
    Output: ReproductionPlan
    """
    
    PLANNING_PROMPT = """You are an expert software engineer helping to reproduce a research paper.

Given the paper context below, generate a reproduction plan with:
1. A list of files to create (with purposes)
2. Key components to implement
3. Dependencies needed

Paper Title: {title}
Abstract: {abstract}
Method Section: {method}

Output a JSON with this structure:
{{
    "files": [
        {{"path": "main.py", "purpose": "Entry point"}},
        ...
    ],
    "components": ["component1", "component2"],
    "dependencies": ["numpy", "torch", ...]
}}
"""

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(node_name="PlanningNode", **kwargs)
        self.llm_client = llm_client
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is a PaperContext."""
        if not isinstance(input_data, PaperContext):
            return "Input must be a PaperContext"
        if not input_data.title:
            return "PaperContext must have a title"
        return None
    
    async def _execute(self, input_data: PaperContext, **kwargs) -> ReproductionPlan:
        """Generate reproduction plan."""
        paper_context = input_data
        
        # Try LLM-based planning
        if query and ClaudeAgentOptions:
            try:
                prompt = self.PLANNING_PROMPT.format(
                    title=paper_context.title,
                    abstract=paper_context.abstract or "",
                    method=paper_context.method_section or ""
                )
                
                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=1000)
                )
                
                return self._parse_plan(result.response, paper_context)
            except Exception as e:
                logger.warning(f"LLM planning failed, using fallback: {e}")
        
        # Fallback plan
        return self._fallback_plan(paper_context)
    
    def _parse_plan(self, response: str, paper_context: PaperContext) -> ReproductionPlan:
        """Parse LLM response into ReproductionPlan."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                # Convert files list to file_structure dict
                file_structure = {
                    f.get("path", ""): f.get("purpose", "")
                    for f in data.get("files", [])
                }
                return ReproductionPlan(
                    project_name=paper_context.title,
                    description=paper_context.abstract or "Paper reproduction",
                    file_structure=file_structure,
                    key_components=data.get("components", []),
                    dependencies=data.get("dependencies", []),
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, using fallback")
        
        return self._fallback_plan(paper_context)
    
    def _fallback_plan(self, paper_context: PaperContext) -> ReproductionPlan:
        """Generate fallback plan when LLM is unavailable."""
        return ReproductionPlan(
            project_name=paper_context.title,
            description=paper_context.abstract or "Paper reproduction",
            file_structure={
                "main.py": "Main entry point",
                "model.py": "Model implementation",
                "data.py": "Data loading",
                "config.py": "Configuration",
                "utils.py": "Utility functions",
                "requirements.txt": "Dependencies",
            },
            key_components=["Model", "DataLoader", "Trainer"],
            dependencies=["numpy", "torch", "transformers"],
        )

