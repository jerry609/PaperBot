# repro/nodes/analysis_node.py
"""
Analysis Node for Paper2Code pipeline.
Phase 2: Extract implementation specifications from paper context.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, ReproductionPlan, ImplementationSpec

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class AnalysisNode(BaseNode[ImplementationSpec]):
    """
    Extract implementation specifications from paper context.
    
    Input: (PaperContext, ReproductionPlan)
    Output: ImplementationSpec
    """
    
    ANALYSIS_PROMPT = """Analyze this paper and extract implementation specifications.

Paper: {title}
Abstract: {abstract}
Method: {method}
Planned Components: {components}

For each component, provide:
1. Class/function name
2. Key methods or attributes
3. Input/output types

Output JSON:
{{
    "specs": [
        {{"name": "ModelName", "type": "class", "methods": ["forward", "train"], "description": "..."}},
        ...
    ],
    "hyperparameters": {{"learning_rate": 0.001, ...}},
    "data_format": "csv/json/..."
}}
"""

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(node_name="AnalysisNode", **kwargs)
        self.llm_client = llm_client
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is (PaperContext, ReproductionPlan)."""
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            return "Input must be (PaperContext, ReproductionPlan)"
        paper_context, plan = input_data
        if not isinstance(paper_context, PaperContext):
            return "First element must be PaperContext"
        if not isinstance(plan, ReproductionPlan):
            return "Second element must be ReproductionPlan"
        return None
    
    async def _execute(self, input_data: tuple, **kwargs) -> ImplementationSpec:
        """Extract implementation specifications."""
        paper_context, plan = input_data
        
        # Try LLM-based analysis
        if query and ClaudeAgentOptions:
            try:
                prompt = self.ANALYSIS_PROMPT.format(
                    title=paper_context.title,
                    abstract=paper_context.abstract or "",
                    method=paper_context.method_section or "",
                    components=", ".join(plan.components)
                )
                
                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=1500)
                )
                
                return self._parse_spec(result.response)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using fallback: {e}")
        
        # Fallback spec
        return self._fallback_spec(plan)
    
    def _parse_spec(self, response: str) -> ImplementationSpec:
        """Parse LLM response into ImplementationSpec."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return ImplementationSpec(
                    component_specs=data.get("specs", []),
                    hyperparameters=data.get("hyperparameters", {}),
                    data_format=data.get("data_format", "csv"),
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse spec JSON, using fallback")
        
        return self._fallback_spec(None)
    
    def _fallback_spec(self, plan: Optional[ReproductionPlan]) -> ImplementationSpec:
        """Generate fallback spec."""
        return ImplementationSpec(
            component_specs=[
                {"name": "Model", "type": "class", "methods": ["forward", "train"]},
                {"name": "DataLoader", "type": "class", "methods": ["load", "preprocess"]},
            ],
            hyperparameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
            data_format="csv",
        )
