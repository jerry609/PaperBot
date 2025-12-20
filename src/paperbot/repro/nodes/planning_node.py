# repro/nodes/planning_node.py
"""
Planning Node for Paper2Code pipeline.
Phase 1: Generate high-level reproduction plan from paper context.

Enhanced with Blueprint support for more efficient context usage.
"""

import logging
import json
from typing import Dict, Any, Optional, Union, List
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, ReproductionPlan, Blueprint

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class PlanningNode(BaseNode[ReproductionPlan]):
    """
    Generate high-level reproduction plan from paper context.

    Enhanced to support both PaperContext and Blueprint inputs.
    When Blueprint is provided, uses compressed context for efficiency.

    Input: PaperContext or (PaperContext, Blueprint)
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

    BLUEPRINT_PLANNING_PROMPT = """You are an expert software engineer helping to reproduce a research paper.

Given the distilled paper blueprint below, generate a detailed reproduction plan with:
1. A list of files to create (matching the module hierarchy)
2. Key components to implement
3. Dependencies needed (based on framework hints)

{blueprint_context}

Output a JSON with this structure:
{{
    "files": [
        {{"path": "main.py", "purpose": "Entry point"}},
        {{"path": "model.py", "purpose": "Model implementation"}},
        {{"path": "config.py", "purpose": "Configuration"}},
        ...
    ],
    "components": ["component1", "component2"],
    "dependencies": ["numpy", "torch", ...]
}}

Ensure the file structure matches the module hierarchy from the blueprint.
Include files for each major component identified.
"""

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(node_name="PlanningNode", **kwargs)
        self.llm_client = llm_client

    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is a PaperContext or (PaperContext, Blueprint) tuple."""
        if isinstance(input_data, tuple):
            if len(input_data) < 1:
                return "Tuple must contain at least PaperContext"
            if not isinstance(input_data[0], PaperContext):
                return "First element must be PaperContext"
            if len(input_data) > 1 and input_data[1] is not None:
                if not isinstance(input_data[1], Blueprint):
                    return "Second element must be Blueprint or None"
            return None
        if not isinstance(input_data, PaperContext):
            return "Input must be a PaperContext or (PaperContext, Blueprint) tuple"
        if not input_data.title:
            return "PaperContext must have a title"
        return None

    async def _execute(self, input_data: Union[PaperContext, tuple], **kwargs) -> ReproductionPlan:
        """Generate reproduction plan."""
        # Parse input
        if isinstance(input_data, tuple):
            paper_context = input_data[0]
            blueprint = input_data[1] if len(input_data) > 1 else None
        else:
            paper_context = input_data
            blueprint = None

        # Try LLM-based planning with Blueprint if available
        if query and ClaudeAgentOptions:
            try:
                if blueprint:
                    # Use compressed blueprint context
                    prompt = self.BLUEPRINT_PLANNING_PROMPT.format(
                        blueprint_context=blueprint.to_compressed_context(max_tokens=2000)
                    )
                else:
                    # Fallback to raw paper context
                    prompt = self.PLANNING_PROMPT.format(
                        title=paper_context.title,
                        abstract=paper_context.abstract or "",
                        method=paper_context.method_section or ""
                    )

                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=1000)
                )

                return self._parse_plan(result.response, paper_context, blueprint)
            except Exception as e:
                logger.warning(f"LLM planning failed, using fallback: {e}")

        # Fallback plan
        return self._fallback_plan(paper_context, blueprint)
    
    def _parse_plan(
        self,
        response: str,
        paper_context: PaperContext,
        blueprint: Optional[Blueprint] = None
    ) -> ReproductionPlan:
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

                # Enhance dependencies with framework hints from blueprint
                dependencies = data.get("dependencies", [])
                if blueprint and blueprint.framework_hints:
                    for hint in blueprint.framework_hints:
                        if hint == "pytorch" and "torch" not in dependencies:
                            dependencies.append("torch")
                        elif hint == "tensorflow" and "tensorflow" not in dependencies:
                            dependencies.append("tensorflow")
                        elif hint == "transformers" and "transformers" not in dependencies:
                            dependencies.append("transformers")

                return ReproductionPlan(
                    project_name=paper_context.title,
                    description=paper_context.abstract or "Paper reproduction",
                    file_structure=file_structure,
                    key_components=data.get("components", []),
                    dependencies=dependencies,
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, using fallback")

        return self._fallback_plan(paper_context, blueprint)

    def _fallback_plan(
        self,
        paper_context: PaperContext,
        blueprint: Optional[Blueprint] = None
    ) -> ReproductionPlan:
        """Generate fallback plan when LLM is unavailable."""
        # Build file structure from blueprint if available
        if blueprint and blueprint.module_hierarchy:
            file_structure = self._build_file_structure_from_blueprint(blueprint)
            key_components = self._extract_components_from_blueprint(blueprint)
            dependencies = self._infer_dependencies_from_blueprint(blueprint)
        else:
            file_structure = {
                "main.py": "Main entry point",
                "model.py": "Model implementation",
                "data.py": "Data loading",
                "config.py": "Configuration",
                "utils.py": "Utility functions",
                "requirements.txt": "Dependencies",
            }
            key_components = ["Model", "DataLoader", "Trainer"]
            dependencies = ["numpy", "torch", "transformers"]

        return ReproductionPlan(
            project_name=paper_context.title,
            description=paper_context.abstract or "Paper reproduction",
            file_structure=file_structure,
            key_components=key_components,
            dependencies=dependencies,
        )

    def _build_file_structure_from_blueprint(self, blueprint: Blueprint) -> Dict[str, str]:
        """Build file structure based on blueprint module hierarchy."""
        file_structure = {
            "main.py": "Main entry point",
            "config.py": "Configuration and hyperparameters",
            "requirements.txt": "Dependencies",
        }

        # Add files based on module hierarchy
        for parent, children in blueprint.module_hierarchy.items():
            if parent == "model":
                file_structure["model.py"] = "Main model implementation"
                for child in children:
                    if child not in ["encoder", "decoder"]:  # Keep these in model.py
                        file_structure[f"{child}.py"] = f"{child.replace('_', ' ').title()} module"
            elif parent in ["encoder", "decoder"]:
                # These go into model.py as components
                pass
            else:
                file_structure[f"{parent}.py"] = f"{parent.replace('_', ' ').title()} implementation"

        # Add training-related files based on optimization strategy
        if blueprint.optimization_strategy:
            file_structure["trainer.py"] = "Training loop and optimization"

        # Add data loading if we have I/O specs
        if blueprint.input_output_spec:
            file_structure["data.py"] = "Data loading and preprocessing"

        # Add evaluation if we have loss functions
        if blueprint.loss_functions:
            file_structure["losses.py"] = "Loss function implementations"

        return file_structure

    def _extract_components_from_blueprint(self, blueprint: Blueprint) -> List[str]:
        """Extract key components from blueprint."""
        components = []

        # Add architecture-specific components
        arch_components = {
            "transformer": ["Encoder", "Decoder", "Attention", "FFN"],
            "cnn": ["ConvBlock", "Backbone", "Head"],
            "gnn": ["MessagePassing", "Aggregation", "Readout"],
            "rnn": ["RNNCell", "Encoder", "Decoder"],
            "diffusion": ["UNet", "NoiseScheduler", "Sampler"],
        }

        if blueprint.architecture_type in arch_components:
            components.extend(arch_components[blueprint.architecture_type])
        else:
            components.extend(["Model", "DataLoader", "Trainer"])

        # Add algorithm-based components
        for algo in blueprint.core_algorithms:
            components.append(algo.name.replace(" ", ""))

        return list(set(components))[:10]  # Limit to 10 components

    def _infer_dependencies_from_blueprint(self, blueprint: Blueprint) -> List[str]:
        """Infer dependencies from blueprint."""
        dependencies = ["numpy"]

        # Framework-based dependencies
        for framework in blueprint.framework_hints:
            if framework == "pytorch":
                dependencies.extend(["torch", "torchvision"])
            elif framework == "tensorflow":
                dependencies.extend(["tensorflow"])
            elif framework == "transformers":
                dependencies.extend(["transformers", "tokenizers"])
            elif framework == "jax":
                dependencies.extend(["jax", "flax", "optax"])

        # Domain-based dependencies
        domain_deps = {
            "nlp": ["transformers", "tokenizers", "datasets"],
            "cv": ["torchvision", "opencv-python", "albumentations"],
            "audio": ["torchaudio", "librosa"],
            "rl": ["gymnasium", "stable-baselines3"],
        }

        if blueprint.domain in domain_deps:
            for dep in domain_deps[blueprint.domain]:
                if dep not in dependencies:
                    dependencies.append(dep)

        # Common utilities
        dependencies.extend(["tqdm", "pyyaml", "tensorboard"])

        return list(set(dependencies))

