# repro/nodes/environment_node.py
"""
Environment Inference Node for Paper2Code pipeline.
Phase 1.5: Infer execution environment from paper context.

Analyzes paper metadata and code snippets to determine:
- Python version
- PyTorch/TensorFlow versions
- CUDA requirements
- Base Docker image
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, EnvironmentSpec

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


# Framework version mapping based on paper publication year
# Format: (year, (pytorch_version, tensorflow_version, python_version, cuda_version))
YEAR_TO_VERSIONS: Dict[int, Tuple[str, str, str, str]] = {
    2024: ("2.2.0", "2.15.0", "3.11", "12.1"),
    2023: ("2.0.0", "2.12.0", "3.10", "11.8"),
    2022: ("1.13.0", "2.10.0", "3.10", "11.7"),
    2021: ("1.10.0", "2.6.0", "3.9", "11.3"),
    2020: ("1.7.0", "2.4.0", "3.8", "11.0"),
    2019: ("1.4.0", "2.0.0", "3.7", "10.1"),
    2018: ("1.0.0", "1.12.0", "3.6", "9.2"),
    2017: ("0.4.0", "1.4.0", "3.6", "8.0"),
}

# Common import patterns to detect framework usage
PYTORCH_PATTERNS = [
    r'\bimport\s+torch\b',
    r'\bfrom\s+torch\b',
    r'\btorch\.nn\b',
    r'\btorch\.optim\b',
    r'\bnn\.Module\b',
]

TENSORFLOW_PATTERNS = [
    r'\bimport\s+tensorflow\b',
    r'\bfrom\s+tensorflow\b',
    r'\btf\.keras\b',
    r'\bkeras\.\b',
    r'\btf\.nn\b',
]

JAX_PATTERNS = [
    r'\bimport\s+jax\b',
    r'\bfrom\s+jax\b',
    r'\bjax\.numpy\b',
]


class EnvironmentInferenceNode(BaseNode[EnvironmentSpec]):
    """
    Infer execution environment from paper context.
    
    Input: PaperContext
    Output: EnvironmentSpec
    
    Inference strategies:
    1. Year-based: Use paper publication year to estimate framework versions
    2. Code-based: Parse code snippets for import statements
    3. Explicit: Use hyperparameters if framework versions are mentioned
    4. LLM-based: Ask LLM to infer from method section if available
    """
    
    INFERENCE_PROMPT = """Analyze this paper context and infer the execution environment.

Paper: {title}
Year: {year}
Abstract: {abstract}
Method Section: {method}
Code Snippets: {code_snippets}

Based on the paper's content, determine:
1. Which deep learning framework is used (PyTorch/TensorFlow/JAX/None)?
2. What Python version is appropriate?
3. Are there any specific library requirements mentioned?

Output JSON:
{{
    "framework": "pytorch|tensorflow|jax|none",
    "framework_version": "x.y.z or null",
    "python_version": "3.x",
    "cuda_required": true|false,
    "cuda_version": "x.y or null",
    "additional_packages": ["package1", "package2"],
    "reasoning": "Brief explanation"
}}
"""

    def __init__(self, llm_client=None, prefer_conda: bool = False, **kwargs):
        super().__init__(node_name="EnvironmentInferenceNode", **kwargs)
        self.llm_client = llm_client
        self.prefer_conda = prefer_conda
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is PaperContext."""
        if not isinstance(input_data, PaperContext):
            return "Input must be PaperContext"
        return None
    
    async def _execute(self, input_data: PaperContext, **kwargs) -> EnvironmentSpec:
        """Infer execution environment from paper context."""
        paper_context = input_data
        
        # Strategy 1: Detect framework from code snippets
        framework, code_based = self._detect_framework_from_code(paper_context)
        
        # Strategy 2: Get version estimates from year
        year = self._extract_year(paper_context)
        year_versions = YEAR_TO_VERSIONS.get(year, YEAR_TO_VERSIONS[2023])
        pytorch_ver, tf_ver, python_ver, cuda_ver = year_versions
        
        # Strategy 3: Try LLM-based inference for better accuracy
        llm_spec = await self._llm_inference(paper_context, year)
        
        # Merge strategies
        spec = self._merge_inference_results(
            framework, code_based, year, year_versions, llm_spec, paper_context
        )
        
        # Generate environment files
        spec.generate_dockerfile()
        if self.prefer_conda:
            spec.generate_conda_yaml()
        
        return spec
    
    def _detect_framework_from_code(self, ctx: PaperContext) -> Tuple[str, bool]:
        """Detect ML framework from code snippets and method section."""
        text = " ".join([
            ctx.method_section or "",
            " ".join(ctx.algorithm_blocks),
            ctx.abstract,
        ])
        
        pytorch_score = sum(1 for p in PYTORCH_PATTERNS if re.search(p, text))
        tf_score = sum(1 for p in TENSORFLOW_PATTERNS if re.search(p, text))
        jax_score = sum(1 for p in JAX_PATTERNS if re.search(p, text))
        
        if pytorch_score > tf_score and pytorch_score > jax_score:
            return "pytorch", True
        elif tf_score > pytorch_score and tf_score > jax_score:
            return "tensorflow", True
        elif jax_score > 0:
            return "jax", True
        
        return "pytorch", False  # Default to PyTorch
    
    def _extract_year(self, ctx: PaperContext) -> int:
        """Extract publication year from paper context."""
        # Try to find year in title or abstract
        current_year = datetime.now().year
        
        # Check hyperparameters for explicit year
        if "year" in ctx.hyperparameters:
            try:
                return int(ctx.hyperparameters["year"])
            except (ValueError, TypeError):
                pass
        
        # Try to extract from title/abstract
        year_match = re.search(r'\b(20[12]\d)\b', ctx.title + " " + ctx.abstract)
        if year_match:
            return int(year_match.group(1))
        
        # Default to current year
        return current_year
    
    async def _llm_inference(
        self, ctx: PaperContext, year: int
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to infer environment from paper content."""
        if not query or not ClaudeAgentOptions:
            return None
        
        try:
            prompt = self.INFERENCE_PROMPT.format(
                title=ctx.title,
                year=year,
                abstract=ctx.abstract[:500],
                method=ctx.method_section[:1000] if ctx.method_section else "Not provided",
                code_snippets="\n".join(ctx.algorithm_blocks[:2]),
            )
            
            result = query(
                prompt=prompt,
                options=ClaudeAgentOptions(max_tokens=500)
            )
            
            return self._parse_llm_response(result.response)
        except Exception as e:
            logger.warning(f"LLM inference failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        import json
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None
    
    def _merge_inference_results(
        self,
        framework: str,
        code_based: bool,
        year: int,
        year_versions: Tuple[str, str, str, str],
        llm_spec: Optional[Dict[str, Any]],
        ctx: PaperContext,
    ) -> EnvironmentSpec:
        """Merge multiple inference strategies into final spec."""
        pytorch_ver, tf_ver, python_ver, cuda_ver = year_versions
        
        spec = EnvironmentSpec(
            python_version=python_ver,
            inferred_from="year" if not code_based else "code_snippets",
        )
        
        # Apply LLM results if available
        if llm_spec:
            spec.inferred_from = "llm+" + spec.inferred_from
            if llm_spec.get("framework_version"):
                if framework == "pytorch":
                    pytorch_ver = llm_spec["framework_version"]
                elif framework == "tensorflow":
                    tf_ver = llm_spec["framework_version"]
            if llm_spec.get("python_version"):
                spec.python_version = llm_spec["python_version"]
            if llm_spec.get("cuda_version"):
                cuda_ver = llm_spec["cuda_version"]
            if llm_spec.get("additional_packages"):
                spec.pip_requirements.extend(llm_spec["additional_packages"])
        
        # Set framework versions
        if framework == "pytorch":
            spec.pytorch_version = pytorch_ver
            spec.base_image = f"pytorch/pytorch:{pytorch_ver}-cuda{cuda_ver.replace('.', '')[:3]}-cudnn8-runtime"
            spec.pip_requirements.extend(["numpy", "tqdm", "tensorboard"])
        elif framework == "tensorflow":
            spec.tensorflow_version = tf_ver
            spec.base_image = f"tensorflow/tensorflow:{tf_ver}-gpu"
            spec.pip_requirements.extend(["numpy", "keras"])
        elif framework == "jax":
            spec.pip_requirements.extend(["jax", "jaxlib", "flax"])
            spec.base_image = f"python:{python_ver}-slim"
        else:
            spec.base_image = f"python:{python_ver}-slim"
        
        # Set CUDA version if GPU framework detected
        if framework in ("pytorch", "tensorflow"):
            spec.cuda_version = cuda_ver
        
        # Add common ML packages
        spec.pip_requirements.extend([
            "scipy",
            "scikit-learn",
            "matplotlib",
            "pandas",
        ])
        
        # Deduplicate
        spec.pip_requirements = list(dict.fromkeys(spec.pip_requirements))
        
        return spec
