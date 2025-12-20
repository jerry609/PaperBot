# repro/models.py
"""
Data models for Paper2Code-style reproduction pipeline.

Includes:
- Blueprint: Distilled paper structure for efficient context
- PaperContext: Raw paper information
- ReproductionPlan: High-level implementation plan
- ImplementationSpec: Detailed specs for code generation
- ReproductionResult: Final pipeline output
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ReproPhase(Enum):
    """Phases of the reproduction pipeline."""
    PLANNING = "planning"
    ENVIRONMENT = "environment"  # NEW: Environment inference phase
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"


class VerificationStep(Enum):
    """Fine-grained verification steps."""
    SYNTAX_CHECK = "syntax_check"
    IMPORT_CHECK = "import_check"
    UNIT_TEST = "unit_test"
    SMOKE_RUN = "smoke_run"


class ErrorType(Enum):
    """Error types for Self-Healing Debugger."""
    SYNTAX = "syntax"           # SyntaxError - fix Python code
    DEPENDENCY = "dependency"   # ImportError/ModuleNotFoundError - update requirements
    LOGIC = "logic"             # Runtime errors - regenerate module
    UNKNOWN = "unknown"


# ============================================================================
# Blueprint - Distilled Paper Structure (DeepCode-inspired)
# ============================================================================

@dataclass
class AlgorithmSpec:
    """Specification for a core algorithm."""
    name: str
    pseudocode: str = ""
    complexity: str = ""  # e.g., "O(n^2)", "O(n log n)"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class Blueprint:
    """
    Distilled paper structure for efficient LLM context.

    Inspired by DeepCode's Blueprint Distillation, this class compresses
    paper information into a structured format that maximizes signal
    while minimizing token usage.

    Attributes:
        # Architecture Layer
        architecture_type: Main architecture (transformer, cnn, gnn, hybrid)
        module_hierarchy: Module tree, e.g., {"model": ["encoder", "decoder"]}
        data_flow: Data flow graph, e.g., [("input", "encoder"), ...]

        # Algorithm Layer
        core_algorithms: List of algorithm specifications
        loss_functions: Loss function names/formulas
        optimization_strategy: Training strategy description

        # Implementation Layer
        key_hyperparameters: Extracted hyperparameters
        input_output_spec: I/O specifications

        # Metadata
        paper_year: Publication year (for version inference)
        framework_hints: Detected frameworks ["pytorch", "tensorflow", ...]
    """
    # Architecture Layer
    architecture_type: str = "unknown"  # transformer/cnn/gnn/rnn/hybrid
    module_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    data_flow: List[Tuple[str, str]] = field(default_factory=list)

    # Algorithm Layer
    core_algorithms: List[AlgorithmSpec] = field(default_factory=list)
    loss_functions: List[str] = field(default_factory=list)
    optimization_strategy: str = ""

    # Implementation Layer
    key_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    input_output_spec: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    paper_title: str = ""
    paper_year: int = 2024
    framework_hints: List[str] = field(default_factory=list)
    domain: str = ""  # nlp/cv/audio/multimodal/rl

    def to_compressed_context(self, max_tokens: int = 2000) -> str:
        """
        Generate compressed context string for LLM prompts.

        Prioritizes information by importance:
        1. Architecture type and main components
        2. Core algorithms with pseudocode
        3. Key hyperparameters
        4. I/O specifications

        Args:
            max_tokens: Approximate token budget

        Returns:
            Compressed context string
        """
        sections = []
        char_budget = max_tokens * 4  # Approximate chars per token

        # Header
        header = f"## Blueprint: {self.paper_title} ({self.paper_year})\n"
        header += f"Architecture: {self.architecture_type}\n"
        header += f"Domain: {self.domain}\n"
        if self.framework_hints:
            header += f"Frameworks: {', '.join(self.framework_hints)}\n"
        sections.append(header)
        char_budget -= len(header)

        # Module hierarchy
        if self.module_hierarchy and char_budget > 200:
            hierarchy = "### Module Structure\n"
            for parent, children in self.module_hierarchy.items():
                hierarchy += f"- {parent}: {', '.join(children)}\n"
            sections.append(hierarchy)
            char_budget -= len(hierarchy)

        # Core algorithms
        if self.core_algorithms and char_budget > 500:
            algos = "### Core Algorithms\n"
            for algo in self.core_algorithms[:3]:  # Limit to top 3
                algos += f"**{algo.name}**"
                if algo.complexity:
                    algos += f" ({algo.complexity})"
                algos += "\n"
                if algo.pseudocode and char_budget > 300:
                    algos += f"```\n{algo.pseudocode[:500]}\n```\n"
            sections.append(algos)
            char_budget -= len(algos)

        # Hyperparameters
        if self.key_hyperparameters and char_budget > 100:
            params = "### Hyperparameters\n"
            for key, value in list(self.key_hyperparameters.items())[:10]:
                params += f"- {key}: {value}\n"
            sections.append(params)
            char_budget -= len(params)

        # Loss functions
        if self.loss_functions and char_budget > 50:
            losses = f"### Loss: {', '.join(self.loss_functions[:3])}\n"
            sections.append(losses)

        # I/O spec
        if self.input_output_spec and char_budget > 100:
            io_spec = "### I/O Specification\n"
            for key, spec in self.input_output_spec.items():
                io_spec += f"- {key}: {spec}\n"
            sections.append(io_spec)

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "architecture_type": self.architecture_type,
            "module_hierarchy": self.module_hierarchy,
            "data_flow": self.data_flow,
            "core_algorithms": [
                {"name": a.name, "pseudocode": a.pseudocode, "complexity": a.complexity}
                for a in self.core_algorithms
            ],
            "loss_functions": self.loss_functions,
            "optimization_strategy": self.optimization_strategy,
            "key_hyperparameters": self.key_hyperparameters,
            "input_output_spec": self.input_output_spec,
            "paper_title": self.paper_title,
            "paper_year": self.paper_year,
            "framework_hints": self.framework_hints,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blueprint":
        """Create Blueprint from dictionary."""
        algorithms = [
            AlgorithmSpec(**algo) for algo in data.get("core_algorithms", [])
        ]
        return cls(
            architecture_type=data.get("architecture_type", "unknown"),
            module_hierarchy=data.get("module_hierarchy", {}),
            data_flow=data.get("data_flow", []),
            core_algorithms=algorithms,
            loss_functions=data.get("loss_functions", []),
            optimization_strategy=data.get("optimization_strategy", ""),
            key_hyperparameters=data.get("key_hyperparameters", {}),
            input_output_spec=data.get("input_output_spec", {}),
            paper_title=data.get("paper_title", ""),
            paper_year=data.get("paper_year", 2024),
            framework_hints=data.get("framework_hints", []),
            domain=data.get("domain", ""),
        )


@dataclass
class EnvironmentSpec:
    """
    Environment specification for code reproduction.
    Generated by EnvironmentInferenceNode based on paper analysis.
    """
    python_version: str = "3.10"
    pytorch_version: Optional[str] = None
    tensorflow_version: Optional[str] = None
    cuda_version: Optional[str] = None
    base_image: str = "python:3.10-slim"
    pip_requirements: List[str] = field(default_factory=list)
    conda_channels: List[str] = field(default_factory=lambda: ["conda-forge", "pytorch"])
    conda_dependencies: List[str] = field(default_factory=list)
    dockerfile_content: Optional[str] = None
    conda_yaml_content: Optional[str] = None
    inferred_from: str = ""  # Source of inference: "paper_year", "code_snippets", "explicit"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "tensorflow_version": self.tensorflow_version,
            "cuda_version": self.cuda_version,
            "base_image": self.base_image,
            "pip_requirements": self.pip_requirements,
            "conda_channels": self.conda_channels,
            "conda_dependencies": self.conda_dependencies,
            "inferred_from": self.inferred_from,
        }
    
    def generate_dockerfile(self) -> str:
        """Generate a Dockerfile based on the inferred environment."""
        lines = [f"FROM {self.base_image}", ""]
        lines.append("# Install system dependencies")
        lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\")
        lines.append("    git curl && \\")
        lines.append("    rm -rf /var/lib/apt/lists/*")
        lines.append("")
        lines.append("WORKDIR /app")
        lines.append("")
        
        # Install PyTorch/TensorFlow if specified
        if self.pytorch_version:
            cuda_suffix = f"+cu{self.cuda_version.replace('.', '')}" if self.cuda_version else ""
            lines.append(f"# Install PyTorch {self.pytorch_version}")
            lines.append(f"RUN pip install torch=={self.pytorch_version}{cuda_suffix} --index-url https://download.pytorch.org/whl/{'cu' + self.cuda_version.replace('.', '') if self.cuda_version else 'cpu'}")
            lines.append("")
        
        if self.tensorflow_version:
            lines.append(f"# Install TensorFlow {self.tensorflow_version}")
            lines.append(f"RUN pip install tensorflow=={self.tensorflow_version}")
            lines.append("")
        
        lines.append("# Install additional requirements")
        lines.append("COPY requirements.txt .")
        lines.append("RUN pip install -r requirements.txt")
        lines.append("")
        lines.append("COPY . .")
        lines.append("")
        lines.append('CMD ["python", "main.py"]')
        
        self.dockerfile_content = "\n".join(lines)
        return self.dockerfile_content
    
    def generate_conda_yaml(self) -> str:
        """Generate a conda environment.yaml file."""
        import yaml
        
        env = {
            "name": "repro_env",
            "channels": self.conda_channels,
            "dependencies": [
                f"python={self.python_version}",
            ]
        }
        
        if self.pytorch_version:
            env["dependencies"].append(f"pytorch={self.pytorch_version}")
            if self.cuda_version:
                env["dependencies"].append(f"cudatoolkit={self.cuda_version}")
        
        if self.tensorflow_version:
            env["dependencies"].append(f"tensorflow={self.tensorflow_version}")
        
        env["dependencies"].extend(self.conda_dependencies)
        
        if self.pip_requirements:
            env["dependencies"].append({"pip": self.pip_requirements})
        
        self.conda_yaml_content = yaml.dump(env, default_flow_style=False)
        return self.conda_yaml_content


@dataclass
class PaperContext:
    """
    Context from the paper being reproduced.
    This is the key input for Paper2Code-style generation.
    """
    title: str
    abstract: str
    method_section: Optional[str] = None  # Key algorithms/methods text
    algorithm_blocks: List[str] = field(default_factory=list)  # Pseudocode blocks
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    datasets_mentioned: List[str] = field(default_factory=list)
    github_url: Optional[str] = None  # If exists, use as reference
    
    def to_prompt_context(self) -> str:
        """Format paper context for LLM prompts."""
        ctx = f"## Paper: {self.title}\n\n"
        ctx += f"### Abstract\n{self.abstract}\n\n"
        if self.method_section:
            ctx += f"### Method Section\n{self.method_section[:3000]}...\n\n"
        if self.algorithm_blocks:
            ctx += "### Algorithm Blocks\n"
            for i, algo in enumerate(self.algorithm_blocks[:3], 1):
                ctx += f"```\n{algo}\n```\n"
        if self.hyperparameters:
            ctx += f"### Hyperparameters\n{self.hyperparameters}\n\n"
        return ctx


@dataclass
class ReproductionPlan:
    """
    Output of the Planning Agent.
    High-level structure of what needs to be implemented.
    """
    project_name: str
    description: str
    file_structure: Dict[str, str] = field(default_factory=dict)  # path -> purpose
    entry_point: str = "main.py"
    dependencies: List[str] = field(default_factory=list)
    key_components: List[str] = field(default_factory=list)  # e.g., ["DataLoader", "Model", "Trainer"]
    estimated_complexity: str = "medium"  # low/medium/high
    
    def to_prompt_context(self) -> str:
        """Format plan for downstream agents."""
        ctx = f"## Reproduction Plan: {self.project_name}\n\n"
        ctx += f"**Description:** {self.description}\n\n"
        ctx += "### File Structure\n"
        for path, purpose in self.file_structure.items():
            ctx += f"- `{path}`: {purpose}\n"
        ctx += f"\n**Entry Point:** `{self.entry_point}`\n"
        ctx += f"**Key Components:** {', '.join(self.key_components)}\n"
        ctx += f"**Dependencies:** {', '.join(self.dependencies)}\n"
        return ctx


@dataclass
class ImplementationSpec:
    """
    Output of the Analysis Agent.
    Detailed specifications for code generation.
    """
    # Model architecture details
    model_type: str = ""  # e.g., "transformer", "cnn", "mlp"
    layers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Training details
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    
    # Data details
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    data_format: str = ""  # e.g., "csv", "json", "numpy"
    
    # Additional configs
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Format spec for code generation."""
        ctx = "## Implementation Specification\n\n"
        ctx += f"**Model Type:** {self.model_type}\n"
        ctx += f"**Optimizer:** {self.optimizer} (lr={self.learning_rate})\n"
        ctx += f"**Batch Size:** {self.batch_size}, **Epochs:** {self.epochs}\n"
        if self.input_shape:
            ctx += f"**Input Shape:** {self.input_shape}\n"
        if self.layers:
            ctx += "**Layers:**\n"
            for layer in self.layers:
                ctx += f"  - {layer}\n"
        return ctx


@dataclass
class VerificationResult:
    """
    Output of the Verification Agent.
    Fine-grained results for each check.
    """
    step: VerificationStep
    passed: bool
    message: str = ""
    logs: str = ""
    duration_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "passed": self.passed,
            "message": self.message,
            "logs": self.logs[-500:],  # Truncate
            "duration_sec": self.duration_sec
        }


@dataclass
class ReproductionResult:
    """
    Final output of the entire reproduction pipeline.
    """
    paper_title: str
    status: ReproPhase = ReproPhase.PLANNING
    
    # Phase outputs
    plan: Optional[ReproductionPlan] = None
    spec: Optional[ImplementationSpec] = None
    generated_files: Dict[str, str] = field(default_factory=dict)  # path -> content
    
    # Verification details
    verification_results: List[VerificationResult] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    overall_score: int = 0  # 0-100
    phases_completed: List[str] = field(default_factory=list)
    retry_count: int = 0
    total_duration_sec: float = 0.0
    
    # Errors
    error: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def compute_score(self) -> int:
        """Compute overall score based on verification results."""
        if not self.verification_results:
            return 0
        
        weights = {
            VerificationStep.SYNTAX_CHECK: 25,
            VerificationStep.IMPORT_CHECK: 25,
            VerificationStep.UNIT_TEST: 30,
            VerificationStep.SMOKE_RUN: 20
        }
        
        score = 0
        for vr in self.verification_results:
            if vr.passed:
                score += weights.get(vr.step, 0)
        
        self.overall_score = score
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "status": self.status.value if isinstance(self.status, ReproPhase) else self.status,
            "overall_score": self.overall_score,
            "phases_completed": self.phases_completed,
            "verification_results": [vr.to_dict() for vr in self.verification_results],
            "verification": self.verification,
            "retry_count": self.retry_count,
            "total_duration_sec": self.total_duration_sec,
            "error": self.error,
            "errors": self.errors,
            "generated_files": list(self.generated_files.keys())
        }

