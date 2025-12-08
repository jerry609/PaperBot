# repro/models.py
"""
Data models for Paper2Code-style reproduction pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ReproPhase(Enum):
    """Phases of the reproduction pipeline."""
    PLANNING = "planning"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VERIFICATION = "verification"


class VerificationStep(Enum):
    """Fine-grained verification steps."""
    SYNTAX_CHECK = "syntax_check"
    IMPORT_CHECK = "import_check"
    UNIT_TEST = "unit_test"
    SMOKE_RUN = "smoke_run"


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
    status: str  # "success", "partial", "failed"
    
    # Phase outputs
    plan: Optional[ReproductionPlan] = None
    spec: Optional[ImplementationSpec] = None
    generated_files: Dict[str, str] = field(default_factory=dict)  # path -> content
    
    # Verification details
    verification_results: List[VerificationResult] = field(default_factory=list)
    
    # Metrics
    overall_score: int = 0  # 0-100
    phases_completed: List[str] = field(default_factory=list)
    retry_count: int = 0
    total_duration_sec: float = 0.0
    
    # Errors
    error: Optional[str] = None
    
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
            "status": self.status,
            "overall_score": self.overall_score,
            "phases_completed": self.phases_completed,
            "verification_results": [vr.to_dict() for vr in self.verification_results],
            "retry_count": self.retry_count,
            "total_duration_sec": self.total_duration_sec,
            "error": self.error,
            "generated_files": list(self.generated_files.keys())
        }
