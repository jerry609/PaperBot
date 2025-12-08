# repro/generation_agent.py
"""
Generation Agent for Paper2Code-style reproduction.
Uses Claude Agent SDK to generate code based on plan and spec.
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
    logger.warning("claude-agent-sdk not installed; GenerationAgent will use templates.")


class GenerationAgent:
    """
    Phase 3 of Paper2Code pipeline:
    - Generate code files based on plan and spec
    - Iterative refinement on syntax errors
    """

    GENERATION_SYSTEM_PROMPT = """You are an expert ML engineer generating code from research paper specifications.
Generate clean, well-documented, runnable Python code.
Follow best practices: type hints, docstrings, error handling.
Code should be modular and easy to understand.
Only output the requested code file content, no explanations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get("model", "claude-3-5-sonnet-latest")
        self.max_retries = self.config.get("max_retries", 2)

    async def generate_code(
        self,
        paper_context: PaperContext,
        plan: ReproductionPlan,
        spec: ImplementationSpec
    ) -> Dict[str, str]:
        """
        Generate all code files based on plan and spec.
        Returns dict of {filename: content}.
        """
        generated_files = {}

        # Generate each file in the plan
        for filepath, purpose in plan.file_structure.items():
            if filepath == "requirements.txt":
                content = self._generate_requirements(plan)
            else:
                content = await self._generate_file(
                    filepath, purpose, paper_context, plan, spec
                )
            generated_files[filepath] = content

        return generated_files

    async def _generate_file(
        self,
        filepath: str,
        purpose: str,
        paper_context: PaperContext,
        plan: ReproductionPlan,
        spec: ImplementationSpec
    ) -> str:
        """Generate a single code file."""
        if query is None or ClaudeAgentOptions is None:
            return self._fallback_template(filepath, purpose, plan, spec)

        try:
            prompt = f"""Generate Python code for this file.

**File:** `{filepath}`
**Purpose:** {purpose}

{paper_context.to_prompt_context()}

{plan.to_prompt_context()}

{spec.to_prompt_context()}

Requirements:
1. Complete, runnable code (no placeholders like "# TODO")
2. Proper imports at the top
3. Type hints and docstrings
4. Follow Python best practices

Generate ONLY the code for `{filepath}`. Start directly with imports or code, no markdown."""

            opts = ClaudeAgentOptions(
                system_prompt=self.GENERATION_SYSTEM_PROMPT,
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

            # Clean response (remove markdown code blocks if present)
            code = self._clean_code_response(response_text)
            return code

        except Exception as e:
            logger.warning(f"Code generation failed for {filepath}: {e}")
            return self._fallback_template(filepath, purpose, plan, spec)

    async def refine_code(
        self,
        filepath: str,
        code: str,
        error_message: str
    ) -> str:
        """
        Refine code based on error feedback (RePro-style iterative refinement).
        """
        if query is None or ClaudeAgentOptions is None:
            return code  # No refinement possible without SDK

        try:
            prompt = f"""Fix this Python code based on the error.

**File:** `{filepath}`

**Current Code:**
```python
{code[:3000]}
```

**Error:**
```
{error_message[:1000]}
```

Fix the error and return the complete corrected code.
Output ONLY the fixed code, no explanations."""

            opts = ClaudeAgentOptions(
                system_prompt="You are a code debugging expert. Fix errors precisely.",
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

            return self._clean_code_response(response_text)

        except Exception as e:
            logger.warning(f"Code refinement failed: {e}")
            return code

    def _clean_code_response(self, response: str) -> str:
        """Remove markdown code blocks from response."""
        lines = response.strip().split("\n")
        
        # Remove leading ```python or ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        
        # Remove trailing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        
        return "\n".join(lines)

    def _generate_requirements(self, plan: ReproductionPlan) -> str:
        """Generate requirements.txt from plan dependencies."""
        deps = plan.dependencies or ["torch", "numpy", "tqdm"]
        return "\n".join(deps)

    def _fallback_template(
        self,
        filepath: str,
        purpose: str,
        plan: ReproductionPlan,
        spec: ImplementationSpec
    ) -> str:
        """Fallback code templates when LLM is unavailable."""
        templates = {
            "main.py": self._template_main(plan, spec),
            "model.py": self._template_model(spec),
            "data.py": self._template_data(spec),
            "config.py": self._template_config(spec),
            "utils.py": self._template_utils(),
        }
        return templates.get(filepath, f"# {filepath}\n# {purpose}\n\n# TODO: Implement")

    def _template_main(self, plan: ReproductionPlan, spec: ImplementationSpec) -> str:
        return f'''"""
{plan.description}
Auto-generated reproduction code.
"""

import argparse
from config import Config
from model import Model
from data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="{plan.description}")
    parser.add_argument("--epochs", type=int, default={spec.epochs})
    parser.add_argument("--batch_size", type=int, default={spec.batch_size})
    parser.add_argument("--lr", type=float, default={spec.learning_rate})
    args = parser.parse_args()

    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print(f"Starting training with config: {{config}}")
    
    # Initialize model and data
    model = Model(config)
    data_loader = DataLoader(config)
    
    # Training loop
    for epoch in range(config.epochs):
        loss = train_epoch(model, data_loader)
        print(f"Epoch {{epoch+1}}/{{config.epochs}}, Loss: {{loss:.4f}}")

    print("Training complete!")


def train_epoch(model, data_loader):
    """Single training epoch."""
    # TODO: Implement actual training logic
    return 0.0


if __name__ == "__main__":
    main()
'''

    def _template_model(self, spec: ImplementationSpec) -> str:
        return f'''"""
Model definition.
Model type: {spec.model_type}
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Main model class."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # TODO: Define layers based on paper
        self.layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
'''

    def _template_data(self, spec: ImplementationSpec) -> str:
        return f'''"""
Data loading and preprocessing.
Data format: {spec.data_format}
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # TODO: Load data

    def __len__(self):
        return 1000  # TODO: Return actual length

    def __getitem__(self, idx):
        # TODO: Return actual data
        return torch.randn(768), torch.randint(0, 10, (1,)).item()


class DataLoader:
    """Data loader wrapper."""

    def __init__(self, config, data_path="data"):
        self.config = config
        self.dataset = CustomDataset(data_path)
        self.loader = TorchDataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

    def __iter__(self):
        return iter(self.loader)
'''

    def _template_config(self, spec: ImplementationSpec) -> str:
        return f'''"""
Configuration and hyperparameters.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Training configuration."""
    epochs: int = {spec.epochs}
    batch_size: int = {spec.batch_size}
    learning_rate: float = {spec.learning_rate}
    optimizer: str = "{spec.optimizer}"
    
    # Model config
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
'''

    def _template_utils(self) -> str:
        return '''"""
Utility functions.
"""

import logging
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''
