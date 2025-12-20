# repro/memory/code_memory.py
"""
Stateful Code Memory for Paper2Code pipeline.

Provides cross-node context sharing during code generation:
- Tracks all generated files and their contents
- Maintains a symbol index for quick lookup
- Computes relevant context for each generation step
- Manages token budget for context injection
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .symbol_index import SymbolIndex, SymbolInfo

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Metadata about a generated file."""
    path: str
    content: str
    purpose: str = ""
    generation_order: int = 0
    dependencies: Set[str] = field(default_factory=set)  # files this imports from
    dependents: Set[str] = field(default_factory=set)  # files that import this


class CodeMemory:
    """
    Stateful Code Memory - cross-node context sharing for code generation.

    Features:
    - Track all generated files and symbols
    - Compute relevant context for each file generation
    - Manage token budget for context injection
    - Support dependency-aware generation ordering

    Usage:
        memory = CodeMemory(max_context_tokens=8000)

        # After generating each file
        memory.add_file("config.py", config_code, purpose="Configuration")
        memory.add_file("model.py", model_code, purpose="Model definition")

        # Before generating next file
        context = memory.get_relevant_context("trainer.py", "Training logic")
        # Use context in LLM prompt
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_context_tokens: int = 8000):
        """
        Initialize CodeMemory.

        Args:
            max_context_tokens: Maximum tokens for context injection
        """
        self.max_context_tokens = max_context_tokens
        self._files: Dict[str, FileInfo] = {}
        self._symbol_index = SymbolIndex()
        self._generation_order: List[str] = []

    def add_file(self, path: str, content: str, purpose: str = "") -> None:
        """
        Add a generated file to memory.

        Args:
            path: File path (relative)
            content: File content
            purpose: Brief description of file purpose
        """
        # Index symbols
        self._symbol_index.index_file(path, content)

        # Extract dependencies (imports)
        dependencies = self._extract_imports(content)

        # Create file info
        file_info = FileInfo(
            path=path,
            content=content,
            purpose=purpose,
            generation_order=len(self._generation_order),
            dependencies=dependencies,
        )

        self._files[path] = file_info
        self._generation_order.append(path)

        # Update reverse dependencies
        for dep in dependencies:
            if dep in self._files:
                self._files[dep].dependents.add(path)

        logger.debug(f"Added file to memory: {path} ({len(content)} chars)")

    def _extract_imports(self, content: str) -> Set[str]:
        """Extract local file imports from code."""
        imports = set()

        # Match: from .module import ... or from module import ...
        for match in re.finditer(r'from\s+\.?(\w+)\s+import', content):
            module = match.group(1)
            # Check if it's a local module
            if f"{module}.py" in self._files or module in [
                Path(p).stem for p in self._files
            ]:
                imports.add(f"{module}.py")

        # Match: import module
        for match in re.finditer(r'^import\s+(\w+)', content, re.MULTILINE):
            module = match.group(1)
            if f"{module}.py" in self._files:
                imports.add(f"{module}.py")

        return imports

    def get_file(self, path: str) -> Optional[str]:
        """Get file content by path."""
        if path in self._files:
            return self._files[path].content
        return None

    def get_relevant_context(
        self,
        current_file: str,
        query: str,
        include_interfaces: bool = True,
    ) -> str:
        """
        Get relevant context for generating a file.

        Prioritizes:
        1. Direct dependencies (files that current_file will import)
        2. Symbols matching the query
        3. Interface summaries of all generated files

        Args:
            current_file: File being generated
            query: Description of what's being generated (purpose)
            include_interfaces: Whether to include interface summaries

        Returns:
            Context string within token budget
        """
        context_parts = []
        remaining_chars = self.max_context_tokens * self.CHARS_PER_TOKEN

        # 1. Predict dependencies based on file name
        predicted_deps = self._predict_dependencies(current_file)
        for dep_path in predicted_deps:
            if dep_path in self._files and remaining_chars > 0:
                file_info = self._files[dep_path]
                header = f"# === {dep_path} ({file_info.purpose}) ===\n"
                content = file_info.content

                # Truncate if needed
                available = remaining_chars - len(header)
                if len(content) > available:
                    content = content[:available] + "\n# ... (truncated)"

                context_parts.append(header + content)
                remaining_chars -= len(header) + len(content)

        # 2. Find relevant symbols based on query
        if remaining_chars > 500:  # Only if we have enough budget
            query_tokens = set(query.lower().split())
            relevant_symbols = []

            for name, info in self._symbol_index._symbols.items():
                # Score by query match
                name_tokens = set(name.lower().split("_"))
                if query_tokens & name_tokens:
                    relevant_symbols.append(info)

            # Add symbol definitions
            for sym in relevant_symbols[:5]:  # Limit to top 5
                if sym.file not in predicted_deps:  # Avoid duplication
                    summary = sym.to_summary()
                    if sym.docstring:
                        summary += f'\n    """{sym.docstring[:100]}..."""'
                    context_parts.append(f"# From {sym.file}:\n{summary}")
                    remaining_chars -= len(summary) + 50

        # 3. Interface summaries
        if include_interfaces and remaining_chars > 200:
            interfaces = self.get_interface_summary()
            if len(interfaces) < remaining_chars:
                context_parts.append(f"\n# === Available Interfaces ===\n{interfaces}")

        return "\n\n".join(context_parts)

    def _predict_dependencies(self, current_file: str) -> List[str]:
        """
        Predict which files the current file will likely depend on.

        Based on common patterns:
        - main.py depends on everything
        - trainer.py depends on model.py, data.py, config.py
        - model.py depends on config.py, layers.py
        """
        stem = Path(current_file).stem.lower()
        deps = []

        # Universal dependencies
        for common in ["config.py", "utils.py"]:
            if common in self._files and common != current_file:
                deps.append(common)

        # File-specific dependencies
        dep_map = {
            "main": ["model", "data", "trainer", "config"],
            "trainer": ["model", "data", "config"],
            "model": ["config", "layers", "modules"],
            "data": ["config", "utils"],
            "evaluation": ["model", "data", "config"],
        }

        if stem in dep_map:
            for dep in dep_map[stem]:
                dep_file = f"{dep}.py"
                if dep_file in self._files and dep_file != current_file:
                    deps.append(dep_file)

        return deps

    def get_interface_summary(self) -> str:
        """
        Get interface summary for all generated files.

        Returns a concise overview of classes/functions.
        """
        summaries = []
        for path in self._generation_order:
            summary = self._symbol_index.get_interface_summary(path)
            if summary:
                summaries.append(summary)
        return "\n\n".join(summaries)

    def get_symbol_definition(self, symbol_name: str) -> Optional[str]:
        """
        Get the full definition of a symbol.

        Extracts the relevant code block from the source file.
        """
        info = self._symbol_index.get_symbol(symbol_name)
        if not info or info.file not in self._files:
            return None

        content = self._files[info.file].content
        lines = content.split("\n")

        # Find the symbol definition
        start_line = info.line - 1  # 0-indexed

        if info.kind in ("class", "function", "async_function"):
            # Find the end of the block (next top-level definition or EOF)
            end_line = start_line + 1
            initial_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

            while end_line < len(lines):
                line = lines[end_line]
                if line.strip():  # Non-empty line
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= initial_indent and not line.strip().startswith(("def ", "class ", "@")):
                        break
                end_line += 1

            return "\n".join(lines[start_line:end_line])

        return lines[start_line] if start_line < len(lines) else None

    def compute_generation_order(self, file_structure: Dict[str, str]) -> List[str]:
        """
        Compute optimal generation order based on dependencies.

        Uses topological sort to ensure dependencies are generated first.

        Args:
            file_structure: Dict mapping path to purpose

        Returns:
            Ordered list of file paths
        """
        # Priority layers
        layers = {
            0: ["config.py", "constants.py", "types.py"],
            1: ["utils.py", "helpers.py"],
            2: ["layers.py", "modules.py"],
            3: ["model.py", "network.py"],
            4: ["data.py", "dataset.py", "dataloader.py"],
            5: ["trainer.py", "train.py"],
            6: ["evaluation.py", "metrics.py"],
            7: ["main.py", "run.py", "__init__.py"],
        }

        # Build ordered list
        ordered = []
        remaining = set(file_structure.keys())

        # Add by layer priority
        for layer_idx in sorted(layers.keys()):
            for pattern in layers[layer_idx]:
                for path in list(remaining):
                    if Path(path).name == pattern or pattern in path.lower():
                        ordered.append(path)
                        remaining.discard(path)

        # Add remaining files
        ordered.extend(sorted(remaining))

        return ordered

    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get the file dependency graph."""
        return {path: info.dependencies for path, info in self._files.items()}

    def clear(self) -> None:
        """Clear all memory."""
        self._files.clear()
        self._symbol_index = SymbolIndex()
        self._generation_order.clear()

    @property
    def files(self) -> Dict[str, str]:
        """Get all files as path -> content dict."""
        return {path: info.content for path, info in self._files.items()}

    @property
    def file_count(self) -> int:
        """Get number of files in memory."""
        return len(self._files)

    def to_dict(self) -> Dict:
        """Export memory to dictionary."""
        return {
            "files": {
                path: {
                    "content": info.content,
                    "purpose": info.purpose,
                    "generation_order": info.generation_order,
                    "dependencies": list(info.dependencies),
                }
                for path, info in self._files.items()
            },
            "generation_order": self._generation_order,
            "symbol_index": self._symbol_index.to_dict(),
        }
