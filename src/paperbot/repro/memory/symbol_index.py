# repro/memory/symbol_index.py
"""
Symbol Index for code memory.

Uses Python AST to parse and index code symbols (classes, functions, variables)
for quick lookup during code generation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a code symbol."""
    name: str
    kind: str  # "class", "function", "variable", "import"
    file: str
    line: int
    signature: str = ""
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # symbols this depends on

    def to_summary(self) -> str:
        """Generate a compact summary for context injection."""
        if self.kind == "class":
            return f"class {self.name}{self.signature}"
        elif self.kind == "function":
            return f"def {self.name}{self.signature}"
        else:
            return f"{self.name}: {self.kind}"


class SymbolIndex:
    """
    AST-based symbol indexer.

    Indexes classes, functions, and module-level variables from Python source code.
    Supports dependency tracking between symbols.
    """

    def __init__(self):
        self._symbols: Dict[str, SymbolInfo] = {}  # name -> info
        self._file_symbols: Dict[str, Set[str]] = {}  # file -> symbol names
        self._reverse_deps: Dict[str, Set[str]] = {}  # symbol -> dependents

    def index_file(self, path: str, content: str) -> List[SymbolInfo]:
        """
        Parse a Python file and index all symbols.

        Args:
            path: File path (used as identifier)
            content: Python source code

        Returns:
            List of indexed SymbolInfo objects
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Failed to parse {path}: {e}")
            return []

        symbols = []
        self._file_symbols[path] = set()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                symbol = self._index_class(node, path)
                symbols.append(symbol)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                symbol = self._index_function(node, path)
                symbols.append(symbol)
            elif isinstance(node, ast.Assign):
                # Module-level variables
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbol = SymbolInfo(
                            name=target.id,
                            kind="variable",
                            file=path,
                            line=node.lineno,
                        )
                        symbols.append(symbol)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Track imports for dependency resolution
                for alias in node.names:
                    name = alias.asname or alias.name
                    symbol = SymbolInfo(
                        name=name,
                        kind="import",
                        file=path,
                        line=node.lineno,
                    )
                    symbols.append(symbol)

        # Register all symbols
        for symbol in symbols:
            self._symbols[symbol.name] = symbol
            self._file_symbols[path].add(symbol.name)

        return symbols

    def _index_class(self, node: ast.ClassDef, path: str) -> SymbolInfo:
        """Index a class definition."""
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        signature = f"({', '.join(bases)})" if bases else ""

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract method signatures
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        return SymbolInfo(
            name=node.name,
            kind="class",
            file=path,
            line=node.lineno,
            signature=signature,
            docstring=docstring,
            dependencies=bases + methods,
        )

    def _index_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> SymbolInfo:
        """Index a function definition."""
        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(arg_str)

        signature = f"({', '.join(args)})"

        # Add return type if present
        if node.returns:
            try:
                signature += f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Find dependencies (function calls within the body)
        deps = self._extract_dependencies(node)

        return SymbolInfo(
            name=node.name,
            kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            file=path,
            line=node.lineno,
            signature=signature,
            docstring=docstring,
            dependencies=deps,
        )

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract symbol dependencies from an AST node."""
        deps = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    deps.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # e.g., self.method or module.function
                    if isinstance(child.func.value, ast.Name):
                        deps.add(child.func.value.id)
            elif isinstance(child, ast.Name):
                # Any name reference
                deps.add(child.id)

        # Filter out builtins and common names
        builtins = {"print", "len", "range", "str", "int", "float", "list", "dict", "set", "self", "cls"}
        return [d for d in deps if d not in builtins]

    def get_symbol(self, name: str) -> Optional[SymbolInfo]:
        """Get a symbol by name."""
        return self._symbols.get(name)

    def get_file_symbols(self, path: str) -> List[SymbolInfo]:
        """Get all symbols in a file."""
        names = self._file_symbols.get(path, set())
        return [self._symbols[name] for name in names if name in self._symbols]

    def find_dependents(self, symbol_name: str) -> Set[str]:
        """Find all symbols that depend on the given symbol."""
        dependents = set()
        for name, info in self._symbols.items():
            if symbol_name in info.dependencies:
                dependents.add(name)
        return dependents

    def get_interface_summary(self, path: str) -> str:
        """
        Generate a concise interface summary for a file.

        Useful for injecting into LLM context.
        """
        symbols = self.get_file_symbols(path)
        if not symbols:
            return ""

        lines = [f"# {path}"]
        for sym in symbols:
            if sym.kind in ("class", "function", "async_function"):
                lines.append(sym.to_summary())

        return "\n".join(lines)

    def clear_file(self, path: str) -> None:
        """Remove all symbols from a file."""
        if path in self._file_symbols:
            for name in self._file_symbols[path]:
                self._symbols.pop(name, None)
            del self._file_symbols[path]

    def to_dict(self) -> Dict:
        """Export index to dictionary."""
        return {
            "symbols": {name: vars(info) for name, info in self._symbols.items()},
            "file_symbols": {path: list(names) for path, names in self._file_symbols.items()},
        }
