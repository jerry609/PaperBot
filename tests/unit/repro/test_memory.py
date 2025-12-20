# tests/unit/repro/test_memory.py
"""
Unit tests for CodeMemory and SymbolIndex modules.
"""

import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from repro.memory.symbol_index import SymbolIndex, SymbolInfo
from repro.memory.code_memory import CodeMemory, FileInfo


class TestSymbolInfo(unittest.TestCase):
    """Tests for SymbolInfo dataclass."""

    def test_to_summary_class(self):
        info = SymbolInfo(
            name="MyClass",
            kind="class",
            file="model.py",
            line=10,
            signature="(BaseModel)",
        )
        summary = info.to_summary()
        self.assertEqual(summary, "class MyClass(BaseModel)")

    def test_to_summary_function(self):
        info = SymbolInfo(
            name="train",
            kind="function",
            file="trainer.py",
            line=5,
            signature="(model, data, epochs: int)",
        )
        summary = info.to_summary()
        self.assertEqual(summary, "def train(model, data, epochs: int)")

    def test_to_summary_variable(self):
        info = SymbolInfo(
            name="LEARNING_RATE",
            kind="variable",
            file="config.py",
            line=1,
        )
        summary = info.to_summary()
        self.assertEqual(summary, "LEARNING_RATE: variable")


class TestSymbolIndex(unittest.TestCase):
    """Tests for SymbolIndex class."""

    def setUp(self):
        self.index = SymbolIndex()

    def test_index_simple_function(self):
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
        symbols = self.index.index_file("utils.py", code)

        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0].name, "hello")
        self.assertEqual(symbols[0].kind, "function")
        self.assertIn("name: str", symbols[0].signature)
        self.assertEqual(symbols[0].docstring, "Say hello.")

    def test_index_class(self):
        code = '''
class Model(nn.Module):
    """A neural network model."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc(x)
'''
        symbols = self.index.index_file("model.py", code)

        # Should find the class
        class_symbols = [s for s in symbols if s.kind == "class"]
        self.assertEqual(len(class_symbols), 1)
        self.assertEqual(class_symbols[0].name, "Model")
        self.assertIn("nn.Module", class_symbols[0].signature)
        self.assertEqual(class_symbols[0].docstring, "A neural network model.")

    def test_index_async_function(self):
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    pass
'''
        symbols = self.index.index_file("api.py", code)

        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0].kind, "async_function")
        self.assertEqual(symbols[0].name, "fetch_data")

    def test_index_module_variables(self):
        code = '''
MAX_SIZE = 1024
DEFAULT_NAME = "model"
'''
        symbols = self.index.index_file("constants.py", code)

        var_symbols = [s for s in symbols if s.kind == "variable"]
        self.assertEqual(len(var_symbols), 2)
        names = {s.name for s in var_symbols}
        self.assertIn("MAX_SIZE", names)
        self.assertIn("DEFAULT_NAME", names)

    def test_index_imports(self):
        code = '''
import torch
from torch import nn
from typing import List, Optional
'''
        symbols = self.index.index_file("imports.py", code)

        import_symbols = [s for s in symbols if s.kind == "import"]
        self.assertTrue(len(import_symbols) >= 2)

    def test_get_symbol(self):
        code = "def my_func(): pass"
        self.index.index_file("test.py", code)

        symbol = self.index.get_symbol("my_func")
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.name, "my_func")

    def test_get_file_symbols(self):
        code = '''
def func1(): pass
def func2(): pass
class MyClass: pass
'''
        self.index.index_file("multi.py", code)

        symbols = self.index.get_file_symbols("multi.py")
        names = {s.name for s in symbols}
        self.assertIn("func1", names)
        self.assertIn("func2", names)
        self.assertIn("MyClass", names)

    def test_find_dependents(self):
        code1 = "def helper(): pass"
        code2 = '''
def main():
    helper()
'''
        self.index.index_file("utils.py", code1)
        self.index.index_file("main.py", code2)

        dependents = self.index.find_dependents("helper")
        self.assertIn("main", dependents)

    def test_get_interface_summary(self):
        code = '''
class Model:
    """The main model."""
    pass

def train(model):
    """Train the model."""
    pass
'''
        self.index.index_file("model.py", code)

        summary = self.index.get_interface_summary("model.py")
        self.assertIn("model.py", summary)
        self.assertIn("class Model", summary)
        self.assertIn("def train", summary)

    def test_clear_file(self):
        code = "def func(): pass"
        self.index.index_file("temp.py", code)

        self.assertIsNotNone(self.index.get_symbol("func"))

        self.index.clear_file("temp.py")

        self.assertIsNone(self.index.get_symbol("func"))

    def test_syntax_error_handling(self):
        invalid_code = "def broken("
        symbols = self.index.index_file("broken.py", invalid_code)

        # Should handle gracefully and return empty list
        self.assertEqual(symbols, [])

    def test_to_dict(self):
        code = "def func(): pass"
        self.index.index_file("test.py", code)

        data = self.index.to_dict()
        self.assertIn("symbols", data)
        self.assertIn("file_symbols", data)
        self.assertIn("func", data["symbols"])


class TestCodeMemory(unittest.TestCase):
    """Tests for CodeMemory class."""

    def setUp(self):
        self.memory = CodeMemory(max_context_tokens=1000)

    def test_add_file(self):
        code = "print('hello')"
        self.memory.add_file("main.py", code, purpose="Entry point")

        self.assertEqual(self.memory.file_count, 1)
        self.assertEqual(self.memory.get_file("main.py"), code)

    def test_files_property(self):
        self.memory.add_file("a.py", "# a")
        self.memory.add_file("b.py", "# b")

        files = self.memory.files
        self.assertEqual(len(files), 2)
        self.assertEqual(files["a.py"], "# a")
        self.assertEqual(files["b.py"], "# b")

    def test_extract_imports(self):
        config_code = "CONFIG = {}"
        model_code = '''
from config import CONFIG

class Model:
    pass
'''
        self.memory.add_file("config.py", config_code)
        self.memory.add_file("model.py", model_code)

        # Check dependencies were detected
        deps = self.memory._files["model.py"].dependencies
        self.assertIn("config.py", deps)

    def test_compute_generation_order(self):
        file_structure = {
            "main.py": "entry",
            "model.py": "model",
            "config.py": "config",
            "utils.py": "utils",
            "data.py": "data",
        }

        order = self.memory.compute_generation_order(file_structure)

        # Config should come before model
        config_idx = order.index("config.py")
        model_idx = order.index("model.py")
        main_idx = order.index("main.py")

        self.assertLess(config_idx, model_idx)
        self.assertLess(model_idx, main_idx)

    def test_get_relevant_context(self):
        self.memory.add_file("config.py", "LR = 0.001", purpose="Configuration")
        self.memory.add_file("model.py", "class Model: pass", purpose="Model definition")

        context = self.memory.get_relevant_context("trainer.py", "Training logic")

        # Should include config and model as dependencies
        self.assertIn("config.py", context)

    def test_get_interface_summary(self):
        self.memory.add_file("model.py", "class Model: pass")
        self.memory.add_file("data.py", "def load_data(): pass")

        summary = self.memory.get_interface_summary()

        self.assertIn("class Model", summary)
        self.assertIn("def load_data", summary)

    def test_get_symbol_definition(self):
        code = '''
def my_function(x, y):
    """Add two numbers."""
    return x + y

def other():
    pass
'''
        self.memory.add_file("math.py", code)

        definition = self.memory.get_symbol_definition("my_function")

        self.assertIn("def my_function", definition)
        self.assertIn("return x + y", definition)

    def test_get_dependency_graph(self):
        self.memory.add_file("config.py", "X = 1")
        self.memory.add_file("model.py", "from config import X")

        graph = self.memory.get_dependency_graph()

        self.assertIn("config.py", graph)
        self.assertIn("model.py", graph)

    def test_clear(self):
        self.memory.add_file("a.py", "# a")
        self.memory.add_file("b.py", "# b")

        self.assertEqual(self.memory.file_count, 2)

        self.memory.clear()

        self.assertEqual(self.memory.file_count, 0)

    def test_to_dict(self):
        self.memory.add_file("test.py", "# test", purpose="Test file")

        data = self.memory.to_dict()

        self.assertIn("files", data)
        self.assertIn("generation_order", data)
        self.assertIn("symbol_index", data)
        self.assertIn("test.py", data["files"])
        self.assertEqual(data["files"]["test.py"]["purpose"], "Test file")

    def test_predict_dependencies(self):
        self.memory.add_file("config.py", "# config")
        self.memory.add_file("model.py", "# model")
        self.memory.add_file("data.py", "# data")

        deps = self.memory._predict_dependencies("trainer.py")

        self.assertIn("config.py", deps)
        self.assertIn("model.py", deps)
        self.assertIn("data.py", deps)

    def test_token_budget_respected(self):
        # Create a memory with small token budget
        memory = CodeMemory(max_context_tokens=50)  # ~200 chars

        # Add a large file
        large_content = "x = 1\n" * 100  # Much larger than budget
        memory.add_file("large.py", large_content)

        context = memory.get_relevant_context("test.py", "test")

        # Context should be truncated
        self.assertLess(len(context), len(large_content))


class TestFileInfo(unittest.TestCase):
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        info = FileInfo(
            path="model.py",
            content="class Model: pass",
            purpose="Model definition",
            generation_order=0,
        )

        self.assertEqual(info.path, "model.py")
        self.assertEqual(info.purpose, "Model definition")
        self.assertEqual(info.dependencies, set())
        self.assertEqual(info.dependents, set())


if __name__ == "__main__":
    unittest.main()
