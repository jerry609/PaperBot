"""Pytest coverage for repro code memory primitives."""

from __future__ import annotations

from paperbot.repro.memory.code_memory import CodeMemory, FileInfo
from paperbot.repro.memory.symbol_index import SymbolIndex, SymbolInfo


def test_symbol_info_to_summary_variants():
    class_info = SymbolInfo(
        name="MyClass",
        kind="class",
        file="model.py",
        line=10,
        signature="(BaseModel)",
    )
    function_info = SymbolInfo(
        name="train",
        kind="function",
        file="trainer.py",
        line=20,
        signature="(model, data, epochs: int)",
    )
    variable_info = SymbolInfo(
        name="LEARNING_RATE",
        kind="variable",
        file="config.py",
        line=1,
    )

    assert class_info.to_summary() == "class MyClass(BaseModel)"
    assert function_info.to_summary() == "def train(model, data, epochs: int)"
    assert variable_info.to_summary() == "LEARNING_RATE: variable"


def test_symbol_index_extracts_symbols_docstrings_and_imports():
    code = """
import torch
from torch import nn

class Model(nn.Module):
    \"\"\"A neural network model.\"\"\"

    def forward(self, x):
        return x


async def fetch_data(url: str) -> dict:
    \"\"\"Fetch data asynchronously.\"\"\"
    return {}
"""
    index = SymbolIndex()

    symbols = index.index_file("model.py", code)

    class_symbols = [symbol for symbol in symbols if symbol.kind == "class"]
    async_symbols = [symbol for symbol in symbols if symbol.kind == "async_function"]
    import_symbols = [symbol for symbol in symbols if symbol.kind == "import"]

    assert class_symbols[0].name == "Model"
    assert class_symbols[0].docstring == "A neural network model."
    assert async_symbols[0].name == "fetch_data"
    assert len(import_symbols) >= 2


def test_symbol_index_handles_syntax_errors_gracefully():
    index = SymbolIndex()

    assert index.index_file("broken.py", "def broken(") == []


def test_symbol_index_tracks_file_symbols_and_dependents():
    index = SymbolIndex()
    index.index_file("utils.py", "def helper():\n    pass\n")
    index.index_file("main.py", "def main():\n    helper()\n")

    file_symbols = {symbol.name for symbol in index.get_file_symbols("utils.py")}
    dependents = index.find_dependents("helper")

    assert "helper" in file_symbols
    assert "main" in dependents
    assert index.get_symbol("helper") is not None


def test_code_memory_tracks_files_and_dependencies():
    memory = CodeMemory(max_context_tokens=1000)
    memory.add_file("config.py", "CONFIG = {}", purpose="Configuration")
    memory.add_file(
        "model.py",
        "from config import CONFIG\n\nclass Model:\n    pass\n",
        purpose="Model definition",
    )

    assert memory.file_count == 2
    assert memory.get_file("config.py") == "CONFIG = {}"
    assert memory.files["model.py"].startswith("from config import CONFIG")
    assert "config.py" in memory._files["model.py"].dependencies


def test_code_memory_computes_generation_order_from_dependencies():
    memory = CodeMemory()
    file_structure = {
        "main.py": "entry",
        "trainer.py": "training logic",
        "model.py": "model",
        "config.py": "configuration",
        "data.py": "data loading",
    }

    order = memory.compute_generation_order(file_structure)

    assert order.index("config.py") < order.index("model.py")
    assert order.index("model.py") < order.index("trainer.py")
    assert order.index("main.py") == len(order) - 1


def test_code_memory_relevant_context_and_interface_summary_include_known_files():
    memory = CodeMemory(max_context_tokens=1000)
    memory.add_file("config.py", "LR = 0.001", purpose="Configuration")
    memory.add_file("model.py", "class Model:\n    pass\n", purpose="Model definition")
    memory.add_file("data.py", "def load_data():\n    pass\n", purpose="Data loading")

    context = memory.get_relevant_context("trainer.py", "Training logic")
    summary = memory.get_interface_summary()

    assert "config.py" in context
    assert "class Model" in summary
    assert "def load_data" in summary


def test_code_memory_serialization_and_clear_reset_state():
    memory = CodeMemory(max_context_tokens=200)
    large_content = "x = 1\n" * 100
    memory.add_file("large.py", large_content, purpose="Large file")

    context = memory.get_relevant_context("test.py", "test")
    payload = memory.to_dict()

    assert len(context) < len(large_content)
    assert "large.py" in payload["files"]
    assert "symbol_index" in payload

    memory.clear()

    assert memory.file_count == 0


def test_file_info_defaults():
    file_info = FileInfo(
        path="model.py",
        content="class Model: pass",
        purpose="Model definition",
        generation_order=0,
    )

    assert file_info.path == "model.py"
    assert file_info.purpose == "Model definition"
    assert file_info.dependencies == set()
    assert file_info.dependents == set()
