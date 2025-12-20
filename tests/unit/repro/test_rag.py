# tests/unit/repro/test_rag.py
"""
Unit tests for CodeKnowledgeBase (RAG) module.
"""

import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from repro.rag.knowledge_base import CodeKnowledgeBase, CodePattern, BUILTIN_PATTERNS


class TestCodePattern(unittest.TestCase):
    """Tests for CodePattern dataclass."""

    def test_pattern_creation(self):
        pattern = CodePattern(
            name="test_pattern",
            description="A test pattern",
            code="def test(): pass",
            tags=["test", "unit"],
            source="test suite",
        )

        self.assertEqual(pattern.name, "test_pattern")
        self.assertEqual(pattern.description, "A test pattern")
        self.assertEqual(pattern.code, "def test(): pass")
        self.assertEqual(pattern.tags, ["test", "unit"])
        self.assertEqual(pattern.source, "test suite")
        self.assertEqual(pattern.language, "python")  # default

    def test_to_context(self):
        pattern = CodePattern(
            name="my_pattern",
            description="Does something useful",
            code="x = 1\ny = 2",
            tags=["example"],
        )

        context = pattern.to_context()

        self.assertIn("# Pattern: my_pattern", context)
        self.assertIn("# Does something useful", context)
        self.assertIn("x = 1", context)
        self.assertIn("y = 2", context)


class TestCodeKnowledgeBase(unittest.TestCase):
    """Tests for CodeKnowledgeBase class."""

    def setUp(self):
        self.kb = CodeKnowledgeBase()

    def test_add_pattern(self):
        pattern = CodePattern(
            name="test",
            description="Test",
            code="# test",
            tags=["testing"],
        )

        self.kb.add_pattern(pattern)

        self.assertEqual(len(self.kb.patterns), 1)
        self.assertIn("test", self.kb.patterns)

    def test_search_by_tag(self):
        pattern1 = CodePattern(
            name="pytorch_model",
            description="PyTorch model",
            code="class Model: pass",
            tags=["pytorch", "model", "deep learning"],
        )
        pattern2 = CodePattern(
            name="tensorflow_model",
            description="TensorFlow model",
            code="class TFModel: pass",
            tags=["tensorflow", "model", "deep learning"],
        )

        self.kb.add_pattern(pattern1)
        self.kb.add_pattern(pattern2)

        # Search for pytorch
        results = self.kb.search("pytorch", k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "pytorch_model")

        # Search for model (should find both)
        results = self.kb.search("model", k=3)
        self.assertEqual(len(results), 2)

    def test_search_by_name(self):
        pattern = CodePattern(
            name="transformer_encoder",
            description="Encoder block",
            code="# encoder",
            tags=["attention"],
        )

        self.kb.add_pattern(pattern)

        results = self.kb.search("transformer encoder", k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "transformer_encoder")

    def test_search_by_description(self):
        pattern = CodePattern(
            name="pattern1",
            description="Training loop with validation",
            code="# code",
            tags=["training"],
        )

        self.kb.add_pattern(pattern)

        results = self.kb.search("validation", k=3)
        self.assertEqual(len(results), 1)

    def test_search_scoring(self):
        # Pattern with tag match should score higher than description match
        pattern1 = CodePattern(
            name="high_score",
            description="Other description",
            code="# code",
            tags=["pytorch", "training"],
        )
        pattern2 = CodePattern(
            name="low_score",
            description="Something about pytorch training",
            code="# code",
            tags=["other"],
        )

        self.kb.add_pattern(pattern1)
        self.kb.add_pattern(pattern2)

        results = self.kb.search("pytorch training", k=2)

        # Tag matches score higher
        self.assertEqual(results[0].name, "high_score")

    def test_search_limit_k(self):
        for i in range(10):
            pattern = CodePattern(
                name=f"pattern_{i}",
                description="Test pattern",
                code="# code",
                tags=["test"],
            )
            self.kb.add_pattern(pattern)

        results = self.kb.search("test", k=3)
        self.assertEqual(len(results), 3)

    def test_search_no_results(self):
        pattern = CodePattern(
            name="pytorch",
            description="PyTorch",
            code="# code",
            tags=["pytorch"],
        )
        self.kb.add_pattern(pattern)

        results = self.kb.search("tensorflow java", k=3)
        self.assertEqual(len(results), 0)

    def test_get_by_tag(self):
        pattern1 = CodePattern(
            name="p1",
            description="D1",
            code="#",
            tags=["common", "a"],
        )
        pattern2 = CodePattern(
            name="p2",
            description="D2",
            code="#",
            tags=["common", "b"],
        )

        self.kb.add_pattern(pattern1)
        self.kb.add_pattern(pattern2)

        results = self.kb.get_by_tag("common")
        self.assertEqual(len(results), 2)

        results = self.kb.get_by_tag("a")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "p1")

    def test_get_pattern(self):
        pattern = CodePattern(
            name="my_pattern",
            description="D",
            code="#",
            tags=[],
        )
        self.kb.add_pattern(pattern)

        result = self.kb.get_pattern("my_pattern")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "my_pattern")

        result = self.kb.get_pattern("nonexistent")
        self.assertIsNone(result)

    def test_list_tags(self):
        pattern1 = CodePattern(name="p1", description="", code="", tags=["alpha", "beta"])
        pattern2 = CodePattern(name="p2", description="", code="", tags=["gamma", "alpha"])

        self.kb.add_pattern(pattern1)
        self.kb.add_pattern(pattern2)

        tags = self.kb.list_tags()

        self.assertIn("alpha", tags)
        self.assertIn("beta", tags)
        self.assertIn("gamma", tags)
        self.assertEqual(tags, sorted(tags))  # Should be sorted

    def test_list_patterns(self):
        for i in range(3):
            pattern = CodePattern(name=f"p{i}", description="", code="", tags=[])
            self.kb.add_pattern(pattern)

        patterns = self.kb.list_patterns()

        self.assertEqual(len(patterns), 3)
        self.assertIn("p0", patterns)
        self.assertIn("p1", patterns)
        self.assertIn("p2", patterns)

    def test_to_dict(self):
        pattern = CodePattern(
            name="test",
            description="Test pattern",
            code="x = 1",
            tags=["tag1", "tag2"],
            source="unit test",
        )
        self.kb.add_pattern(pattern)

        data = self.kb.to_dict()

        self.assertIn("patterns", data)
        self.assertIn("tag_index", data)
        self.assertIn("test", data["patterns"])
        self.assertEqual(data["patterns"]["test"]["description"], "Test pattern")
        self.assertEqual(data["patterns"]["test"]["code_length"], 5)

    def test_case_insensitive_search(self):
        pattern = CodePattern(
            name="PyTorch_Model",
            description="A PYTORCH model",
            code="#",
            tags=["PyTorch", "MODEL"],
        )
        self.kb.add_pattern(pattern)

        # All variations should work
        results = self.kb.search("pytorch", k=3)
        self.assertEqual(len(results), 1)

        results = self.kb.search("PYTORCH", k=3)
        self.assertEqual(len(results), 1)

        results = self.kb.search("PyTorch", k=3)
        self.assertEqual(len(results), 1)


class TestBuiltinPatterns(unittest.TestCase):
    """Tests for built-in patterns."""

    def test_from_builtin(self):
        kb = CodeKnowledgeBase.from_builtin()

        self.assertGreater(len(kb.patterns), 0)
        self.assertGreater(len(kb.list_tags()), 0)

    def test_builtin_patterns_valid(self):
        # All built-in patterns should have required fields
        for pattern in BUILTIN_PATTERNS:
            self.assertTrue(pattern.name, "Pattern must have a name")
            self.assertTrue(pattern.description, "Pattern must have a description")
            self.assertTrue(pattern.code, "Pattern must have code")
            self.assertTrue(pattern.tags, "Pattern must have tags")

    def test_builtin_training_loop(self):
        kb = CodeKnowledgeBase.from_builtin()

        results = kb.search("pytorch training loop", k=3)

        self.assertGreater(len(results), 0)
        # Should find the training loop pattern
        names = [r.name for r in results]
        self.assertIn("pytorch_training_loop", names)

    def test_builtin_transformer(self):
        kb = CodeKnowledgeBase.from_builtin()

        results = kb.search("transformer attention encoder", k=3)

        self.assertGreater(len(results), 0)
        names = [r.name for r in results]
        self.assertIn("transformer_encoder_block", names)

    def test_builtin_dataloader(self):
        kb = CodeKnowledgeBase.from_builtin()

        results = kb.search("dataset dataloader", k=3)

        self.assertGreater(len(results), 0)

    def test_builtin_config(self):
        kb = CodeKnowledgeBase.from_builtin()

        results = kb.search("config dataclass settings", k=3)

        self.assertGreater(len(results), 0)

    def test_builtin_checkpoint(self):
        kb = CodeKnowledgeBase.from_builtin()

        results = kb.search("checkpoint save load model", k=3)

        self.assertGreater(len(results), 0)
        names = [r.name for r in results]
        self.assertIn("checkpoint_manager", names)

    def test_builtin_patterns_have_valid_python(self):
        """Verify all built-in pattern code is valid Python."""
        for pattern in BUILTIN_PATTERNS:
            try:
                # Try to compile the code (syntax check)
                compile(pattern.code, f"<{pattern.name}>", "exec")
            except SyntaxError as e:
                self.fail(f"Pattern {pattern.name} has invalid Python: {e}")


if __name__ == "__main__":
    unittest.main()
