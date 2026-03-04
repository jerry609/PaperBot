"""Unit tests for issue #162: CodeMemory persistence via ReproExperienceStore."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from paperbot.infrastructure.stores.repro_experience_store import ReproExperienceStore
from paperbot.repro.memory.code_memory import CodeMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(db_url: str = "sqlite://") -> ReproExperienceStore:
    return ReproExperienceStore(db_url=db_url, auto_create_schema=True)


def _make_memory(store=None) -> CodeMemory:
    return CodeMemory(max_context_tokens=4000, experience_store=store)


# ---------------------------------------------------------------------------
# ReproExperienceStore CRUD
# ---------------------------------------------------------------------------

class TestReproExperienceStore:
    def test_add_and_get_by_paper_id(self):
        store = _make_store()
        store.add(
            pattern_type="success_pattern",
            content="Generated model.py",
            paper_id="arxiv:1234",
            code_snippet="class Model: pass",
        )
        rows = store.get_by_paper_id("arxiv:1234")
        assert len(rows) == 1
        assert rows[0]["pattern_type"] == "success_pattern"
        assert rows[0]["content"] == "Generated model.py"
        assert rows[0]["paper_id"] == "arxiv:1234"

    def test_add_and_get_by_pack_id(self):
        store = _make_store()
        store.add(
            pattern_type="failure_reason",
            content="[syntax] fixed indent",
            pack_id="ctxp_abc",
        )
        rows = store.get_by_pack_id("ctxp_abc")
        assert len(rows) == 1
        assert rows[0]["pack_id"] == "ctxp_abc"

    def test_filter_by_pattern_type(self):
        store = _make_store()
        store.add(pattern_type="success_pattern", content="ok", paper_id="p1")
        store.add(pattern_type="failure_reason", content="fail", paper_id="p1")
        rows = store.get_by_paper_id("p1", pattern_type="success_pattern")
        assert all(r["pattern_type"] == "success_pattern" for r in rows)
        assert len(rows) == 1

    def test_invalid_pattern_type_raises(self):
        store = _make_store()
        with pytest.raises(ValueError):
            store.add(pattern_type="unknown_type", content="bad")

    def test_returns_empty_for_unknown_paper(self):
        store = _make_store()
        assert store.get_by_paper_id("does_not_exist") == []

    def test_limit_respected(self):
        store = _make_store()
        for i in range(10):
            store.add(pattern_type="success_pattern", content=f"file_{i}.py", paper_id="p_limit")
        rows = store.get_by_paper_id("p_limit", limit=3)
        assert len(rows) == 3

    def test_deduplicates_same_paper_type_and_content(self):
        store = _make_store()
        first = store.add(
            pattern_type="success_pattern",
            content="Successfully generated model.py",
            paper_id="p_dup",
        )
        second = store.add(
            pattern_type="success_pattern",
            content="Successfully generated model.py",
            paper_id="p_dup",
            pack_id="ctxp_new",
        )
        rows = store.get_by_paper_id("p_dup")
        assert len(rows) == 1
        assert first.id == second.id
        assert rows[0]["pack_id"] == "ctxp_new"


# ---------------------------------------------------------------------------
# CodeMemory persistence methods
# ---------------------------------------------------------------------------

class TestCodeMemoryRecordMethods:
    def test_record_success_pattern_calls_store(self):
        mock_store = MagicMock()
        mem = _make_memory(store=mock_store)
        mem.record_success_pattern(paper_id="p1", filepath="model.py", code_snippet="x=1")
        mock_store.add.assert_called_once()
        kwargs = mock_store.add.call_args.kwargs
        assert kwargs["pattern_type"] == "success_pattern"
        assert "model.py" in kwargs["content"]

    def test_record_verified_structure_calls_store(self):
        mock_store = MagicMock()
        mem = _make_memory(store=mock_store)
        mem.record_verified_structure(paper_id="p1", description="syntax+imports passed")
        mock_store.add.assert_called_once()
        kwargs = mock_store.add.call_args.kwargs
        assert kwargs["pattern_type"] == "verified_structure"

    def test_record_failure_reason_calls_store(self):
        mock_store = MagicMock()
        mem = _make_memory(store=mock_store)
        mem.record_failure_reason(
            paper_id="p1", error_type="SYNTAX", fix_applied="fixed indent at line 5"
        )
        mock_store.add.assert_called_once()
        kwargs = mock_store.add.call_args.kwargs
        assert kwargs["pattern_type"] == "failure_reason"
        assert "SYNTAX" in kwargs["content"]

    def test_no_store_skips_silently(self):
        mem = _make_memory(store=None)
        # Should not raise
        mem.record_success_pattern(paper_id="p1", filepath="x.py")
        mem.record_verified_structure(paper_id="p1", description="ok")
        mem.record_failure_reason(paper_id="p1", error_type="LOGIC", fix_applied="fixed")

    def test_store_exception_does_not_propagate(self):
        mock_store = MagicMock()
        mock_store.add.side_effect = RuntimeError("db gone")
        mem = _make_memory(store=mock_store)
        # Should not raise despite store error
        mem.record_success_pattern(paper_id="p1", filepath="x.py")


# ---------------------------------------------------------------------------
# CodeMemory.load_experiences_from_db
# ---------------------------------------------------------------------------

class TestLoadExperiencesFromDb:
    def test_loads_and_stores_prior_experiences(self):
        mock_store = MagicMock()
        mock_store.get_by_paper_id.return_value = [
            {"id": 1, "pattern_type": "success_pattern", "content": "Generated model.py",
             "pack_id": None, "paper_id": "p1", "code_snippet": None, "created_at": None},
        ]
        mock_store.get_by_pack_id.return_value = []
        mem = _make_memory(store=mock_store)
        mem.load_experiences_from_db("p1")
        assert len(mem._prior_experiences) == 1
        mock_store.get_by_paper_id.assert_called_once_with("p1", limit=20)

    def test_no_store_skips_silently(self):
        mem = _make_memory(store=None)
        mem.load_experiences_from_db("p1")
        assert mem._prior_experiences == []

    def test_prior_experiences_injected_into_context(self):
        mock_store = MagicMock()
        mock_store.get_by_paper_id.return_value = [
            {"id": 1, "pattern_type": "success_pattern", "content": "Generated model.py",
             "pack_id": None, "paper_id": "p1", "code_snippet": None, "created_at": None},
        ]
        mock_store.get_by_pack_id.return_value = []
        mem = _make_memory(store=mock_store)
        mem.load_experiences_from_db("p1")
        ctx = mem.get_relevant_context("trainer.py", "training loop")
        assert "Prior Experience" in ctx or "success_pattern" in ctx

    def test_clear_resets_prior_experiences(self):
        mock_store = MagicMock()
        mock_store.get_by_paper_id.return_value = [
            {"id": 1, "pattern_type": "success_pattern", "content": "x",
             "pack_id": None, "paper_id": "p1", "code_snippet": None, "created_at": None},
        ]
        mock_store.get_by_pack_id.return_value = []
        mem = _make_memory(store=mock_store)
        mem.load_experiences_from_db("p1")
        assert len(mem._prior_experiences) == 1
        mem.clear()
        assert mem._prior_experiences == []
