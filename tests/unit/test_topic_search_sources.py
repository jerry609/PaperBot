import pytest

from paperbot.application.workflows.topic_search_sources import (
    TopicSearchSourceRegistry,
    dedupe_sources,
)


class _DummySource:
    name = "dummy"

    def search(self, *, query, branches, show_per_branch):
        return []


def test_topic_source_registry_register_create_available():
    registry = TopicSearchSourceRegistry()
    registry.register("dummy", _DummySource)

    assert registry.available() == ["dummy"]
    source = registry.create("dummy")
    assert source.name == "dummy"


def test_topic_source_registry_unknown_source():
    registry = TopicSearchSourceRegistry()
    with pytest.raises(KeyError):
        registry.create("missing")


def test_dedupe_sources_normalizes_case_and_blank():
    assert dedupe_sources([" papers_cool ", "PAPERS_COOL", "", "custom"]) == [
        "papers_cool",
        "custom",
    ]
