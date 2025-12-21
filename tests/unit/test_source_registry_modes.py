from paperbot.application.registries.default_sources import register_default_sources
from paperbot.application.registries.source_registry import SourceRegistry, AcquisitionMode


def test_default_sources_include_twitter_import_only_and_disabled():
    reg = register_default_sources(SourceRegistry())
    x = reg.get("twitter_x")
    assert x is not None
    assert x.acquisition_mode == AcquisitionMode.import_only
    assert x.enabled_by_default is False


def test_default_sources_include_semantic_scholar_api_first():
    reg = register_default_sources(SourceRegistry())
    s2 = reg.get("semantic_scholar")
    assert s2 is not None
    assert s2.acquisition_mode == AcquisitionMode.api_first
    assert s2.enabled_by_default is True


