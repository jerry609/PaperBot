from paperbot.domain.scholar import Scholar


def test_scholar_from_config_minimal():
    s = Scholar.from_config({"name": "Dawn Song", "semantic_scholar_id": "1741101", "keywords": ["AI Security"]})
    assert s.name == "Dawn Song"
    assert s.semantic_scholar_id == "1741101"
    assert s.scholar_id == "1741101"
    assert "AI Security" in (s.keywords or [])


