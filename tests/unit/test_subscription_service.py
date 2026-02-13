from pathlib import Path

import pytest
import yaml

from paperbot.infrastructure.services.subscription_service import SubscriptionService


def _write_config(path: Path, scholars: list[dict]):
    payload = {
        "subscriptions": {
            "scholars": scholars,
            "settings": {"check_interval": "weekly"},
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_subscription_service_add_update_remove(tmp_path: Path):
    cfg = tmp_path / "scholars.yaml"
    _write_config(cfg, [])

    service = SubscriptionService(config_path=str(cfg))

    added = service.add_scholar(
        {
            "name": "Alice",
            "semantic_scholar_id": "1001",
            "keywords": ["rag", "agent"],
            "affiliations": ["Lab A"],
        }
    )
    assert added["name"] == "Alice"

    updated = service.update_scholar(
        "1001",
        {
            "name": "Alice Updated",
            "keywords": ["safety"],
            "affiliations": ["Lab B"],
        },
    )
    assert updated is not None
    assert updated["name"] == "Alice Updated"
    assert updated["keywords"] == ["safety"]
    assert updated["affiliations"] == ["Lab B"]

    removed = service.remove_scholar("1001")
    assert removed is not None
    assert removed["semantic_scholar_id"] == "1001"

    reloaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert reloaded["subscriptions"]["scholars"] == []


def test_subscription_service_update_duplicate_semantic_id(tmp_path: Path):
    cfg = tmp_path / "scholars.yaml"
    _write_config(
        cfg,
        [
            {"name": "Alice", "semantic_scholar_id": "1001"},
            {"name": "Bob", "semantic_scholar_id": "1002"},
        ],
    )

    service = SubscriptionService(config_path=str(cfg))

    with pytest.raises(ValueError, match="already exists"):
        service.update_scholar("1001", {"semantic_scholar_id": "1002"})
