from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_from_dict_preserves_defaults_for_partial_repro_config() -> None:
    settings = Settings.from_dict({"repro": {"docker_image": "custom:latest"}})

    assert settings.repro["docker_image"] == "custom:latest"
    assert settings.repro["executor"] == "auto"
    assert settings.repro["timeout_sec"] == 300
    assert settings.repro["mem_limit"] == "1g"
    assert settings.repro["e2b_template"] == "Python3"


def test_from_dict_preserves_defaults_for_partial_dict_sections() -> None:
    settings = Settings.from_dict(
        {
            "data_source": {"dataset_name": "demo-dataset"},
            "report": {"template": "custom_report.md.j2"},
            "collab": {"host": {"model": "gpt-4.1-mini"}},
        }
    )

    assert settings.data_source == {
        "type": "api",
        "dataset_name": "demo-dataset",
        "dataset_path": None,
    }
    assert settings.report == {"template": "custom_report.md.j2"}
    assert settings.collab["enabled"] is False
    assert settings.collab["host"]["model"] == "gpt-4.1-mini"
    assert settings.collab["host"]["temperature"] == 0.3


def test_from_dict_loads_obsidian_section() -> None:
    settings = Settings.from_dict(
        {
            "obsidian": {
                "enabled": True,
                "vault_path": "/tmp/vault",
                "root_dir": "Research Notes",
                "paper_template_path": "/tmp/paper.md.j2",
            }
        }
    )

    assert settings.obsidian.enabled is True
    assert settings.obsidian.vault_path == "/tmp/vault"
    assert settings.obsidian.root_dir == "Research Notes"
    assert settings.obsidian.paper_template_path == "/tmp/paper.md.j2"


def test_load_from_file_merges_partial_nested_dicts(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
repro:
  docker_image: "python:3.12"
data_source:
  dataset_path: "/tmp/data.csv"
collab:
  enabled: true
  host:
    top_p: 0.7
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.load_from_file(str(config_path))

    assert settings.repro["docker_image"] == "python:3.12"
    assert settings.repro["executor"] == "auto"
    assert settings.data_source["type"] == "api"
    assert settings.data_source["dataset_path"] == "/tmp/data.csv"
    assert settings.collab["enabled"] is True
    assert settings.collab["host"]["model"] == "gpt-4o-mini"
    assert settings.collab["host"]["top_p"] == 0.7


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        ({"collab": None}, "collab"),
        ({"data_source": "invalid"}, "data_source"),
        ({"report": "invalid"}, "report"),
        ({"repro": None}, "repro"),
    ],
)
def test_from_dict_raises_validation_error_for_invalid_nested_section_types(
    payload: dict,
    field_name: str,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings.from_dict(payload)

    assert field_name in str(excinfo.value)


def test_load_environment_variables_overrides_obsidian(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_ENABLED", "true")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_VAULT_PATH", "/tmp/obsidian-vault")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_ROOT_DIR", "PaperBot Notes")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_PAPER_TEMPLATE", "/tmp/paper.md.j2")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_AUTO_EXPORT", "false")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_AUTO_SYNC_TRACKS", "false")
    monkeypatch.setenv("PAPERBOT_OBSIDIAN_EXPORT_LIMIT", "42")

    settings.load_environment_variables()

    assert settings.obsidian.enabled is True
    assert settings.obsidian.vault_path == "/tmp/obsidian-vault"
    assert settings.obsidian.root_dir == "PaperBot Notes"
    assert settings.obsidian.paper_template_path == "/tmp/paper.md.j2"
    assert settings.obsidian.auto_export_on_save is False
    assert settings.obsidian.auto_sync_tracks is False
    assert settings.obsidian.export_limit == 42
