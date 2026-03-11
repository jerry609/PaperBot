from __future__ import annotations

from pathlib import Path

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
