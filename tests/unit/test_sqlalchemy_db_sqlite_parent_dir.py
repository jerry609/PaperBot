from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.stores.sqlalchemy_db import create_db_engine


def test_create_db_engine_creates_parent_dir_for_relative_sqlite(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "data").exists()

    engine = create_db_engine("sqlite:///data/test.db")
    try:
        assert (tmp_path / "data").is_dir()
        # Ensure connection works and file can be created.
        with engine.connect() as conn:
            conn.exec_driver_sql("select 1")
    finally:
        engine.dispose()

