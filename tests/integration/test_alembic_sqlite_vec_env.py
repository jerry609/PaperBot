from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def test_sqlite_vec_is_loaded_for_followup_migrations(tmp_path):
    pytest.importorskip("sqlite_vec")

    db_path = tmp_path / "alembic-sqlite-vec.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    env = os.environ.copy()
    env["PAPERBOT_DB_URL"] = db_url

    def run_upgrade(revision: str) -> subprocess.CompletedProcess[str]:
        script = (
            "from alembic import command; "
            "from alembic.config import Config; "
            f"cfg = Config(r'{ALEMBIC_INI.as_posix()}'); "
            f"cfg.set_main_option('script_location', r'{(REPO_ROOT / 'alembic').as_posix()}'); "
            f"command.upgrade(cfg, '{revision}')"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    result_0021 = run_upgrade("0021_repro_code_experience")
    assert result_0021.returncode == 0, result_0021.stdout + result_0021.stderr

    import sqlite_vec

    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[8])"
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_items_vec_ai
            AFTER INSERT ON memory_items
            WHEN new.embedding IS NOT NULL
            BEGIN
                INSERT OR REPLACE INTO vec_items(rowid, embedding) VALUES (new.id, new.embedding);
            END
            """
        )
        objects = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            ).fetchall()
        }

    assert "vec_items" in objects
    assert "memory_items_vec_ai" in objects

    result_0022 = run_upgrade("0022_repro_experience_dedup")
    assert result_0022.returncode == 0, result_0022.stdout + result_0022.stderr
