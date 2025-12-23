from __future__ import annotations

import os
import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure `paperbot` can be imported when running Alembic without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _get_db_url() -> str:
    return os.getenv("PAPERBOT_DB_URL") or config.get_main_option("sqlalchemy.url")


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    # sqlite:///relative/path.db or sqlite:////abs/path.db
    if not db_url.startswith("sqlite:"):
        return
    if db_url.startswith("sqlite:///"):
        path = db_url.replace("sqlite:///", "", 1)
    elif db_url.startswith("sqlite:////"):
        path = db_url.replace("sqlite:////", "/", 1)
    else:
        # sqlite:// (rare) or sqlite:pure-memory
        return
    if path in (":memory:", ""):
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_target_metadata():
    # Import here so env.py doesn't import app code unless needed.
    from paperbot.infrastructure.stores.models import Base  # noqa

    return Base.metadata


target_metadata = get_target_metadata()


def run_migrations_offline() -> None:
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _get_db_url()
    _ensure_sqlite_parent_dir(configuration["sqlalchemy.url"])

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

