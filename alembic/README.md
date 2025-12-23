# Alembic migrations

This directory contains DB migrations for PaperBot's local SQLite (and future Postgres) persistence.

## Quickstart

```bash
export PAPERBOT_DB_URL="sqlite:///data/paperbot.db"
alembic upgrade head
```

Notes:
- Local dev previously used `Base.metadata.create_all()` as a safety net; migrations are now the preferred path.
- If you have an existing local DB created before migrations existed, `alembic upgrade head` should be best-effort and
  skip tables/columns that already exist.

