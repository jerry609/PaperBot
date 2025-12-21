# Alembic migrations

This directory contains DB migrations for PaperBot's local SQLite (and future Postgres) persistence.

## Quickstart

```bash
export PAPERBOT_DB_URL="sqlite:///data/paperbot.db"
alembic upgrade head
```


