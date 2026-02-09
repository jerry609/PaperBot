# AGENTS.md

This document is the working contract for contributors and coding agents in this repository.
Follow it as the default behavior unless a task-specific instruction says otherwise.

## 1) Repository Overview

PaperBot is a mixed Python + Next.js project.

- `src/` - Python backend package (agents, workflows, API)
- `tests/` - pytest suite (unit + integration)
- `alembic/` + `alembic.ini` - SQLAlchemy/Alembic migrations
- `web/` - Next.js dashboard (DeepCode Studio + UI)
- `cli/` - CLI and terminal tooling
- `docs/` - design notes and plans (`docs/PLAN.md`)
- `asset/`, `public/` - static assets for docs/UI

## 2) Local Development

### Backend (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.paperbot.api.main:app --reload --port 8000
```

### Web (Next.js)

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000` for the web app.

## 3) Build, Lint, and Test

### Python

```bash
pytest
# or
pytest -q
python -m black .
python -m isort .
pyright
```

### Web

```bash
cd web
npm run lint
npm run build
```

Notes:
- Integration tests live under `tests/integration/`.
- Prefer deterministic tests and mock external APIs when possible.

## 4) Style Conventions

### Python
- Black formatting, `line-length = 100`
- isort with Black profile
- Naming: `snake_case` for functions/variables, `PascalCase` for classes

### TypeScript/React
- Follow established patterns in `web/src/`
- Keep Tailwind usage consistent with nearby components

Do not edit generated/vendor directories:
- `web/.next/`
- `web/node_modules/`

## 5) Commits and Pull Requests

### Commits
Use Conventional Commits:
- `feat: ...`
- `fix: ...`
- `refactor: ...`
- `docs: ...`

### Pull Requests
Each PR should include:
- What changed and why
- Any config/environment impact
- Linked issue (if available)
- Validation evidence (for example: `pytest`, `npm run build`)
- Screenshots or clips for UI changes under `web/`

## 6) Configuration

Bootstrap local env:

```bash
cp env.example .env
```

Then fill required keys (for example `OPENAI_API_KEY`).

## 7) Agent Working Rules

- Make the smallest safe change that satisfies the request.
- Keep edits local and avoid unrelated refactors.
- Prefer fast, targeted checks before broad test runs.
- If behavior changes, add or update tests.
- When uncertain, document assumptions explicitly in the final note.
