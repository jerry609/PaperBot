# Repository Guidelines

## Project Structure

- `src/`: Python package source (PaperBot backend: agents, workflows, API).
- `tests/`: Pytest suite (unit + integration tests).
- `alembic/` + `alembic.ini`: Database migrations (SQLAlchemy/Alembic).
- `web/`: Next.js web dashboard (DeepCode Studio + UI).
- `cli/`: Terminal/CLI tooling.
- `docs/`: Design notes and plans (see `docs/PLAN.md`).
- `asset/`, `public/`: Images and static assets used by docs/UI.

## Build, Test, and Development Commands

**Backend (Python)**
- Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run API (dev): `python -m uvicorn src.paperbot.api.main:app --reload --port 8000`
- Tests: `pytest` (or `pytest -q`; integration tests live under `tests/integration/`)

**Web (Next.js)**
- Install: `cd web && npm install`
- Dev server: `cd web && npm run dev` (open `http://localhost:3000`)
- Lint: `cd web && npm run lint`
- Build: `cd web && npm run build`

## Coding Style & Naming Conventions

- Python: formatted with Black (`line-length = 100`) and import-sorted with isort (Black profile). Prefer `snake_case` for functions/vars and `PascalCase` for classes.
  - Format: `python -m black .`
  - Imports: `python -m isort .`
  - Types: `pyright` (config in `pyrightconfig.json`)
- TypeScript/React: follow existing component patterns in `web/src/` and keep Tailwind class usage consistent.
- Avoid editing generated/vendor outputs (`web/.next/`, `web/node_modules/`).

## Testing Guidelines

- Use pytest for Python. Add tests alongside existing suites and name files `test_*.py`.
- Prefer small, deterministic tests; mock external APIs where possible.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`).
- PRs should include:
  - Clear description of behavior change and any config/env impacts.
  - Linked issue (if applicable).
  - For UI changes in `web/`, include screenshots or short clips.
  - Evidence of validation (e.g., `pytest`, `npm run build`).

## Configuration Tips

- Copy `env.example` to `.env` for local runs: `cp env.example .env` and set required API keys (e.g., `OPENAI_API_KEY`).
