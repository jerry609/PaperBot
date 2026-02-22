# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Python Backend

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run database migrations (required before first run)
export PAPERBOT_DB_URL="sqlite:///data/paperbot.db"
alembic upgrade head

# Start API server (dev mode with auto-reload)
python -m uvicorn src.paperbot.api.main:app --reload --port 8000

# Code formatting
python -m black .
python -m isort .

# Type checking
pyright src/
```

### Testing

```bash
# Run all tests
pytest -q

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run CI offline gates (same tests as GitHub Actions)
PYTHONPATH=src pytest -q \
  tests/unit/test_scholar_from_config.py \
  tests/unit/test_source_registry_modes.py \
  tests/integration/test_eventlog_sqlalchemy.py \
  tests/integration/test_crawler_contract_parsers.py

# Run evaluation smoke tests
python evals/runners/run_scholar_pipeline_smoke.py
python evals/runners/run_track_pipeline_smoke.py
python evals/runners/run_eventlog_replay_smoke.py
```

### Web Dashboard (Next.js)

```bash
cd web
npm install
npm run dev      # Dev server at http://localhost:3000
npm run build    # Production build
npm run lint     # ESLint
```

### Terminal CLI (Ink/React)

```bash
cd cli
npm install
npm run build
npm start
```

## Architecture Overview

PaperBot is a multi-agent research workflow framework with three main components:

1. **Python Backend** (`src/paperbot/`) - FastAPI server with SSE streaming, multi-agent orchestration
2. **Web Dashboard** (`web/`) - Next.js 16 + React 19 + Tailwind CSS
3. **Terminal UI** (`cli/`) - Ink/React CLI

### Core Python Structure

```
src/paperbot/
├── agents/              # Multi-agent implementations (BaseAgent in base.py)
│   ├── research/        # Paper analysis agents
│   ├── scholar_tracking/# Scholar monitoring agents
│   ├── review/          # Peer review simulation
│   ├── verification/    # Claim verification
│   └── prompts/         # Agent prompt templates
├── api/                 # FastAPI server
│   ├── main.py          # App setup, CORS, routers
│   ├── streaming.py     # SSE utilities
│   └── routes/          # Endpoint handlers (track, analyze, gen_code, review, etc.)
├── core/                # Core abstractions
│   ├── collaboration/   # AgentCoordinator, ScoreShareBus, FailFast
│   ├── di/              # Dependency injection container
│   ├── pipeline/        # Task pipeline framework
│   └── report_engine/   # Report generation with Jinja2 templates
├── repro/               # Paper2Code pipeline (ReproAgent)
│   ├── orchestrator.py  # Multi-stage execution
│   ├── agents/          # Planning/Coding/Debugging agents
│   ├── memory/          # CodeMemory - cross-file context with AST indexing
│   └── rag/             # CodeRAG - pattern retrieval
├── context_engine/      # Research context & track routing
├── infrastructure/      # External services and persistence
│   ├── llm/             # LLM client (OpenAI/Anthropic)
│   ├── stores/          # SQLAlchemy repositories
│   ├── event_log/       # Event persistence
│   └── api_clients/     # Semantic Scholar, GitHub clients
└── workflows/           # Workflow orchestration, scheduler
```

### Key Architectural Patterns

- **Multi-Agent Orchestration**: `AgentCoordinator` registers agents, broadcasts tasks, collects results. `ScoreShareBus` enables cross-stage evaluation sharing. `FailFastEvaluator` stops low-quality work early.
- **Paper2Code Pipeline**: Multi-stage (Planning → Coding → Verification → Debugging) with `CodeMemory` for stateful cross-file context and `CodeRAG` for pattern retrieval.
- **Repository Pattern**: Interface definitions in `application/ports/`, implementations in `infrastructure/stores/`.
- **DI Container**: Single `Container.instance()` holds all services (`core/di/`).

### API Endpoints

Key SSE-streaming endpoints:
- `GET /api/track` - Scholar tracking
- `POST /api/analyze` - Paper analysis
- `POST /api/gen-code` - Paper2Code generation
- `POST /api/review` - Deep review simulation
- `POST /api/research/*` - Personalized research context & tracks
- `GET/POST /api/runbook/*`, `/api/sandbox/*` - DeepCode Studio workspace & execution

## Configuration

Copy `env.example` to `.env` and configure:

```bash
# Required for LLM features
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
SEMANTIC_SCHOLAR_API_KEY=...
GITHUB_TOKEN=...

# Runtime
PAPERBOT_DB_URL=sqlite:///data/paperbot.db
```

Configuration files:
- `config/config.yaml` - Main app config (models, venues, thresholds)
- `config/settings.py` - Pydantic settings
- `config/scholar_subscriptions.yaml` - Tracked scholars

## Code Style

- **Python**: Black (line-length 100), isort (Black profile), pyright for type checking
- **TypeScript**: Follow existing patterns in `web/src/`
- **Commits**: Conventional Commits format (`feat:`, `fix:`, `refactor:`, `docs:`)

## Adding New Components

### New Agent
1. Create `src/paperbot/agents/{feature}/agent.py` extending `BaseAgent`
2. Implement `async def execute(self, input) -> ExecutionResult`
3. Add prompt templates in `agents/prompts/{feature}/`
4. Register in DI container or coordinator
5. Add unit tests in `tests/unit/`

### New API Endpoint
1. Create `src/paperbot/api/routes/{feature}.py` with `APIRouter`
2. Use SSE streaming if needed (see `streaming.py`)
3. Register router in `api/main.py`
4. Add integration tests

### Database Changes
1. Add model in `infrastructure/stores/models.py`
2. Create migration: `alembic revision --autogenerate -m "description"`
3. Apply: `alembic upgrade head`
