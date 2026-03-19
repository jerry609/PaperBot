# Codebase Structure

**Analysis Date:** 2026-03-15

## Directory Layout

```
PaperBot/
├── src/paperbot/                # Python backend source code
│   ├── domain/                  # Business domain models
│   ├── application/             # Use cases, services, workflows
│   ├── core/                    # Core abstractions and infrastructure
│   ├── agents/                  # Multi-agent implementations
│   ├── infrastructure/          # External integrations and persistence
│   ├── api/                     # FastAPI HTTP server
│   ├── repro/                   # Paper2Code pipeline
│   ├── memory/                  # Persistent agent memory
│   ├── mcp/                     # Model Context Protocol server
│   ├── context_engine/          # Research context management
│   ├── workflows/               # Workflow definitions (deprecated, use application/workflows)
│   ├── presentation/            # UI components and formatters
│   ├── utils/                   # Shared utilities
│   ├── compat/                  # Compatibility shims
│   └── __init__.py              # Lazy-load export facade
├── web/                         # Next.js dashboard UI
│   ├── src/app/                 # App Router pages
│   ├── src/components/          # React components
│   ├── src/lib/                 # Utilities and API clients
│   ├── src/hooks/               # Custom React hooks
│   └── src/types/               # TypeScript types
├── cli/                         # Ink/React CLI application
│   └── src/                     # CLI source code
├── config/                      # Configuration files
│   ├── config.yaml              # Application configuration
│   ├── settings.py              # Dataclass settings
│   ├── validated_settings.py    # Pydantic validation
│   └── models.py                # Configuration models
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── evals/                       # Evaluation runners and benchmarks
│   ├── runners/                 # Test runners (smoke tests, benchmarks)
│   ├── fixtures/                # Test fixtures and mock data
│   ├── memory/                  # Memory evaluation tests
│   ├── scorers/                 # Evaluation scorers
│   └── reports/                 # Generated reports
├── alembic/                     # Database migrations
│   └── versions/                # Individual migration files
├── docs/                        # Project documentation
├── scripts/                     # Utility scripts
├── .planning/                   # GSD planning documents
├── .claude/                     # Claude Code configuration
├── data/                        # Runtime data directory
├── logs/                        # Application logs
├── reports/                     # Generated reports
├── datasets/                    # Evaluation datasets
├── pyproject.toml               # Python project configuration
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Developer instructions
├── main.py                      # Legacy standalone entry point
├── Makefile                     # Build commands
├── alembic.ini                  # Alembic configuration
└── README.md                    # Project readme
```

## Directory Purposes

**`src/paperbot/domain/`:**
- Purpose: Business domain models independent of frameworks
- Contains: PaperMeta, Scholar, Track, Harvest, Enrichment, Feedback, Identity, Influence
- Key files: `paper.py`, `scholar.py`, `influence.py`, `harvest.py`

**`src/paperbot/application/`:**
- Purpose: Application layer implementing use cases
- Contains: Services, workflows, ports (abstractions), registries, collaboration schemas
- Key directories:
  - `services/`: LLMService, AnchorService, PaperSearchService, EnrichmentPipeline
  - `workflows/`: DailyPaper, ScholarPipeline, HarvestPipeline, UnifiedTopicSearch
  - `ports/`: Interface definitions for infrastructure (EventLogPort, PaperSearchPort, MemoryPort, etc.)
  - `collaboration/`: Agent message schemas and coordination models
  - `registries/`: Source and provider registries

**`src/paperbot/core/`:**
- Purpose: Core abstractions and foundational patterns
- Contains: DI container, Pipeline framework, Executable abstraction, Error handling
- Key files:
  - `di/container.py`: Lightweight DI with singleton support
  - `pipeline/pipeline.py`: Declarative stage orchestration
  - `abstractions/executable.py`: Executable interface and ExecutionResult
  - `workflow_coordinator.py`: Multi-agent orchestration
  - `fail_fast.py`: Early termination evaluator
  - `collaboration/score_bus.py`: Cross-stage score sharing

**`src/paperbot/agents/`:**
- Purpose: Specialized agents for different analysis tasks
- Contains: ResearchAgent, CodeAnalysisAgent, QualityAgent, ReviewerAgent, VerificationAgent
- Key files:
  - `base.py`: BaseAgent with Template Method pattern
  - `research/agent.py`: Research and novelty analysis
  - `code_analysis/agent.py`: Code quality evaluation
  - `quality/agent.py`: Paper quality assessment
  - `review/agent.py`: Peer review simulation
  - `verification/agent.py`: Claim verification

**`src/paperbot/infrastructure/`:**
- Purpose: External integrations and persistence
- Key subdirectories:
  - `llm/providers/`: OpenAI, Anthropic, Ollama LLM implementations
  - `stores/`: SQLAlchemy repositories (PaperStore, ResearchStore, MemoryStore, etc.)
  - `connectors/`: Data source APIs (ArXiv, Reddit, HuggingFace, X, etc.)
  - `crawling/`: HTTP downloader and parser layer
  - `adapters/`: Search API adapters
  - `api_clients/`: Third-party clients (Semantic Scholar, GitHub)
  - `harvesters/`: Bulk paper collection
  - `event_log/`: Event persistence implementations
  - `queue/`: ARQ async job queue
  - `services/`: Email, push notifications

**`src/paperbot/api/`:**
- Purpose: HTTP API endpoints
- Contains:
  - `main.py`: FastAPI app setup and router registration
  - `routes/`: 25+ endpoint handlers (track, analyze, research, gen_code, etc.)
  - `streaming.py`: Server-Sent Events utilities
  - `middleware/`: Auth, CORS, rate limiting
  - `error_handling/`: Centralized error handling

**`src/paperbot/repro/`:**
- Purpose: Paper2Code pipeline for code generation
- Contains:
  - `orchestrator.py`: Multi-stage pipeline orchestration
  - `repro_agent.py`: Main coordination agent
  - `agents/`: Planning, Coding, Debugging, Verification agents
  - `nodes/`: Individual pipeline stage implementations
  - `memory/`: CodeMemory, SymbolIndex, CodeRAG
  - Executors: `docker_executor.py`, `e2b_executor.py`

**`src/paperbot/memory/`:**
- Purpose: Persistent agent memory and learning
- Contains: Extractors, parsers, evaluators, schema

**`src/paperbot/mcp/`:**
- Purpose: Model Context Protocol server
- Contains: Tool and resource definitions, server setup

**`web/`:**
- Purpose: Next.js browser-based UI
- Key directories:
  - `src/app/`: App Router pages (dashboard, papers, research, scholars, settings, studio, workflows, wiki)
  - `src/components/`: React components organized by feature
  - `src/lib/`: API clients, utilities
  - `src/types/`: TypeScript type definitions

**`cli/`:**
- Purpose: Terminal CLI using Ink/React
- Contains: TUI components and utilities

**`config/`:**
- Purpose: Application configuration
- Files:
  - `config.yaml`: Main app config (models, venues, thresholds)
  - `settings.py`: Dataclass-based settings with env var overrides
  - `validated_settings.py`: Pydantic validation wrapper
  - `models.py`: Pydantic configuration models (AppConfig, LLMConfig, ReproConfig, etc.)

**`tests/`:**
- Purpose: Test suites organized by type
- Patterns:
  - Unit tests in `unit/` - fast, isolated, mocked
  - Integration tests in `integration/` - test layer boundaries
  - E2E tests in `e2e/` - full workflow testing
  - Uses `conftest.py` for shared fixtures

**`evals/`:**
- Purpose: Evaluation and benchmarking
- Contains:
  - `runners/`: Smoke tests and benchmark runners
  - `fixtures/`: Test data and mock responses
  - `memory/`: Memory evaluation tests
  - `scorers/`: Scoring logic for evaluations

## Key File Locations

**Entry Points:**
- `src/paperbot/api/main.py`: FastAPI server entry point (HTTP)
- `src/paperbot/infrastructure/queue/arq_worker.py`: Background job worker (async)
- `cli/src/index.tsx`: CLI entry point (terminal)
- `web/src/app/page.tsx`: Web dashboard entry (browser)
- `src/paperbot/mcp/serve.py`: MCP server entry point

**Configuration:**
- `config/config.yaml`: Application configuration (models, venues, thresholds)
- `.env`: Environment variables (secrets, API keys) - NOT committed
- `alembic.ini`: Database migration configuration
- `pyproject.toml`: Python project metadata and dependencies

**Core Logic:**
- `src/paperbot/domain/paper.py`: Paper domain model
- `src/paperbot/domain/scholar.py`: Scholar domain model
- `src/paperbot/core/di/container.py`: Dependency injection container
- `src/paperbot/core/pipeline/pipeline.py`: Pipeline orchestration
- `src/paperbot/core/workflow_coordinator.py`: Agent coordinator
- `src/paperbot/application/services/llm_service.py`: LLM service

**Testing:**
- `tests/conftest.py`: Shared test fixtures and configuration
- `tests/unit/`: Unit test files
- `tests/integration/`: Integration test files
- `tests/e2e/`: End-to-end test files

**Database:**
- `src/paperbot/infrastructure/stores/models.py`: SQLAlchemy ORM models
- `alembic/versions/`: Migration files

## Naming Conventions

**Files:**
- `agent.py`: Main agent implementation (e.g., `research/agent.py`)
- `*_store.py`: Repository/persistence layer (e.g., `paper_store.py`)
- `*_service.py`: Application service (e.g., `llm_service.py`)
- `*_port.py`: Interface/contract definition (e.g., `event_log_port.py`)
- `test_*.py`: Unit/integration tests
- `conftest.py`: Pytest configuration and shared fixtures
- `*_executor.py`: Execution engine (e.g., `docker_executor.py`)

**Directories:**
- `agents/{feature}/`: Feature-specific agent (e.g., `research/`, `code_analysis/`)
- `infrastructure/{category}/`: External integration category (e.g., `llm/`, `connectors/`)
- `api/routes/`: Endpoint handler per feature
- `application/workflows/`: Multi-stage pipeline per workflow
- `tests/{type}/test_*.py`: Test file per unit/integration/e2e

**Classes:**
- `{Feature}Agent`: Agent classes (ResearchAgent, CodeAnalysisAgent)
- `{Domain}Store`: Repository classes (PaperStore, ResearchStore)
- `{Service}Service`: Service classes (LLMService, PaperSearchService)
- `{Feature}Port`: Port/interface classes (EventLogPort, MemoryPort)
- `Base{Concept}`: Abstract base classes (BaseAgent, BaseExecutor)

**Functions:**
- `async def execute()`: Main execution method on agents/nodes
- `async def process()`: Main processing method (internal to BaseAgent)
- `def validate()`: Input validation method
- `def _execute()`: Protected override hook in agents

**Variables:**
- `container`: Dependency injection container instance
- `llm_client`: LLM client instance
- `result`: ExecutionResult objects
- `ctx`: Pipeline context
- `paper`: Paper domain model

## Where to Add New Code

**New Feature/Agent:**
- Primary code: `src/paperbot/agents/{feature}/agent.py`
- Prompts: `src/paperbot/agents/prompts/{feature}/`
- Tests: `tests/unit/agents/test_{feature}_agent.py`
- State (if needed): `src/paperbot/agents/state/{feature}_state.py`

**New API Endpoint:**
- Implementation: `src/paperbot/api/routes/{feature}.py`
- Router setup: Register in `src/paperbot/api/main.py`
- Tests: `tests/integration/test_{feature}_routes.py`

**New Service:**
- Implementation: `src/paperbot/application/services/{service}.py`
- Interface: `src/paperbot/application/ports/{service}_port.py` (if external dependencies)
- Tests: `tests/unit/test_{service}.py`

**New Domain Model:**
- Model: `src/paperbot/domain/{model}.py`
- No framework dependencies - pure dataclasses/value objects

**New Infrastructure Integration:**
- API client: `src/paperbot/infrastructure/api_clients/{service}_client.py`
- Connector: `src/paperbot/infrastructure/connectors/{source}.py`
- Store: `src/paperbot/infrastructure/stores/{entity}_store.py`
- Tests: `tests/integration/test_{integration}.py`

**New Workflow/Pipeline:**
- Implementation: `src/paperbot/application/workflows/{workflow}.py`
- Analysis nodes: `src/paperbot/application/workflows/analysis/{component}.py`
- Tests: `tests/integration/test_{workflow}.py`

**Shared Utilities:**
- Helpers: `src/paperbot/utils/{category}.py`
- Formatters: `src/paperbot/presentation/{formatter}.py`

**Tests:**
- Unit (no external deps): `tests/unit/test_{unit}.py`
- Integration (layers combined): `tests/integration/test_{integration}.py`
- E2E (full workflows): `tests/e2e/test_{feature}.py`

## Special Directories

**`src/paperbot/compat/`:**
- Purpose: Backward compatibility shims
- Generated: No
- Committed: Yes

**`src/paperbot/presentation/`:**
- Purpose: UI rendering and formatting utilities
- Generated: No
- Committed: Yes

**`alembic/versions/`:**
- Purpose: Database migration scripts
- Generated: Yes (via `alembic revision --autogenerate`)
- Committed: Yes

**`data/`:**
- Purpose: Runtime data and caches
- Generated: Yes (SQLite DB, cache files)
- Committed: No (.gitignored)

**`logs/`:**
- Purpose: Application log files
- Generated: Yes
- Committed: No

**`reports/`:**
- Purpose: Generated reports and artifacts
- Generated: Yes
- Committed: No

**`datasets/`:**
- Purpose: Evaluation datasets
- Generated: Partial (processed data)
- Committed: Some (metadata)

**`.planning/`:**
- Purpose: GSD phase planning documents
- Generated: Yes (via /gsd commands)
- Committed: Yes

**`e2b-template/`:**
- Purpose: E2B sandbox template for code execution
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-03-15*
