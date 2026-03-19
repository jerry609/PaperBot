# Architecture

**Analysis Date:** 2026-03-15

## Pattern Overview

**Overall:** Domain-Driven Design (DDD) with layered architecture and multi-agent orchestration

**Key Characteristics:**
- Domain models (`domain/`) define core entities independent of frameworks
- Application layer (`application/`) implements use cases via services and workflows
- Infrastructure layer (`infrastructure/`) handles persistence, external APIs, and queues
- Core abstractions (`core/`) provide execution models, DI, and pipeline orchestration
- Multi-agent system with specialized agents for different analysis types
- SSE (Server-Sent Events) streaming for real-time API responses
- Async/await patterns throughout for non-blocking I/O

## Layers

**Domain Layer:**
- Purpose: Define core business concepts and value objects independent of technology
- Location: `src/paperbot/domain/`
- Contains: `Paper` (PaperMeta, CodeMeta), `Scholar`, `Track`, `Harvest`, `Enrichment`, `Feedback`, `Identity`
- Depends on: None (isolated from frameworks)
- Used by: Application and Infrastructure layers

**Application Layer:**
- Purpose: Implement use cases, orchestrate domain logic, and coordinate agents
- Location: `src/paperbot/application/`
- Contains:
  - `services/`: Business logic services (LLMService, PaperSearchService, AnchorService)
  - `workflows/`: Multi-stage pipelines (DailyPaper, Scholar Pipeline, Harvest Pipeline)
  - `ports/`: Interface definitions for infrastructure dependencies (EventLogPort, PaperSearchPort, MemoryPort)
  - `collaboration/`: Cross-agent message schemas and coordination models
  - `registries/`: Source and provider registries
  - `prompts/`: LLM prompt templates
- Depends on: Domain, Ports (abstractions only)
- Used by: API routes, Agents, Workflows

**Core Layer:**
- Purpose: Provide foundational abstractions and infrastructure patterns
- Location: `src/paperbot/core/`
- Contains:
  - `abstractions/`: Executable interface, ExecutionResult, ensure_execution_result
  - `di/`: Lightweight dependency injection container with singleton support
  - `pipeline/`: Declarative pipeline framework (Pipeline, PipelineStage, PipelineResult)
  - `collaboration/`: ScoreShareBus (cross-stage evaluation), FailFastEvaluator (early termination)
  - `errors/`: Custom exception hierarchy
  - `workflow_coordinator.py`: ScholarWorkflowCoordinator orchestrates agent stages
  - `state.py`: Workflow state management
  - `report_engine/`: Report generation with Jinja2 templates
- Depends on: Domain
- Used by: Agents, Workflows, Services

**Agent Layer:**
- Purpose: Specialized autonomous agents for different analysis tasks
- Location: `src/paperbot/agents/`
- Contains:
  - `base.py`: BaseAgent with Template Method pattern (validate → execute → post_process)
  - `research/`: ResearchAgent - research and novelty analysis
  - `code_analysis/`: CodeAnalysisAgent - code quality and implementation assessment
  - `quality/`: QualityAgent - paper quality evaluation
  - `review/`: ReviewerAgent - deep peer review simulation
  - `verification/`: VerificationAgent - claim verification
  - `scholar_tracking/`: Scholar monitoring and influence analysis
  - `conference/`: Conference research and trend analysis
  - `documentation/`: Documentation generation
  - `mixins/`: JSONParserMixin for structured output extraction
  - `prompts/`: Agent-specific prompt templates
  - `state/`: Agent state management
- Depends on: Core, Domain, Application (services)
- Used by: Workflows, API routes

**Infrastructure Layer:**
- Purpose: Implement external integrations and persistence
- Location: `src/paperbot/infrastructure/`
- Contains:
  - `llm/`: LLM provider implementations (OpenAI, Anthropic, Ollama)
  - `stores/`: SQLAlchemy repositories (PaperStore, ResearchStore, MemoryStore, etc.)
  - `connectors/`: Data source connectors (ArXiv, Reddit, HuggingFace, X, etc.)
  - `crawling/`: HTTP downloader and PDF/HTML parser layer
  - `adapters/`: Search adapters (ArXiv, Semantic Scholar, OpenAlex, etc.)
  - `api_clients/`: Third-party API clients (Semantic Scholar, GitHub)
  - `harvesters/`: Bulk paper collection (ArXiv, OpenAlex, Semantic Scholar)
  - `event_log/`: Event persistence (SQLAlchemy, memory, composite, EventBus)
  - `queue/`: ARQ worker for async jobs (DailyPaper cron, background tasks)
  - `logging/`: Structured logging and monitoring
  - `storage/`: Local/cloud file storage
  - `services/`: Infrastructure services (email, push notifications)
  - `extractors/`: Document extraction (MinEru for figures)
  - `push/`: Push notification adapters
  - `obsidian/`: Obsidian vault export
  - `swarm/`: OpenAI Swarm agent integration
- Depends on: Domain, Application ports
- Used by: Services, Workflows, Agents

**Paper2Code (Repro) Pipeline:**
- Purpose: Multi-stage code generation and execution from papers
- Location: `src/paperbot/repro/`
- Contains:
  - `orchestrator.py`: Manages multi-stage pipeline (Planning → Blueprint → Environment → Generation → Verification)
  - `repro_agent.py`: Main agent coordinating code generation
  - `agents/`: Planning, Coding, Debugging, Verification agents
  - `nodes/`: Pipeline execution nodes
  - `memory/`: CodeMemory (cross-file context tracking), SymbolIndex (AST-based code indexing)
  - `rag/`: CodeRAG for pattern retrieval from knowledge base
  - `docker_executor.py`, `e2b_executor.py`: Execution backends for sandboxed code
  - `verification_runtime.py`: Runtime verification and testing
- Depends on: Core, Domain, Infrastructure
- Used by: API routes, Workflows

**API Layer:**
- Purpose: HTTP endpoints with SSE streaming for client interfaces
- Location: `src/paperbot/api/`
- Contains:
  - `main.py`: FastAPI app setup, router registration, middleware
  - `routes/`: Endpoint handlers organized by feature (track, analyze, gen_code, research, etc.)
  - `streaming.py`: SSE utilities for event streaming
  - `middleware/`: Authentication, CORS, rate limiting
  - `error_handling/`: Centralized error handling
- Depends on: All layers (facade pattern)
- Used by: CLI, Web, external clients

**Context Engine:**
- Purpose: Research context management and track routing
- Location: `src/paperbot/context_engine/`
- Provides: Context-aware paper filtering and track assignment

**Memory Module:**
- Purpose: Persistent memory for agent state and learning
- Location: `src/paperbot/memory/`
- Contains: Memory storage, parsers, extractors, evaluators

**MCP Server:**
- Purpose: Model Context Protocol implementation for tool/resource exposure
- Location: `src/paperbot/mcp/`
- Contains:
  - `server.py`: MCP server definition
  - `serve.py`: Server startup
  - `tools/`: Tool implementations
  - `resources/`: Resource definitions

## Data Flow

**Paper Analysis Pipeline:**

1. **Paper Ingestion** (API: POST /api/analyze or ARQ worker)
   - Paper input validated and enriched with metadata
   - Retrieved from sources or provided directly
   - Stored in `infrastructure/stores/paper_store.py`

2. **Multi-Agent Analysis** (ScholarWorkflowCoordinator)
   - ResearchAgent: Analyzes research contribution and novelty
   - CodeAnalysisAgent: Evaluates code quality (if code present)
   - QualityAgent: Assesses paper quality across multiple dimensions
   - ReviewerAgent: Simulates peer review (optional)
   - VerificationAgent: Verifies claims (optional)

3. **Evaluation & Scoring** (ScoreShareBus)
   - Agents compute scores and share via ScoreShareBus
   - FailFastEvaluator may terminate pipeline early if quality too low
   - Influence calculation done independently or conditionally

4. **Report Generation** (ReportEngine)
   - Scores aggregated and formatted via Jinja2 templates
   - Generated report stored and returned

**Scholar Tracking:**

1. **Scholar Registration** (API: POST /api/track or direct)
   - Scholar added to tracking database
   - Research interests configured

2. **Daily Paper Collection** (ARQ scheduled task)
   - Queries execute against registered sources
   - Papers deduplicated and ranked
   - Report generated and sent to subscriber

3. **Enrichment Pipeline** (services/enrichment_pipeline.py)
   - Papers enriched with author metadata
   - Code detection and analysis
   - Citation trends tracked

**Paper2Code Generation:**

1. **Planning Stage** (repro/agents/)
   - Extract key techniques from paper
   - Understand data requirements
   - Design implementation plan

2. **Environment Stage**
   - Setup Docker or E2B sandbox
   - Install dependencies
   - Clone necessary repositories

3. **Generation Stage**
   - Generate code based on plan
   - CodeMemory tracks cross-file context
   - CodeRAG retrieves similar patterns

4. **Verification Stage**
   - Run generated code
   - Compare outputs with paper results
   - Flag issues for debugging

**State Management:**
- Pipeline context (`PipelineContext`) flows through stages
- AgentCoordinator broadcasts tasks to registered agents
- EventLog captures all pipeline events for audit/replay
- Memory module tracks agent decisions and learning

## Key Abstractions

**Executable Interface:**
- Purpose: Standardized contract for all executable components (agents, nodes)
- Examples: `src/paperbot/agents/base.py`, `src/paperbot/repro/nodes/`
- Pattern: Two methods - `validate()` and `execute()` returning `ExecutionResult`

**ExecutionResult:**
- Purpose: Uniform response wrapper with success/data/error/metadata
- Location: `src/paperbot/core/abstractions/executable.py`
- Pattern: Used consistently across agents, services, nodes

**Pipeline:**
- Purpose: Declarative multi-stage orchestration
- Location: `src/paperbot/core/pipeline/pipeline.py`
- Pattern: Stages added with `.add_stage()`, run sequentially with context flow-through

**AgentCoordinator:**
- Purpose: Multi-agent task broadcast and result collection
- Location: `src/paperbot/core/collaboration/`
- Pattern: Agents register interest, coordinator broadcasts, collects results

**Container (DI):**
- Purpose: Lightweight dependency injection without framework overhead
- Location: `src/paperbot/core/di/container.py`
- Pattern: Singleton registration and lazy resolution

**Port Pattern:**
- Purpose: Application layer depends on abstractions, infrastructure provides implementations
- Location: `src/paperbot/application/ports/`
- Examples: EventLogPort → multiple implementations (SQLAlchemy, Memory, Composite, EventBus)

## Entry Points

**FastAPI Server:**
- Location: `src/paperbot/api/main.py`
- Triggers: `python -m uvicorn src.paperbot.api.main:app --reload`
- Responsibilities:
  - Registers all route handlers
  - Installs middleware (auth, CORS, rate limiting)
  - Configures error handling
  - Sets up event logging

**ARQ Worker (Background Jobs):**
- Location: `src/paperbot/infrastructure/queue/arq_worker.py`
- Triggers: `arq paperbot.infrastructure.queue.arq_worker.WorkerSettings`
- Responsibilities:
  - Processes DailyPaper scheduled tasks
  - Handles async background jobs
  - Uses Redis for job queue

**CLI (Ink/React):**
- Location: `cli/src/index.tsx`
- Triggers: `npm run start` or installed global command
- Responsibilities:
  - Terminal-based UI for running workflows
  - Connects to API via HTTP
  - Displays SSE streaming events in real-time

**Web Dashboard (Next.js):**
- Location: `web/src/app/` (App Router)
- Triggers: `npm run dev` (development)
- Responsibilities:
  - Browser-based UI for analysis and tracking
  - Pages: dashboard, papers, research, scholars, settings, studio, workflows
  - Connects to API via Vercel AI SDK and fetch

**MCP Server:**
- Location: `src/paperbot/mcp/serve.py`
- Triggers: `paperbot mcp serve` or configured in Claude settings
- Responsibilities:
  - Exposes tools and resources via Model Context Protocol
  - Allows Claude/AI models to invoke PaperBot capabilities

**Standalone Scripts:**
- Location: `evals/runners/`, `main.py`
- Triggers: Direct Python execution
- Responsibilities:
  - Smoke tests for pipelines
  - Memory evaluations
  - Batch processing

## Error Handling

**Strategy:** Centralized error handling with typed exceptions

**Patterns:**
- Custom exception hierarchy in `src/paperbot/core/errors/`
- Agents wrap exceptions in `ExecutionResult(success=False, error=str(e))`
- API routes use `@app.exception_handler()` decorators
- EventLog captures all errors for audit trail
- FailFastEvaluator stops pipelines on critical failures

**Logging:**
- Structured logging via Python logging module
- Infrastructure implements EventLog port for pluggable storage
- Separate logs for API, workers, harvests in `logs/` directory

## Cross-Cutting Concerns

**Logging:**
- Central EventLog port with multiple implementations (SQLAlchemy, Memory, Composite)
- All pipeline events captured with timestamps and metadata

**Validation:**
- Agent input validation via `_validate_input()` hook
- Schema validation in API routes via Pydantic models
- Domain model invariants checked in constructors

**Authentication:**
- API auth via `install_api_auth()` middleware
- Environment variable based for API keys (LLM, Semantic Scholar, GitHub)
- User context available in web/CLI sessions

**Persistence:**
- SQLAlchemy ORM layer in `infrastructure/stores/`
- Migrations managed via Alembic
- Database URL configured via `PAPERBOT_DB_URL` environment variable

**Configuration:**
- Application-wide config in `config/config.yaml`
- Settings via `config/settings.py` with environment overrides
- Pydantic validation in `config/validated_settings.py`

---

*Architecture analysis: 2026-03-15*
