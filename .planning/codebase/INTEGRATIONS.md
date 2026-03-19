# External Integrations

**Analysis Date:** 2026-03-15

## APIs & External Services

**LLM Providers:**
- OpenAI (gpt-4o, gpt-4o-mini, embeddings)
  - SDK: `openai>=1.0.0`
  - Auth: `OPENAI_API_KEY`
  - Custom endpoint support via `OPENAI_BASE_URL`
  - Implementation: `src/paperbot/infrastructure/llm/providers/openai_provider.py`

- Anthropic (claude-3-5-sonnet, claude-3-opus, claude-3-haiku)
  - SDK: `anthropic>=0.3.0`
  - Auth: `ANTHROPIC_API_KEY`
  - Implementation: `src/paperbot/infrastructure/llm/providers/anthropic_provider.py`

- Ollama (self-hosted models)
  - Implementation: `src/paperbot/infrastructure/llm/providers/ollama_provider.py`
  - Local endpoint support via `OLLAMA_BASE_URL`

**Embedding Provider:**
- Dedicated embedding endpoint (optional)
  - Auth: `PAPERBOT_EMBEDDING_API_KEY`
  - Base URL: `PAPERBOT_EMBEDDING_BASE_URL`
  - Model: `PAPERBOT_EMBEDDING_MODEL` (default: text-embedding-3-small)
  - Provider chain: `PAPERBOT_EMBEDDING_PROVIDER_CHAIN`

**Academic Paper Sources:**
- arXiv (via public API)
  - API: `https://export.arxiv.org/api/query`
  - Client: `src/paperbot/infrastructure/connectors/arxiv_connector.py`
  - No auth required, rate-limited

- OpenAlex (via API)
  - Client: `src/paperbot/infrastructure/connectors/openalex_connector.py`
  - Optional auth: `SEMANTIC_SCHOLAR_API_KEY`

- Semantic Scholar (via API + citation graph)
  - Client: `src/paperbot/infrastructure/api_clients/semantic_scholar.py`
  - Auth: `SEMANTIC_SCHOLAR_API_KEY`
  - Citation graph builder: `src/paperbot/infrastructure/api_clients/citation_graph.py`

- PaperScool (via API)
  - Client: `src/paperbot/infrastructure/connectors/paperscool_connector.py`

- Hugging Face Papers (daily papers feed)
  - Client: `src/paperbot/infrastructure/connectors/hf_daily_papers_connector.py`

- OpenReview (reviews & submissions)
  - Client: `src/paperbot/infrastructure/api_clients/openreview_client.py`

- Zotero (library sync)
  - Client: `src/paperbot/infrastructure/connectors/zotero_connector.py`

**Social & Discussion:**
- Reddit (paper discussions)
  - Client: `src/paperbot/infrastructure/connectors/reddit_connector.py`

- X (Twitter) - research trends
  - Client: `src/paperbot/infrastructure/api_clients/x_client.py`
  - Auth: `PAPERBOT_X_BEARER_TOKEN`

**Search Adapters (multi-backend):**
- arXiv search adapter: `src/paperbot/infrastructure/adapters/arxiv_search_adapter.py`
- OpenAlex adapter: `src/paperbot/infrastructure/adapters/openalex_adapter.py`
- Semantic Scholar adapter: `src/paperbot/infrastructure/adapters/s2_search_adapter.py`
- HuggingFace daily adapter: `src/paperbot/infrastructure/adapters/hf_daily_adapter.py`
- PaperScool adapter: `src/paperbot/infrastructure/adapters/paperscool_adapter.py`

**Code Repository Access:**
- GitHub
  - Client: `src/paperbot/infrastructure/api_clients/github_client.py`
  - Auth: `GITHUB_TOKEN`
  - Uses: `requests.Session()` for HTTP requests

**PDF Processing:**
- MinerU Cloud (document parsing)
  - Auth: `MINERU_API_KEY`
  - Base URL: `MINERU_API_BASE_URL` (default: https://mineru.net/api/v4)
  - Model: `MINERU_MODEL_VERSION` (vlm or pipeline)
  - Async task polling with configurable timeout: `MINERU_MAX_WAIT_SECONDS`

**University Libraries (gated access):**
- ACM Digital Library (DL.ACM.org)
  - Access: `ACM_LIBRARY_URL` (institutional login URL)

- IEEE Xplore (IEEE.org papers)
  - HTTP/2 support via `httpx[http2]` for reliable downloads

## Data Storage

**Databases:**
- PostgreSQL (recommended for production)
  - Connection: `PAPERBOT_DB_URL` (SQLAlchemy connection string)
  - Supabase pooler compatible: `postgresql+psycopg://...@pooler.supabase.com:6543`
  - ORM: SQLAlchemy 2.0+
  - Client: `psycopg[binary]>=3.2.0`
  - Migrations: Alembic (run `alembic upgrade head`)

- SQLite (default for local development)
  - Default: `sqlite:///data/paperbot.db`
  - Schema models: `src/paperbot/infrastructure/stores/models.py`
  - Stores:
    - `paper_store.py` - Papers, metadata
    - `research_store.py` - Research tracks, context
    - `memory_store.py` - Conversation history, embeddings
    - `author_store.py` - Author tracking
    - `user_store.py` - User accounts, preferences
    - `pipeline_session_store.py` - Workflow execution sessions

**File Storage:**
- Local filesystem
  - Papers directory: configured in `config/config.yaml` (default: `./papers`)
  - Reports: `PAPERBOT_RE_OUTPUT_DIR` (default: output/reports)
  - Runbook workspace: `PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES` (allowed directories for Studio file access)

**Caching:**
- In-memory (development)
  - Implementation: `src/paperbot/infrastructure/event_log/memory_event_log.py`

- Redis (recommended for production)
  - Connection: `PAPERBOT_REDIS_HOST`, `PAPERBOT_REDIS_PORT`
  - ARQ worker uses Redis for task persistence
  - Key: DailyPaper cron job queue

**Vector Search (optional):**
- SQLite-vec (`sqlite-vec>=0.1.6`)
  - Graceful fallback to FTS5 if unavailable
  - Used by: `src/paperbot/infrastructure/stores/document_index_store.py`

## Authentication & Identity

**User Authentication:**
- JWT tokens (RFC 7519)
  - Implementation: `src/paperbot/api/auth/jwt.py`
  - Token creation: `create_access_token(user_id)`
  - Verification: `get_current_user` dependency in FastAPI routes

- Password hashing: bcrypt 4.0.0+
  - Implementation: `src/paperbot/api/auth/password.py`
  - Storage: User model in `user_store.py`

- Email validation
  - Pydantic EmailStr with `email-validator>=2.0.0`

- OAuth2 (NextAuth.js v5 beta)
  - Web/CLI: `next-auth 5.0.0-beta.30`
  - Providers: OpenAI, Anthropic, GitHub (configured in web layout)

**API Credentials:**
- External API keys stored in `.env` (never committed)
- Keyring integration: `keyring>=25.0.0` for OS credential storage
- Keys managed per environment via env vars

## Monitoring & Observability

**Error Tracking:**
- Structured logging via `loguru>=0.7.0`
  - Implementation: `src/paperbot/api/error_handling.py`

**Event Logging:**
- Multiple backends:
  - SQLAlchemy: `src/paperbot/infrastructure/event_log/sqlalchemy_event_log.py` - persistent event storage
  - EventBus: `src/paperbot/infrastructure/event_log/event_bus_event_log.py` - in-process event bus
  - Memory: `src/paperbot/infrastructure/event_log/memory_event_log.py` - test/dev only
  - Composite: `src/paperbot/infrastructure/event_log/composite_event_log.py` - multi-backend

- Event models: `src/paperbot/infrastructure/stores/models.py`
  - `AgentRunModel` - execution run metadata
  - `AgentEventModel` - event trace with trace_id, span_id
  - `ExecutionLogModel` - stdout/stderr logs
  - `ResourceMetricModel` - CPU, memory, I/O usage

**Metrics:**
- Workflow metrics: `workflow_metric_store.py`
- LLM usage tracking: `llm_usage_store.py` (tokens, cost)
- Memory evaluation: `evals/memory/` - retrieval hit rate, deletion compliance

## CI/CD & Deployment

**Hosting:**
- Web dashboard: Vercel (Next.js native, `npm run build && npm start`)
- API backend: AWS/GCP/Azure (Docker-containerized FastAPI)
- Database: Supabase PostgreSQL (or self-managed)
- MCP Server: Python FastMCP server (standalone or embedded)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
- Matrix: Python 3.10, 3.11, 3.12
- Offline test gates: unit, integration, E2E tests
- Uses `requirements-ci.txt` (lighter deps, no weasyprint/heavy packages)
- Eval smoke tests: scholar pipeline, track pipeline, eventlog replay
- Memory evals: deletion compliance, retrieval hit rate

**Task Queue:**
- ARQ (async job queue)
  - Redis backend: `redis>=5.0.0`
  - Worker: `src/paperbot/infrastructure/queue/arq_worker.py`
  - DailyPaper cron: scheduled via ARQ with `PAPERBOT_DAILYPAPER_ENABLED=true`
  - Cron time: `PAPERBOT_DAILYPAPER_CRON_HOUR`, `PAPERBOT_DAILYPAPER_CRON_MINUTE`

## Environment Configuration

**Required env vars for core features:**
```
OPENAI_API_KEY              # or ANTHROPIC_API_KEY for Claude
PAPERBOT_DB_URL            # database connection (default: sqlite:///data/paperbot.db)
PAPERBOT_REDIS_HOST        # Redis for ARQ (optional, default: localhost)
PAPERBOT_REDIS_PORT        # Redis port (optional, default: 6379)
```

**Optional integrations:**
```
SEMANTIC_SCHOLAR_API_KEY   # Paper metadata + citation graph
GITHUB_TOKEN               # Repository code access
PAPERBOT_X_BEARER_TOKEN    # Twitter/X trend tracking
MINERU_API_KEY             # PDF document parsing
E2B_API_KEY                # Cloud sandbox code execution
ACM_LIBRARY_URL            # ACM DL gated access
```

**Notifications:**
```
PAPERBOT_NOTIFY_ENABLED=true
PAPERBOT_NOTIFY_CHANNELS=email,slack,dingtalk
PAPERBOT_NOTIFY_SMTP_HOST=...       # Email via SMTP
PAPERBOT_NOTIFY_SLACK_WEBHOOK_URL=... # Slack integration
PAPERBOT_NOTIFY_DINGTALK_WEBHOOK_URL=... # DingTalk robot
```

**DailyPaper cron:**
```
PAPERBOT_DAILYPAPER_ENABLED=true
PAPERBOT_DAILYPAPER_CRON_HOUR=8
PAPERBOT_DAILYPAPER_QUERIES=query1,query2
PAPERBOT_DAILYPAPER_SOURCES=arxiv,papers_cool
PAPERBOT_DAILYPAPER_ENABLE_LLM=true
```

## Webhooks & Callbacks

**Incoming Webhooks:**
- None detected in codebase

**Outgoing Integrations:**
- Slack notifications (webhook-based, requires `PAPERBOT_NOTIFY_SLACK_WEBHOOK_URL`)
- DingTalk robot notifications (webhook-based, requires `PAPERBOT_NOTIFY_DINGTALK_WEBHOOK_URL`)
- Email notifications (SMTP, requires `PAPERBOT_NOTIFY_SMTP_*` env vars)
- Apprise multi-channel (`apprise>=1.9.0` for unified push delivery)

**SSE Streaming Endpoints (Server-Sent Events):**
- `POST /api/analyze` - Paper analysis stream
- `POST /api/gen-code` - Code generation stream
- `POST /api/track` - Scholar tracking stream
- `POST /api/review` - Deep review stream
- `POST /api/research/*` - Personalized research context stream
- Implementation: `src/paperbot/api/streaming.py` with StandardEvent enum

## MCP Server Integration

**Model Context Protocol:**
- FastMCP server: `src/paperbot/mcp/server.py`
- Tools exposed via MCP:
  - `paper_search` - Find papers across sources
  - `paper_judge` - Quality assessment
  - `paper_summarize` - Generate summaries
  - `relevance` - Relevance scoring
  - `analyze_trends` - Trend analysis
  - `check_scholar` - Scholar tracking
  - `get_research_context` - Personalized context
  - `save_to_memory` - Long-term memory
  - `export_to_obsidian` - Knowledge export
- Python 3.10+ required (mcp constraint)

---

*Integration audit: 2026-03-15*
