# Technology Stack

**Analysis Date:** 2026-03-15

## Languages

**Primary:**
- Python 3.8+ (target 3.10) - Backend server, agents, data processing
- TypeScript 5.7+ - Web dashboard and CLI
- JavaScript (ES2024+) - Next.js/React frontend

**Secondary:**
- YAML - Configuration files (`config/config.yaml`, `config/scholar_subscriptions.yaml`)
- XML/HTML - Document parsing and report generation
- SQL - Database schema and migrations

## Runtime

**Environment:**
- Python 3.10+ (recommended), supports 3.8-3.12 (matrix tested in CI)
- Node.js 18+ (for CLI and web components)

**Package Manager:**
- pip (Python) - `requirements.txt`, `requirements-ci.txt` for CI matrix
- npm (Node.js) - `package.json` in web/ and cli/ directories
- setuptools - Build backend via `pyproject.toml`

**Lockfiles:**
- Python: `requirements.txt` (pinned), `requirements-ci.txt` (lighter CI variant)
- Node.js: Missing (npm uses `package-lock.json` implicitly)

## Frameworks

**Core Backend:**
- FastAPI 0.115.0 - REST API server with SSE streaming
- Starlette 0.37.2-0.39.0 - Web framework (FastAPI foundation)
- Uvicorn 0.32.0+ - ASGI server with auto-reload support
- SQLAlchemy 2.0+ - ORM for database modeling
- Alembic 1.13.0+ - Database migrations

**LLM & AI:**
- OpenAI SDK 1.0.0+ - GPT models (gpt-4o, gpt-4o-mini)
- Anthropic SDK 0.3.0+ - Claude models (claude-3-5-sonnet, claude-3-opus, claude-3-haiku)
- Vercel AI SDK (web) - Unified LLM interface with streaming (`@ai-sdk/*` packages)

**Web Frontend:**
- Next.js 16.1.0 - React framework with App Router
- React 19.2.3 - UI library
- Tailwind CSS v4 - Utility-first CSS framework
- Radix UI 1.4.3 - Headless component library
- Zustand 5.0.9 - State management
- Framer Motion 12.23.26 - Animation library
- @xyflow/react 12.10.0 - DAG visualization
- Monaco Editor 4.7.0 - Code editor component
- XTerm 5.3.0 - Terminal emulation
- Next Auth 5.0.0-beta.30 - Authentication
- Vercel AI SDK 5.0.116 - LLM streaming client

**Terminal CLI:**
- Ink 5.0.1 - React framework for CLI
- Meow 13.2.0 - CLI argument parser
- Chalk 5.3.0 - Terminal color output
- tsx 4.19.2 - TypeScript execution engine

**Testing:**
- pytest 7.0.0+ - Python test runner
- pytest-asyncio 0.21.0+ - Async test support
- respx 0.21.0+ - HTTP mocking (httpx)
- aioresponses 0.7.6+ - Async HTTP mocking (aiohttp)
- pytest-mock 3.12.0+ - Mocking utilities
- vitest 2.1.4 - Node.js test runner
- Playwright 1.58.2+ - E2E browser automation

**Build & Dev:**
- Black 23.0.0+ - Python code formatter
- isort 5.12.0+ - Python import sorter
- Pyright (basic mode) - Python type checker
- ESLint 9+ - JavaScript/TypeScript linter
- TypeScript 5.7.2+ - Type checker

## Key Dependencies

**Critical:**
- pydantic[email] 2.0+ - Data validation and settings management
- requests 2.28.0+ - Synchronous HTTP client
- httpx[http2] 0.27.0-0.28.0 - Async HTTP client with HTTP/2 support (for IEEE/ACM downloads)
- aiohttp 3.8.0+ - Async HTTP client session management
- aiofiles 22.1.0+ - Async file I/O
- beautifulsoup4 4.11.0+ - HTML/XML parsing
- lxml 4.9.0+ - Fast XML processing

**Infrastructure:**
- redis 5.0.0+ - Redis client for ARQ task queue
- arq 0.25.0-0.26.0 - Redis-backed async job queue (DailyPaper cron, background tasks)
- psycopg[binary] 3.2.0+ - PostgreSQL adapter (Supabase support)
- cryptography 41.0.0+ - Encryption for credentials
- keyring 25.0.0+ - OS credential storage

**PDF & Document Processing:**
- pdfplumber 0.7.0+ - PDF text extraction
- PyPDF2 3.0.0+ - PDF manipulation
- weasyprint 62.3+ - HTML to PDF conversion
- markdown 3.4.0+ - Markdown parser
- jinja2 3.1.0+ - Template engine (report generation)
- json-repair 0.22.0+ - JSON cleanup for LLM outputs

**Security & Auth:**
- python-jose[cryptography] 3.3.0+ - JWT token management
- bcrypt 4.0.0+ - Password hashing
- email-validator 2.0.0+ - Email validation

**Data & AI:**
- rapidfuzz 3.0.0+ - Fuzzy string matching for paper deduplication
- numpy 1.24.0+ - Numerical computing (repro pipeline)
- pandas 1.5.0+ - Data analysis and manipulation
- sqlite-vec 0.1.6+ - Vector search in SQLite (optional, graceful fallback)

**Sandbox Execution:**
- docker 7.0.0+ - Docker API client (local code execution)
- e2b-code-interpreter 1.0.0+ - E2B cloud sandbox (secure execution)

**Notifications & Feeds:**
- apprise 1.9.0+ - Multi-channel push notifications (email, Slack, DingTalk)
- feedgen 1.0.0+ - RSS/Atom feed generation
- resend 0.7.0+ - Email delivery service client

**MCP (Model Context Protocol):**
- mcp[fastmcp] 1.8.0-2.0.0 - MCP server (Python 3.10+ only)
- @modelcontextprotocol/sdk 1.25.1 - MCP client (web/CLI)

**Utilities:**
- python-dotenv 0.19.0+ - Environment variable loading
- PyYAML 6.0+ - YAML parsing
- GitPython 3.1.0+ - Git repository access
- loguru 0.7.0+ - Structured logging
- watchdog 4.0.0+ - File system monitoring

## Configuration

**Environment:**
- Loaded via `python-dotenv` from `.env` file (see `env.example`)
- All required keys documented in `env.example`
- Sensitive values: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `E2B_API_KEY`, database credentials

**Key Configuration Files:**
- `pyproject.toml` - Python project metadata, dependencies, tool settings
- `config/config.yaml` - Main app configuration (venues, download settings, analysis depth)
- `config/settings.py` - Pydantic settings loader
- `config/models.py` - Pydantic config models
- `env.example` - Environment variable template
- `.env` - Runtime secrets (not committed)

**Tool Settings:**
- `pyproject.toml` sections:
  - `[tool.black]` - line-length 100, target-version py310
  - `[tool.isort]` - Black profile, line-length 100
  - `[tool.pyright]` - basic mode, Python 3.10
  - `[tool.pytest.ini_options]` - asyncio_mode strict (critical for async tests)

## Platform Requirements

**Development:**
- Python 3.10+ with venv
- Node.js 18+ with npm
- Git
- PostgreSQL or SQLite (default is local SQLite)
- Redis (for ARQ task queue, optional for dev but required for cron features)
- Docker (optional, for code execution sandbox)

**Production:**
- Python 3.10+ runtime
- Node.js 18+ (for web dashboard)
- PostgreSQL database (Supabase pooler supported)
- Redis instance (for ARQ queue)
- Docker daemon (if using docker executor for code runs)
- E2B API key (if using E2B cloud sandbox)

**Deployment Targets:**
- Linux/macOS development environments
- Docker containerization capable
- Cloud: Vercel (Next.js web), AWS/GCP/Azure (FastAPI backend), Supabase (PostgreSQL)

---

*Stack analysis: 2026-03-15*
