# Codebase Concerns

**Analysis Date:** 2026-03-15

## Tech Debt

### N+1 Query Patterns in Data Fetch Loops

**Issue:** Multiple data stores execute sequential queries inside loops instead of batch operations

**Files:**
- `src/paperbot/infrastructure/stores/author_store.py:152` - Loop with per-item author lookups
- `src/paperbot/infrastructure/stores/research_store.py:2267-2289` - Sequential paper URL lookups with `scalar_one_or_none()`
- `src/paperbot/application/services/anchor_service.py:183` - Batch-fetch of author_papers and feedback_rows
- `src/paperbot/application/services/author_backfill_service.py:75` - Sequential paper_authors processing

**Impact:** Severe performance degradation on large datasets. List 100+ items → 100+ queries instead of 1-2 batched queries.

**Fix approach:**
- Refactor to collect IDs first, then fetch all in single batch query using `.in_()` clauses
- Use SQLAlchemy's bulk operations for inserts/updates
- Example: Instead of `for author in authors: session.query(Author).filter_by(name=author.name).scalar_one_or_none()`, do `session.query(Author).filter(Author.name.in_([a.name for a in authors])).all()`

### `scalar_one_or_none()` Can Raise MultipleResultsFound

**Issue:** In `src/paperbot/infrastructure/stores/research_store.py:2277`, URL/title lookups use `scalar_one_or_none()` which throws `MultipleResultsFound` if duplicates exist

**Files:** `src/paperbot/infrastructure/stores/research_store.py:2267-2289`

**Impact:** Uncaught exception crashes paper deduplication. Should gracefully handle duplicates.

**Fix approach:**
- Replace `scalar_one_or_none()` with `.first()` (returns None if 0 or many matches)
- Or add explicit try-catch for `MultipleResultsFound` and log a warning before taking first result

### Sequential GitHub API Calls Without Rate Limiting

**Issue:** In `src/paperbot/api/routes/paperscool.py:1172-1189`, GitHub repo metadata is fetched one-at-a-time inside a loop

**Files:** `src/paperbot/api/routes/paperscool.py:1172-1189`

**Impact:**
- 60 req/hr unauthenticated, 5000/hr authenticated rate limits easily exceeded on modest paper lists
- Multi-minute delays for 100 items (1-2 sec per API call)
- Blocks entire enrichment response

**Fix approach:**
- Use `concurrent.futures.ThreadPoolExecutor` or `asyncio` with bounded concurrency (8-16 workers max)
- Batch requests if GitHub API supports it
- Add rate-limit aware retry with exponential backoff

### Incomplete Implementation TODOs in Code Generation Pipeline

**Issue:** Multiple nodes in repro pipeline have stubbed implementations

**Files:**
- `src/paperbot/repro/nodes/generation_node.py:399, 402, 405, 618, 628, 654, 663, 952` - Training loop, evaluation, inference, model architecture all TODO
- `src/paperbot/workflows/nodes/report_generation_node.py:141` - Topic extraction not implemented

**Impact:** Code generation features partially non-functional. User-facing operations may fail with "not implemented" stubs.

**Fix approach:**
- Complete TODO implementations with proper error states
- Add feature flags to disable incomplete stages
- Document which features are partial/experimental in API responses

### Parallel Execution Unimplemented in Repro Orchestrator

**Issue:** In `src/paperbot/repro/orchestrator.py:422-424`, `run_parallel()` is a stub that delegates to sequential `run()`

**Files:** `src/paperbot/repro/orchestrator.py:422-424`

**Impact:** Code reproduction pipeline cannot parallelize independent stages (blueprint + analysis could run together)

**Fix approach:**
- Implement true concurrent task launching with `asyncio.gather()`
- Ensure dependency tracking (planning must finish before blueprint/analysis)
- Add stage-level error isolation (one failure doesn't halt independent stages)

---

## Known Bugs

### SQLAlchemy Exception Handling Swallows Context

**Issue:** Broad `except Exception` blocks throughout codebase (723 instances) mask root causes

**Files:** Widespread across:
- `src/paperbot/repro/nodes/*.py` - Multiple "except Exception as e" catches
- `src/paperbot/infrastructure/stores/research_store.py:2297-2298` - JSON parse fallback silently ignores malformed data
- `src/paperbot/api/routes/paperscool.py:1222-1224` - Async enrichment swallows all errors

**Impact:** Debugging is difficult. Root causes buried. Silent failures in async operations.

**Fix approach:**
- Use specific exception types: `JSONDecodeError`, `IntegrityError`, `SQLAlchemyError`, etc.
- Log full tracebacks with context before re-raising or returning error state
- For async fire-and-forget hooks, at minimum log to error file

### Unhandled Client Connection Loss in SSE Streams

**Issue:** In `src/paperbot/api/streaming.py`, client disconnection during stream is not explicitly handled

**Files:** `src/paperbot/api/streaming.py:158-240`

**Impact:** Server continues processing after client drops connection, wasting resources. No cleanup of pending tasks.

**Current pattern:** Task cancellation is attempted in `finally` block, but if generator yields large data, memory may leak

**Fix approach:**
- Add explicit client disconnect detection (FastAPI provides `request.is_disconnected()`)
- Periodically check disconnect status in long-running generators
- Cancel pending tasks immediately on disconnect

---

## Security Considerations

### GDPR Compliance Gap: Email Stored as Plaintext

**Issue:** In `src/paperbot/infrastructure/stores/models.py:715-717`, `NewsletterSubscriberModel.email` stored as plaintext String column

**Files:** `src/paperbot/infrastructure/stores/models.py:715-717` and associated code

**Risk:**
- Email addresses leaked if database is compromised
- No encryption at rest
- No GDPR "right to erasure" implementation (only sets `status='unsubscribed'`, row never deleted)

**Current state:**
- Email indexed for lookups (prevents selective encryption)
- No database-level encryption configured

**Recommendations:**
1. Add database-level transparent encryption (PostgreSQL `pgcrypto`, or application-layer encryption)
2. Implement hard-delete method for unsubscribe + 30-day retention policy
3. Hash email for deduplication checks instead of plaintext comparison
4. Document GDPR/CCPA compliance in API docs

### Missing User-Based Access Control on Multi-User Endpoints

**Issue:** In `src/paperbot/api/routes/harvest.py:155-158`, `/harvest/runs` lists ALL harvest runs without user_id filtering

**Files:** `src/paperbot/api/routes/harvest.py:155-158`

**Risk:** Multi-user production deployment allows users to see each other's harvest history

**Current state:** Marked as "Intentional for MVP single-user setup" but dangerous if deployed multi-tenant

**Recommendations:**
- Add `user_id` filtering to all list endpoints that should be per-user
- Audit other endpoints: `/research/*`, `/track/*` for similar gaps
- Add middleware to enforce user_id context validation

### Path Traversal Risk in Runbook File Access

**Issue:** In `src/paperbot/infrastructure/swarm/codex_dispatcher.py:1305`, unsafe generated paths are logged as "skipped" but validation is minimal

**Files:** `src/paperbot/infrastructure/swarm/codex_dispatcher.py:1305`

**Risk:** Code generation agents may attempt to write files outside allowed directories

**Current mitigation:** `PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES` restricts base paths, but validation logic unclear

**Recommendations:**
- Implement strict whitelist-based path validation before all file I/O
- Use `Path.resolve()` to eliminate `..` and symlink attacks
- Reject paths with null bytes, unusual encodings
- Add test cases for traversal attempts

---

## Performance Bottlenecks

### Large File Sizes Indicate Complexity

**Issue:** Multiple modules exceed 1000 LOC, reducing maintainability and increasing bug surface

**Files (largest):**
- `src/paperbot/api/routes/research.py` (4128 lines) - Single mega-endpoint file
- `src/paperbot/api/routes/agent_board.py` (3678 lines) - Multi-stage orchestration
- `src/paperbot/infrastructure/stores/research_store.py` (2310 lines) - Monolithic data layer
- `src/paperbot/infrastructure/swarm/codex_dispatcher.py` (1611 lines) - Code execution dispatch
- `src/paperbot/api/routes/paperscool.py` (1454 lines) - Paper enrichment pipeline

**Impact:** Hard to test individual functions. High cognitive load. Increased merge conflict risk.

**Fix approach:**
- Split `research.py` into `research_queries.py`, `research_mutations.py`, `research_streaming.py`
- Extract store methods into separate service classes (e.g., `TrackQueries`, `PaperQueries`)
- Move orchestration logic out of route handlers into application services

### Memory Growth in Persistent E2B Sandboxes

**Issue:** In `src/paperbot/repro/e2b_executor.py:68-89`, persistent sandboxes (`keep_alive=True`) reuse sessions without memory cleanup

**Files:** `src/paperbot/repro/e2b_executor.py:63-89`

**Impact:**
- Long-running Paper2Code sessions accumulate intermediate artifacts (compiled code, test caches)
- Sandbox memory pressure increases over multiple reproductions
- Timeout handling: `_resolve_sandbox_timeout()` has best-effort refresh but unclear if aggressive

**Fix approach:**
- Add lifecycle hooks to clear `/tmp` and Python caches between reproductions
- Implement memory usage tracking with alerts at 70% threshold
- Add max-session-age limits (recreate sandbox every N hours)

### Vector Similarity Search Not Optimized for Scale

**Issue:** In `src/paperbot/infrastructure/stores/memory_store.py`, embedding queries use FTS5 with wrapped tokens but no vector indices

**Files:** `src/paperbot/infrastructure/stores/memory_store.py:997-1041`

**Impact:**
- SQLite vector search (via `sqlite-vec`) scans all embeddings without index
- Large memory stores (100k+ items) see O(n) query time
- No caching of embedding lookups

**Fix approach:**
- Add vector index creation in schema migrations
- Implement in-process LRU cache for frequently accessed embeddings
- Consider pgvector for PostgreSQL deployments

---

## Fragile Areas

### Broad Exception Handling Masks Silent Failures

**Area:** Exception handling throughout repro pipeline

**Files:**
- `src/paperbot/repro/nodes/planning_node.py:19, 153`
- `src/paperbot/repro/nodes/generation_node.py:26, 133, 263`
- `src/paperbot/repro/docker_executor.py:50, 115, 122`

**Why fragile:**
- `except Exception` catches both expected errors (JSON parse fails) and unexpected ones (memory errors, system crashes)
- Makes tests fragile: error messages change, tests break
- Hides bugs: typos in variable names silently caught and swallowed

**Safe modification approach:**
1. Identify specific error types each block should handle
2. Extract to named error subclasses (e.g., `PlanningValidationError`, `CodeGenerationError`)
3. Add logging before catch: `logger.debug(f"Caught {type(e).__name__}: {e}", exc_info=True)`
4. Test coverage: Add tests that trigger each exception path

**Test coverage gaps:**
- No tests for malformed JSON in generation node
- Docker executor exception paths untested
- Error serialization in event logs not verified

### Circular Import Risk in API Routes

**Issue:** Multiple route files use late imports to avoid circular dependency issues

**Files:**
- `src/paperbot/api/routes/events.py:37` - Comment explains "avoid circular import at module level"
- `src/paperbot/api/routes/track.py:25` - Imports inside functions
- `src/paperbot/mcp/tools/paper_search.py:4` - Uses `register(mcp)` pattern to avoid circulars

**Why fragile:**
- Late imports (inside functions) defeat static analysis tools
- Prevents linters from catching import-time errors
- Can cause surprise failures if function not called in some code paths

**Safe modification approach:**
1. Audit dependency graph to identify circular imports (tools: `pipdeptree`, `graphviz`)
2. Restructure modules to eliminate circles (introduce `core/` or `shared/` module)
3. Use dependency injection to break circular refs
4. Add pre-commit hook: `python -c "import src.paperbot"` to catch import errors

### Async Task Creation Without Tracking

**Issue:** In `src/paperbot/api/routes/repro_context.py:211`, `asyncio.create_task(_run())` creates fire-and-forget background task

**Files:** `src/paperbot/api/routes/repro_context.py:211, 516`

**Why fragile:**
- Task may outlive request lifecycle
- Exceptions in task not propagated
- No way to cancel or track completion from client
- Memory leak if task runs indefinitely

**Safe modification approach:**
1. Store task in request state or shared registry for later await
2. Add timeout wrapper: `asyncio.wait_for(task, timeout=3600)`
3. Add error handler: `task.add_done_callback(lambda t: logger.error(...) if t.exception() else None)`
4. For streaming responses, use SSE to report completion instead of fire-and-forget

---

## Scaling Limits

### Single-User Database Not Multi-Tenant Ready

**Issue:** User context (`get_required_user_id()` dependency) added post-hoc to routes, but underlying stores have no user_id filtering

**Files:**
- `src/paperbot/api/routes/research.py:39` - `get_required_user_id` imported but inconsistently used
- `src/paperbot/infrastructure/stores/research_store.py` - Methods lack user_id parameters

**Current capacity:** Single authenticated user per deployment

**Limit:** Adding multi-user requires schema changes to every table (add `user_id` FK, update all queries with `WHERE user_id = X`)

**Scaling path:**
1. Add `user_id` column to all domain tables (papers, tracks, feedback, embeddings)
2. Update all store queries to filter by user_id
3. Add unique constraints (user_id, domain_id) instead of just domain_id
4. Migrate data: bulk UPDATE to assign single user_id to all existing rows
5. Add middleware to enforce user_id from auth token

### In-Memory Caches Not Distributed

**Issue:** LLM providers, scholar cache, search index stored in-process memory

**Files:**
- `src/paperbot/agents/scholar_tracking/scholar_profile_agent.py:194, 203` - Cache service with `.clear_cache()`, `.clear_all_cache()`
- Multiple lazy singleton pattern: `_get_service()` functions in routes

**Current limit:** Single process/instance. Horizontal scaling requires cache invalidation protocol.

**Scaling path:**
- Migrate to Redis-backed caching (cache decorator with TTL)
- Add pub/sub for invalidation across instances
- Use ARQ (already in stack) for distributed task cache

### ARQ Job Queue Blocking on Long-Running Tasks

**Issue:** `src/paperbot/infrastructure/queue/arq_worker.py` processes DailyPaper cron, but no timeout/circuit-breaker for slow jobs

**Files:** `src/paperbot/infrastructure/queue/arq_worker.py`

**Current capacity:** Single Redis worker process. Long-running harvest/analysis blocks subsequent jobs.

**Limit:** ~10-20 parallel jobs before queue backs up. No SLA enforcement.

**Scaling path:**
1. Add per-job timeout: `arq_setting.timeout = 1800`
2. Implement circuit breaker: fail fast if queue depth exceeds threshold
3. Add job priority: high-priority items (user-triggered) over low-priority (scheduled)
4. Monitor: Redis memory, queue depth, job duration percentiles

---

## Dependencies at Risk

### Docker SDK Optional Dependency Without Graceful Fallback

**Issue:** In `src/paperbot/repro/docker_executor.py:6-14`, ImportError is caught but `HAS_DOCKER` flag doesn't prevent runtime errors

**Files:** `src/paperbot/repro/docker_executor.py:6-14`

**Risk:** If Docker SDK not installed, `.available()` returns False, but user may still call `.run()`, get opaque error

**Current mitigation:** `if not self.client: return ExecutionResult(status="error", ...)`

**Recommendations:**
- Add validation in executor selection logic (fail-fast if all executors unavailable)
- Document required extras: `pip install paperbot[sandbox]` includes docker SDK
- Add health check endpoint that verifies executor availability

### E2B SDK Dependency on API Key at Runtime

**Issue:** In `src/paperbot/repro/e2b_executor.py:84-93`, E2B initialization doesn't fail loudly if API key missing

**Files:** `src/paperbot/repro/e2b_executor.py:84-93`

**Risk:** `.available()` returns False silently, then fallback to Docker (if available). User confusion if intended executor unavailable.

**Recommendations:**
- Add explicit validation in config bootstrap: raise error if E2B requested but unconfigured
- Log warning at startup: "E2B not available, falling back to Docker"
- Document env vars required per executor in CLAUDE.md

---

## Missing Critical Features

### No Observability for Code Execution Failures

**Issue:** Code reproduction pipeline captures exit codes and logs, but no structured error categorization

**Files:**
- `src/paperbot/repro/nodes/verification_node.py:500-504` - Catches `VerificationRuntimePreparationError` but generic `Exception` swallows others
- `src/paperbot/repro/execution_result.py` - Simple `status` field ("success"/"failed"), no error_type field

**Impact:** Cannot distinguish: compilation error vs runtime error vs timeout vs missing dependency. UI cannot suggest fixes.

**Blocking:** Deep review feature requires error classification to suggest code fixes.

**Fix approach:**
- Add `error_category` enum to ExecutionResult: "compilation", "runtime", "dependency", "timeout", "permission"
- Parse stderr output to detect common patterns (ImportError, SyntaxError, TimeoutError, etc.)
- Return structured error metadata (line number, symbol name) for IDE integration

### No Semantic Deduplication of Papers

**Issue:** Paper deduplication only checks DOI, URL, title (exact match). Two papers with same abstract but different editions not caught.

**Files:** `src/paperbot/infrastructure/stores/research_store.py:2260-2289`

**Impact:** Duplicate paper recommendations. Inflated paper counts.

**Fix approach:**
- Add embeddings-based near-duplicate detection
- Hash abstract + venue + year as secondary unique key
- Use cosine similarity threshold (>0.95) to flag near-duplicates

---

## Test Coverage Gaps

### No Integration Tests for Scholar Tracking Workflows

**Issue:** Scholar profile agent (`src/paperbot/agents/scholar_tracking/scholar_profile_agent.py`) and tracking orchestration not covered by integration tests

**Files:** `src/paperbot/agents/scholar_tracking/`

**Risk:** Schema changes break scholar tracking silently. API behavior changes unnoticed.

**Test gaps:**
- No tests for scholar profile fetch + cache update cycle
- No tests for concurrent updates to same scholar
- No tests for missing Semantic Scholar records

**Priority:** Medium (core business logic, but less user-facing than paper analysis)

### Docker/E2B Executor Paths Not Tested in CI

**Issue:** Executors require Docker or E2B API key, so CI doesn't run `test_paper2code_*.py` suite

**Files:** `src/paperbot/repro/*.py` - All executor tests skipped in CI

**Risk:** Code generation bugs only found in production. Breaking changes to code generation undetected.

**Test coverage:** ~30% of repro module

**Fix approach:**
1. Mock executor in unit tests: stub `.run()` to return canned successful result
2. Integration tests: Use test containers (testcontainers-py) to spin up Docker in CI
3. E2E tests: Only run on PRs with special trigger (expensive/slow, don't run every commit)

### Async Error Handling in Streaming Routes Not Tested

**Issue:** SSE streaming error paths (client disconnect, generator exception, timeout) hard to test

**Files:** `src/paperbot/api/streaming.py:158-240`, route handlers using `sse_response()`

**Risk:** Silent stream failures, resource leaks if error handling code breaks

**Test gaps:**
- No test for client disconnect during stream
- No test for generator timeout at different durations
- No test for unhandled exception in event generator

**Fix approach:**
1. Mock StreamingResponse to capture events without full HTTP server
2. Simulate client disconnect with task cancellation
3. Test generator with controlled exception injection at different frame counts

