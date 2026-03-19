# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Runner:**
- pytest 7.0.0+
- Config: `pyproject.toml` with `asyncio_mode = "strict"`

**Key Setting:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
```

**Assertion Library:**
- Standard pytest assertions (no unittest.mock)
- Type hints in fixtures for IntelliSense

**Run Commands:**
```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/unit/test_di_container.py -v

# Run single test
pytest tests/unit/test_di_container.py::test_singleton_instance -v

# CI offline gates (all required tests)
PYTHONPATH=src pytest -q \
  tests/unit/test_scholar_from_config.py \
  tests/unit/test_paper_judge.py \
  tests/integration/test_eventlog_sqlalchemy.py \
  tests/e2e/test_api_track_fullstack_offline.py
```

**Development Coverage:**
- No coverage enforcement in config
- Manual check: `pytest --cov=src --cov-report=html`

## Test File Organization

**Location:**
- Co-located alongside implementation: `tests/unit/test_*.py` mirrors `src/paperbot/`
- Nested structure: `tests/unit/repro/test_*.py` for `src/paperbot/repro/`
- Three tiers: `unit/`, `integration/`, `e2e/`

**Naming:**
- `test_*.py` for test modules (always prefix with `test_`)
- Test functions: `test_<function>_<scenario>` (e.g., `test_paper_judge_single_parses_scores_and_overall`)
- Test classes: `Test<Feature>` (e.g., `TestContainer`)

**Structure:**
```
tests/
├── unit/                    # Fast, isolated, no network/DB
├── unit/repro/              # Paper2Code tests
├── integration/             # Slow, real DB/SQLite, limited network
├── e2e/                     # Full stack, realistic scenarios
└── evals/                   # Performance benchmarks and smoke tests
    ├── runners/             # Evaluation scripts (run_scholar_pipeline_smoke.py)
    ├── memory/              # Memory module tests (deletion_compliance, retrieval_hit_rate)
    └── ...
```

## Test Structure

**Suite Organization (Class-based):**
```python
class TestContainer:
    """Container 测试"""

    def setup_method(self):
        """每个测试前重置容器"""
        Container._instance = None

    def test_singleton_instance(self):
        """单个测试方法"""
        c1 = Container.instance()
        c2 = Container.instance()
        assert c1 is c2
```

**Suite Organization (Function-based):**
```python
def test_paper_judge_single_parses_scores_and_overall():
    """Single test function - no class wrapper"""
    judge = PaperJudge(llm_service=_FakeLLMService(payload))
    result = judge.judge_single(paper={"title": "x", "snippet": "y"}, query="icl")
    assert result.relevance.score == 5
```

**Patterns:**
- `setup_method()` runs before each test in a class (reset singletons, containers)
- Use fixtures only when necessary for shared heavy setup (DB, HTTP mocking)
- Import imports conditionally inside tests to support offline runs:
  ```python
  try:
      from src.paperbot.core.di import Container
  except ImportError:
      from core.di import Container
  ```

**Async Tests:**
Every async test MUST have `@pytest.mark.asyncio` (strict mode):
```python
@pytest.mark.asyncio
async def test_api_track_fullstack_offline_emits_db_events(monkeypatch, tmp_path):
    """Async test with strict asyncio_mode"""
    from paperbot.api import main as api_main
    monkeypatch.setenv("PAPERBOT_DB_URL", f"sqlite:///{tmp_path / 'test.db'}")
    # ... async test body
```

## Mocking

**Framework:** No unittest.mock; use stub classes instead

**Pattern:**
```python
class _FakeLLMService:
    """Fake LLM for testing - methods match real interface"""
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        """Matches LLMService.complete signature"""
        return json.dumps(self.payload)

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "fake", "cost_tier": 2}
```

**HTTP Mocking:**
- `respx` for `httpx` requests (real-world tests with specific API behavior)
- `aioresponses` for `aiohttp` requests
- Pattern:
  ```python
  with respx.mock:
      respx.get("https://api.example.com/path").mock(return_value=httpx.Response(200, json={"key": "value"}))
      # Test code that calls the API
  ```

**Database:**
- Use `tmp_path` fixture for temp SQLite: `f"sqlite:///{tmp_path / 'test.db'}"`
- Always set `auto_create_schema=True` when creating stores with temp DB
- No shared test database; each test gets its own

**What to Mock:**
- External HTTP services (Semantic Scholar API, arXiv, OpenAI, etc.)
- LLM calls (use `_FakeLLMService` with deterministic payloads)
- Long-running operations (sleep, delays)

**What NOT to Mock:**
- Database layer (use temp SQLite instead)
- Business logic (test real logic, not mocked happy paths)
- Configuration/environment (use monkeypatch, not mocks)
- Dataclass initialization (test real objects)

## Fixtures and Factories

**Test Data:**
No factory libraries used; create test data inline:
```python
def _stub_papers() -> List[Dict[str, Any]]:
    return [
        {
            "paper_id": "e2e_paper_001",
            "title": "E2E Offline Paper",
            "authors": ["Alice"],
            "abstract": "offline",
            "year": 2025,
        }
    ]

@pytest.mark.asyncio
async def test_something():
    papers = _stub_papers()
    # Use papers in test
```

**Fixtures (pytest):**
- Rarely used; prefer inline setup
- Example fixture in `tests/conftest.py` (if shared across multiple files):
  ```python
  @pytest.fixture
  def tmp_db_path(tmp_path):
      return f"sqlite:///{tmp_path / 'test.db'}"
  ```

**Location:**
- Module-level helpers: `_function_name()` (leading underscore, lowercase)
- Test-specific data: inline in test function
- Shared fixtures: `tests/conftest.py` (if any)

## Coverage

**Requirements:** No enforced coverage target

**View Coverage:**
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

**Coverage Gaps Identified:**
- Paper2Code module (`repro/`) has good unit coverage but limited e2e
- API endpoints have integration/e2e tests, not isolated unit tests (by design)
- Streaming logic (`streaming.py`) implicitly tested via e2e

## Test Types

**Unit Tests** (`tests/unit/`):
- Fast (< 100ms each)
- Isolated: single class/function tested in isolation
- No network, no real database (use temp SQLite if DB needed)
- Examples:
  - `test_di_container.py`: Container registration and resolution
  - `test_paper_judge.py`: Judge scoring logic with fake LLM
  - `test_pipeline.py`: Pipeline execution with stubs

**Integration Tests** (`tests/integration/`):
- Slower (100ms–2s each)
- Real database (temp SQLite), real data structures
- Limited external calls (use monkeypatch to intercept)
- Examples:
  - `test_eventlog_sqlalchemy.py`: Event persistence and replay
  - `test_crawler_contract_parsers.py`: HTML parsing with real parsers
  - `test_repro_deepcode.py`: Multi-stage Paper2Code execution

**E2E Tests** (`tests/e2e/`):
- Slow (seconds each)
- Full stack: FastAPI app, real routes, real logic
- Network calls stubbed (monkeypatch or respx)
- Examples:
  - `test_api_track_fullstack_offline.py`: Full scholar tracking pipeline
  - `test_events_sse_endpoint.py`: SSE streaming and event log

**Eval Smoke Tests** (`evals/runners/`):
- Verify critical paths work (not broken by refactoring)
- Examples:
  - `run_scholar_pipeline_smoke.py`: Scholar tracking end-to-end
  - `run_track_pipeline_smoke.py`: Paper tracking pipeline
  - `run_eventlog_replay_smoke.py`: Event log persistence and replay

**Eval Memory Tests** (`evals/memory/`):
- Acceptance tests for memory module behavior
- Examples:
  - `test_deletion_compliance.py`: Verify deleted items are not retrievable
  - `test_retrieval_hit_rate.py`: Measure retrieval accuracy
  - `test_scope_isolation.py`: Verify scope boundaries

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_async_service():
    service = MyService()
    result = await service.fetch()
    assert result is not None
```

**Error Testing:**
```python
def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="title required"):
        Judge(paper={}, query="q")
```

**Monkeypatch for Environment:**
```python
@pytest.mark.asyncio
async def test_with_custom_db(monkeypatch):
    monkeypatch.setenv("PAPERBOT_DB_URL", "sqlite:///test.db")
    # Import after monkeypatch to pick up env var
    from paperbot.infrastructure.stores.sqlalchemy_db import get_db_url
    assert "test.db" in get_db_url()
```

**Monkeypatch for Functions:**
```python
@pytest.mark.asyncio
async def test_stubbed_network_call(monkeypatch):
    async def _stub_fetch(url):
        return {"status": "ok"}

    monkeypatch.setattr("paperbot.agents.agent.fetch", _stub_fetch)
    # Now calls to fetch() use the stub
```

**Setup Singleton Reset (DI Container):**
```python
class TestMyClass:
    def setup_method(self):
        """Reset before each test"""
        Container._instance = None

    def test_something(self):
        container = Container.instance()
        # Test uses fresh container
```

**Temp Database in Integration Tests:**
```python
def test_with_sqlite(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    store = PaperStore(db_url=db_url, auto_create_schema=True)
    # Test uses isolated temp DB
```

## Test Examples by Module

**Domain Models** (`tests/unit/`):
- Dataclass initialization, serialization (to_dict, from_dict)
- Immutability where expected
- No external calls

**Application Services** (`tests/unit/`):
- Business logic with fake dependencies
- Edge cases and error conditions
- Deterministic outputs from fakes

**Infrastructure Adapters** (`tests/integration/`):
- Real database layer (temp SQLite)
- API client behavior (respx mocks)
- Serialization round-trips

**API Routes** (`tests/e2e/`):
- Full request/response cycle via TestClient
- Event log persistence
- SSE streaming behavior

**Agents** (`tests/unit/` + `tests/integration/`):
- Agent orchestration with stub services
- Prompt handling and output parsing
- Async execution patterns

---

*Testing analysis: 2026-03-15*
