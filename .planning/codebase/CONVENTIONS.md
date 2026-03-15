# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files:**
- Python: snake_case (`paper_store.py`, `llm_service.py`, `schema_builder.py`)
- TypeScript/React: camelCase for components and utilities (`SentimentChart.tsx`, `auth.ts`)
- Test files: `test_*.py` or `*.test.ts` (match the module name: `test_paper_judge.py` for `paper_judge`)
- Directories: lowercase snake_case (`application`, `infrastructure`, `domain`, `agents`)

**Functions/Methods:**
- Python: snake_case
  - Actions: `create_`, `update_`, `delete_`, `fetch_`, `parse_`
  - Checks: `is_`, `has_`, `can_`
  - Internal: `_private_method()` for module-private; single underscore only (not double)
- TypeScript: camelCase
  - Event handlers: `handle*` (e.g., `handleClick`)
  - Getters: plain name or `get*` (e.g., `getSession()`)
  - Async functions: same casing, clearly typed as `Promise<T>`

**Variables:**
- Python: snake_case for all (module-level, local, instance)
- TypeScript: camelCase for locals and module-level; use `const` by default
- Boolean prefixes in Python: `is_`, `has_`, `can_` (e.g., `has_code`, `is_active`)
- Abbreviations: Avoid; spell out full names (use `service` not `svc`)

**Types:**
- Python Dataclasses: PascalCase (e.g., `PaperMeta`, `Scholar`, `Track`)
- Python Protocol/Interface types: Suffix with `Port` (e.g., `RegistryPort`, `EventLogPort`)
- TypeScript Interfaces: PascalCase (e.g., `SentimentChartProps`)
- Enum members (Python): SCREAMING_SNAKE_CASE (e.g., `MUST_READ`, `WORTH_READING`)

## Code Style

**Formatting:**
- **Python**: Black (line-length 100, target py310)
  - Run: `python -m black .`
- **TypeScript**: ESLint with Next.js config
  - Run: `npm run lint` (in `web/` dir)
  - Uses latest eslint v9 with flat config format
- **Indentation**: 4 spaces (Python), 2 spaces (TypeScript/JavaScript)

**Linting:**
- **Python**: pyright (basic mode, Python 3.10)
  - Config in `pyproject.toml` with `extraPaths = ["src"]`
  - Run: `pyright src/`
- **TypeScript**: ESLint with `eslint-config-next/core-web-vitals` and `eslint-config-next/typescript`
  - Config: `web/eslint.config.mjs` (flat config)
  - Ignores: `.next/`, `out/`, `build/`, `next-env.d.ts`

**isort (Python):**
- Profile: Black
- Line length: 100
- src_paths: `["src"]`

## Import Organization

**Python Order:**
1. Future imports (`from __future__ import ...`)
2. Standard library (`import os`, `from typing import`)
3. Third-party (`from pydantic import`, `from sqlalchemy import`)
4. Local application (`from paperbot.domain import`, `from paperbot.infrastructure import`)
5. Relative imports (rare; use absolute paths to `src.paperbot`)

**Example:**
```python
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel
from sqlalchemy import Column, String

from paperbot.domain.paper import PaperMeta
from paperbot.infrastructure.stores.models import PaperModel
```

**TypeScript Order:**
1. Built-in React/Next imports
2. External libraries (UI, utilities)
3. Local application imports
4. Relative imports (rare; use `@/*` alias)

**Path Aliases:**
- Python: None (use absolute `from src.paperbot...` or `from paperbot...` after setting PYTHONPATH)
- TypeScript: `@/*` maps to `web/src/*` (declared in `web/tsconfig.json`)

## Error Handling

**Patterns:**
- **Exceptions**: Use built-in exceptions when appropriate; create domain-specific exceptions in `domain/` if needed
- **Try/Except**: Catch specific exceptions; use `except Exception as exc:` with logging for fallbacks, not silently ignoring
- **Logging on Error**: Always log at minimum `warning` level before falling back
  ```python
  except Exception as exc:
      logger.warning("Operation failed: %s", exc)
      return fallback_value
  ```
- **Async Errors**: Use `except Exception as exc:` (async functions throw like sync); no special async exception handling needed
- **API Errors**: FastAPI routes catch exceptions and return appropriate HTTP status codes; use `HTTPException(status_code=..., detail=...)` for client errors

## Logging

**Framework:** Python built-in `logging` module

**Pattern:**
```python
import logging

logger = logging.getLogger(__name__)
```

**Usage:**
- Always use `logger.info()`, `logger.warning()`, `logger.debug()`, `logger.error()` — never `print()`
- Use %-formatting: `logger.warning("key=%s value=%d", key, value)` not f-strings (avoids unnecessary work if log level filters it)
- Log at appropriate levels:
  - `debug`: Detailed internal state (rarely needed)
  - `info`: Significant events (task start/end, key decisions)
  - `warning`: Recoverable errors, fallbacks, missing optional data
  - `error`: Unrecoverable errors (rarely used in this codebase; most raise exceptions instead)

**Do NOT log:**
- Secrets (API keys, tokens, passwords)
- PII (full email addresses, user IDs in sensitive contexts)
- Full payloads (log summary only, e.g., `status=200 latency_ms=45`)

## Comments

**When to Comment:**
- Non-obvious logic or algorithms
- Business rules that aren't self-evident from code
- References to external docs (links to issue, architecture doc, paper)
- Workarounds or temporary code: use `# TODO:` or `# FIXME:` with issue number
- Do NOT comment obvious code: `x = 5  # Set x to 5` is noise

**Docstrings (Python):**
- Use docstrings for modules, classes, and public functions
- Single-line docstrings for simple functions: `"""Fetch paper by ID."""`
- Multi-line for complex functions:
  ```python
  def judge_single(self, paper: dict, query: str) -> JudgeResult:
      """Evaluate a single paper against a research query.

      Args:
          paper: Paper dict with title, snippet
          query: Research topic string

      Returns:
          JudgeResult with scores and recommendation
      """
  ```
- Follow existing pattern in codebase (examples in `src/paperbot/domain/paper.py`)
- No type annotations in docstrings (use function signature instead)

**JSDoc (TypeScript):**
- Not consistently used in web codebase; inline comments preferred
- Keep functions small and self-documenting with clear names

## Function Design

**Size:** Keep functions under 50 lines when possible; >100 lines is a code smell

**Parameters:**
- Limit to 5 positional parameters; use `**kwargs` or dataclass for more
- Use keyword-only args after `*` for optional/configurable params:
  ```python
  def complete(self, *, task_type: str = "default", system: str, user: str) -> str:
  ```
- Avoid boolean parameters; use enums or separate methods instead

**Return Values:**
- Return early on error conditions (fail fast pattern):
  ```python
  if not data:
      return None
  # Main logic
  ```
- Async functions always return `Awaitable[T]`; use type hints
- Generator functions use `-> Generator[YieldType, SendType, ReturnType]`

## Module Design

**Exports:**
- `__all__` is not used in this codebase; rely on implicit public API (no leading `_`)
- Private module functions: use single leading underscore (`_helper_func()`)
- Private module attributes: use single underscore (`_instance`, `_cache`)

**Barrel Files:**
- Minimal use; each module imports what it needs directly
- Example: `src/paperbot/domain/__init__.py` may re-export key classes for convenience

**Single Responsibility:**
- Each module serves one purpose
- Utilities scattered into appropriate layers: `application/services/`, `infrastructure/stores/`, etc.
- Avoid circular imports (a sign of poor module organization)

**Async/Await:**
- All async functions decorated with `@pytest.mark.asyncio` in tests (strict mode)
- Async methods don't differ in naming from sync; always declare return type as `Awaitable[T]` or `-> Coroutine[...]`
- Use `async with` for context managers; `async for` for iterables

## File Structure & Length

**Module files:**
- Dataclass definitions: 50–150 lines (one or two domain classes per file)
- Service/adapter files: 100–300 lines (multiple public methods, internal helpers)
- Test files: 50–200 lines per test class/module (split large test suites)

**Large files (>500 lines) are red flags:**
- Consider splitting into sub-modules
- Example: `generation_node.py` (600 lines) has multiple agent classes and could split into `planning_agent.py`, `coding_agent.py`

---

*Convention analysis: 2026-03-15*
