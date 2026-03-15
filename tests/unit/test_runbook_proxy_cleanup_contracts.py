from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_API_ROOT = ROOT / "web" / "src" / "app" / "api" / "runbook"
REMOVED_ROUTE_FILES = (
    RUNBOOK_API_ROOT / "runs" / "[runId]" / "route.ts",
    RUNBOOK_API_ROOT / "smoke" / "route.ts",
)
REMOVED_UPSTREAM_PATTERNS = (
    "/api/runbook/runs/",
    "/api/runbook/smoke",
)


def test_runbook_next_routes_do_not_reference_removed_backend_endpoints() -> None:
    for path in REMOVED_ROUTE_FILES:
        assert not path.exists(), f"stale route file still present: {path.relative_to(ROOT)}"

    for path in sorted(RUNBOOK_API_ROOT.rglob("route.ts")):
        content = path.read_text(encoding="utf-8")
        for pattern in REMOVED_UPSTREAM_PATTERNS:
            assert pattern not in content, f"{path.relative_to(ROOT)} still references {pattern}"
