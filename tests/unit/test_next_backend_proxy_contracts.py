from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_ROOT = ROOT / "web" / "src" / "app" / "api"


def _route_files() -> list[Path]:
    return sorted(
        path
        for path in API_ROOT.rglob("route.ts")
        if "_utils" not in path.parts
    )


def test_next_api_routes_use_shared_backend_proxy_helpers() -> None:
    offenders: list[str] = []

    for file_path in _route_files():
        text = file_path.read_text(encoding="utf-8")
        rel = file_path.relative_to(ROOT)

        if "fetch(" in text:
            offenders.append(f"{rel}: direct fetch")

        if "process.env.PAPERBOT_API_BASE_URL" in text or "process.env.BACKEND_BASE_URL" in text:
            offenders.append(f"{rel}: inlined backend base URL")

    assert not offenders, "\n".join(offenders)


def test_dead_runbook_proxy_routes_are_removed() -> None:
    assert not (API_ROOT / "runbook" / "runs" / "[runId]" / "route.ts").exists()
    assert not (API_ROOT / "runbook" / "smoke" / "route.ts").exists()
