from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Ensure local imports work without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id, make_event
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


async def main() -> Dict[str, Any]:
    tmp_root = Path(tempfile.mkdtemp(prefix="paperbot-eventlog-smoke-"))
    db_path = tmp_root / "paperbot.db"
    db_url = f"sqlite:///{db_path}"
    try:
        evlog = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)

        run_id = new_run_id()
        trace_id = new_trace_id()

        ev1 = make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="smoke",
            stage="stage1",
            attempt=0,
            agent_name="Smoke",
            role="system",
            type="score_update",
            payload={"value": 1},
        )
        ev2 = make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="smoke",
            stage="stage2",
            attempt=1,
            agent_name="Smoke",
            role="system",
            type="stage_event",
            payload={"value": 2},
        )

        evlog.append(ev1)
        evlog.append(ev2)

        events = evlog.list_events(run_id, limit=100)
        assert len(events) >= 2
        assert events[0]["stage"] == "stage1"
        assert events[1]["stage"] == "stage2"

        streamed = list(evlog.stream(run_id))
        assert len(streamed) >= 2

        return {
            "status": "ok",
            "db_url": db_url,
            "run_id": run_id,
            "trace_id": trace_id,
            "events_count": len(events),
            "first_two_stages": [events[0]["stage"], events[1]["stage"]],
        }
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    print(json.dumps(asyncio.run(main()), ensure_ascii=False, indent=2))


