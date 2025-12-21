from __future__ import annotations

from paperbot.application.collaboration.message_schema import make_event, new_run_id, new_trace_id
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def test_sqlalchemy_event_log_persists_and_replays(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'paperbot_test.db'}"
    evlog = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)

    run_id = new_run_id()
    trace_id = new_trace_id()
    evlog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="test",
            stage="s1",
            attempt=0,
            agent_name="Test",
            role="system",
            type="score_update",
            payload={"x": 1},
        )
    )
    evlog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="test",
            stage="s2",
            attempt=1,
            agent_name="Test",
            role="system",
            type="stage_event",
            payload={"x": 2},
        )
    )

    events = evlog.list_events(run_id, limit=100)
    assert len(events) >= 2
    assert [events[0]["stage"], events[1]["stage"]] == ["s1", "s2"]

    streamed = list(evlog.stream(run_id))
    assert len(streamed) >= 2
    assert streamed[0]["stage"] == "s1"


