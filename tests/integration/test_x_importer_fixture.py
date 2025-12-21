from pathlib import Path

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.infrastructure.connectors.x_importer import XImporter
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def test_x_importer_jsonl_fixture_emits_events(tmp_path):
    lines = Path("evals/fixtures/x/twitter_sample.jsonl").read_text(encoding="utf-8").splitlines()
    imp = XImporter()
    records = imp.parse_jsonl(lines)
    assert len(records) == 2
    assert records[0].post_id

    db_url = f"sqlite:///{tmp_path / 'x_it.db'}"
    elog = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)
    run_id = new_run_id()
    imp.emit_events(records, event_log=elog, run_id=run_id, trace_id=new_trace_id())
    events = elog.list_events(run_id, limit=10)
    assert len(events) == 2
    assert events[0]["payload"]["source"] == "twitter_x"


