from pathlib import Path

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.infrastructure.connectors.arxiv_connector import ArxivConnector
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def test_arxiv_connector_parses_fixture_and_emits_events(tmp_path):
    xml = Path("evals/fixtures/arxiv/arxiv_sample.atom.xml").read_text(encoding="utf-8")
    conn = ArxivConnector()
    records = conn.parse_atom(xml)
    assert len(records) == 1
    assert records[0].title
    assert records[0].pdf_url.endswith(".pdf")

    db_url = f"sqlite:///{tmp_path / 'arxiv_it.db'}"
    elog = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)
    run_id = new_run_id()
    conn.emit_events(records, event_log=elog, run_id=run_id, trace_id=new_trace_id())
    events = elog.list_events(run_id, limit=10)
    assert len(events) == 1
    assert events[0]["type"] == "source_record"
    assert events[0]["payload"]["source"] == "arxiv"


