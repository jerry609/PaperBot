from pathlib import Path

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.infrastructure.connectors.reddit_connector import RedditConnector
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def test_reddit_connector_parses_fixture_and_emits_events(tmp_path):
    xml = Path("evals/fixtures/reddit/reddit_sample.rss.xml").read_text(encoding="utf-8")
    conn = RedditConnector()
    records = conn.parse_rss(xml, subreddit="MachineLearning")
    assert len(records) == 2
    assert records[0].link.startswith("https://")

    db_url = f"sqlite:///{tmp_path / 'reddit_it.db'}"
    elog = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)
    run_id = new_run_id()
    conn.emit_events(records, event_log=elog, run_id=run_id, trace_id=new_trace_id())
    events = elog.list_events(run_id, limit=10)
    assert len(events) == 2
    assert events[0]["payload"]["source"] == "reddit"


