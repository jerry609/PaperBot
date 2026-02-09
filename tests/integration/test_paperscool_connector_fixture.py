from pathlib import Path

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.infrastructure.connectors.paperscool_connector import PapersCoolConnector
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def test_paperscool_connector_parses_arxiv_and_venue_fixtures(tmp_path):
    connector = PapersCoolConnector()

    arxiv_html = Path("evals/fixtures/paperscool/arxiv_search_sample.html").read_text(
        encoding="utf-8"
    )
    venue_html = Path("evals/fixtures/paperscool/venue_search_sample.html").read_text(
        encoding="utf-8"
    )

    arxiv_records = connector.parse_search_html(arxiv_html, branch="arxiv")
    venue_records = connector.parse_search_html(venue_html, branch="venue")

    assert len(arxiv_records) == 2
    assert len(venue_records) == 2

    assert arxiv_records[0].paper_id == "2412.19442"
    assert arxiv_records[0].url == "https://papers.cool/arxiv/2412.19442"
    assert arxiv_records[0].authors == ["Haoyang Li", "Yiming Li"]
    assert arxiv_records[0].pdf_stars == 24
    assert arxiv_records[0].kimi_stars == 27

    assert venue_records[0].paper_id == "2025.acl-long.24@ACL"
    assert venue_records[0].source_branch == "venue"
    assert "compression" in venue_records[0].keywords

    db_url = f"sqlite:///{tmp_path / 'paperscool_it.db'}"
    event_log = SqlAlchemyEventLog(db_url=db_url, auto_create_schema=True)
    run_id = new_run_id()

    connector.emit_events(
        arxiv_records + venue_records, event_log=event_log, run_id=run_id, trace_id=new_trace_id()
    )
    events = event_log.list_events(run_id, limit=10)

    assert len(events) == 4
    assert events[0]["payload"]["source"] == "papers_cool"


def test_paperscool_connector_build_search_url():
    connector = PapersCoolConnector()
    url = connector.build_search_url(branch="arxiv", query="  kv cache acceleration  ", show=50)

    assert url.startswith("https://papers.cool/arxiv/search?")
    assert "query=kv+cache+acceleration" in url
    assert "highlight=1" in url
    assert "show=50" in url
