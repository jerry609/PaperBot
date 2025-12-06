import asyncio
import pytest
from agents.conference_research_agent import ConferenceResearchAgent


class DummyDownloader:
    """Mock downloader to simulate download_paper and get_conference_papers."""

    def __init__(self):
        self.download_calls = 0

    async def download_paper(self, url, title, **kwargs):
        self.download_calls += 1
        return {"path": f"/tmp/{title}.pdf", "success": True}

    async def get_conference_papers(self, conf, year):
        return [
            {"title": "p1", "url": "http://example.com/p1.pdf"},
            {"title": "p2", "url": "http://example.com/p2.pdf"},
        ]


@pytest.mark.asyncio
async def test_conference_stats_output(monkeypatch, capsys):
    agent = ConferenceResearchAgent({"max_concurrency": 1, "rate_limit_per_sec": 0})
    dummy = DummyDownloader()
    agent.downloader = dummy  # replace downloader

    async def fake_extract(*args, **kwargs):
        return []

    agent._extract_github_links = fake_extract
    agent._extract_github_from_html = fake_extract

    result = await agent.process("sp", "23")
    captured = capsys.readouterr().out
    assert result["conference"] == "sp"
    assert "total=2" in captured
    assert "downloaded=2" in captured

