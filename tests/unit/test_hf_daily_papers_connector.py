from paperbot.infrastructure.connectors.hf_daily_papers_connector import HFDailyPapersConnector


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_hf_daily_connector_search_filters_and_parses(monkeypatch):
    payload_by_page = {
        0: [
            {
                "paper": {
                    "id": "2602.12345",
                    "title": "KV Cache Compression for Faster LLM Serving",
                    "summary": "We accelerate long-context decoding with KV cache pruning.",
                    "publishedAt": "2026-02-08T01:00:00.000Z",
                    "submittedOnDailyAt": "2026-02-10T06:00:00.000Z",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "ai_keywords": ["kv cache", "acceleration"],
                    "upvotes": 17,
                }
            },
            {
                "paper": {
                    "id": "2602.77777",
                    "title": "Diffusion Model for Images",
                    "summary": "Not related to query.",
                    "publishedAt": "2026-02-07T01:00:00.000Z",
                    "submittedOnDailyAt": "2026-02-10T05:00:00.000Z",
                    "authors": [{"name": "Carol"}],
                    "ai_keywords": ["diffusion"],
                    "upvotes": 30,
                }
            },
        ],
        1: [
            {
                "paper": {
                    "id": "2602.99999",
                    "title": "A Survey on KV Cache Management",
                    "summary": "Comprehensive survey for inference acceleration.",
                    "publishedAt": "2026-02-08T01:00:00.000Z",
                    "submittedOnDailyAt": "2026-02-09T06:00:00.000Z",
                    "authors": [{"name": "Dave"}],
                    "ai_keywords": ["kv", "cache"],
                    "upvotes": 3,
                }
            }
        ],
    }

    calls = []

    def _fake_get(url, params, headers, timeout):
        calls.append(
            {"url": url, "params": dict(params), "headers": dict(headers), "timeout": timeout}
        )
        page = int(params.get("p", 0))
        return _DummyResponse(payload_by_page.get(page, []))

    monkeypatch.setattr(
        "paperbot.infrastructure.connectors.hf_daily_papers_connector.requests.get", _fake_get
    )

    connector = HFDailyPapersConnector()
    rows = connector.search(
        query="kv cache acceleration",
        max_results=5,
        page_size=100,
        max_pages=2,
    )

    assert len(rows) == 2
    assert rows[0].paper_id == "2602.12345"
    assert rows[0].title.startswith("KV Cache Compression")
    assert rows[0].paper_url == "https://huggingface.co/papers/2602.12345"
    assert rows[0].external_url == "https://arxiv.org/abs/2602.12345"
    assert rows[0].pdf_url == "https://arxiv.org/pdf/2602.12345.pdf"

    assert calls[0]["url"] == "https://huggingface.co/api/daily_papers"
    assert calls[0]["params"]["limit"] == 100
    assert calls[0]["params"]["p"] == 0
    assert calls[1]["params"]["p"] == 1


def test_hf_daily_connector_search_handles_empty_query():
    connector = HFDailyPapersConnector()
    rows = connector.search(query="   ", max_results=5)
    assert rows == []


def test_hf_daily_connector_fetch_uses_cache(monkeypatch):
    calls = []

    def _fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": dict(params)})
        return _DummyResponse(
            [
                {
                    "paper": {
                        "id": "2602.12345",
                        "title": "Cached Paper",
                    }
                }
            ]
        )

    monkeypatch.setattr(
        "paperbot.infrastructure.connectors.hf_daily_papers_connector.requests.get", _fake_get
    )

    connector = HFDailyPapersConnector(cache_ttl_s=600)
    first = connector.fetch_daily_papers(limit=20, page=0)
    second = connector.fetch_daily_papers(limit=20, page=0)

    assert first == second
    assert len(calls) == 1
    assert connector.metrics["cache_hits"] == 1


def test_hf_daily_connector_get_trending_modes(monkeypatch):
    payload = [
        {
            "paper": {
                "id": "2602.00002",
                "title": "Older but many upvotes",
                "submittedOnDailyAt": "2026-02-01T00:00:00.000Z",
                "upvotes": 100,
            }
        },
        {
            "paper": {
                "id": "2602.00003",
                "title": "Newest paper",
                "submittedOnDailyAt": "2026-02-10T00:00:00.000Z",
                "upvotes": 30,
            }
        },
    ]

    def _fake_get(url, params, headers, timeout):
        return _DummyResponse(payload if int(params.get("p", 0)) == 0 else [])

    monkeypatch.setattr(
        "paperbot.infrastructure.connectors.hf_daily_papers_connector.requests.get", _fake_get
    )

    connector = HFDailyPapersConnector(cache_ttl_s=0)
    hot = connector.get_trending(mode="hot", limit=2)
    newest = connector.get_trending(mode="new", limit=2)

    assert hot[0].paper_id == "2602.00002"
    assert newest[0].paper_id == "2602.00003"


def test_hf_daily_connector_search_degrades_on_request_error(monkeypatch):
    class _Boom(Exception):
        pass

    def _fake_get(url, params, headers, timeout):
        raise _Boom("timeout")

    monkeypatch.setattr(
        "paperbot.infrastructure.connectors.hf_daily_papers_connector.requests.get", _fake_get
    )

    connector = HFDailyPapersConnector()
    rows = connector.search(query="kv cache", max_results=5)

    assert rows == []
    assert connector.metrics["errors"] >= 1
    assert connector.metrics["degraded"] >= 1
