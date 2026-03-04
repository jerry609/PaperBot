"""Tests for MinerU Cloud API client."""
from paperbot.infrastructure.extractors.mineru_client import Figure, MineruClient


def test_no_api_key_returns_empty():
    client = MineruClient(api_key="")
    result = client.extract_figures("https://arxiv.org/pdf/2401.00001.pdf")
    assert result == []


def test_empty_url_returns_empty():
    client = MineruClient(api_key="test-key")
    result = client.extract_figures("")
    assert result == []


def test_validate_source_url_requires_http():
    client = MineruClient(api_key="test-key")
    try:
        client._validate_source_url("file:///tmp/a.pdf")
        assert False, "expected ValueError for non-http source"
    except ValueError as exc:
        assert "http(s)" in str(exc)


def test_validate_source_url_rejects_github_and_aws():
    client = MineruClient(api_key="test-key")
    for url in (
        "https://github.com/a/b/raw/main/paper.pdf",
        "https://raw.githubusercontent.com/a/b/main/paper.pdf",
        "https://bucket.s3.amazonaws.com/paper.pdf",
    ):
        try:
            client._validate_source_url(url)
            assert False, f"expected ValueError for URL: {url}"
        except ValueError as exc:
            assert "github/aws" in str(exc)


def test_extract_figures_rejects_unsupported_host_early(monkeypatch):
    client = MineruClient(api_key="test-key")

    def _should_not_call(*args, **kwargs):
        raise AssertionError("network call should not be reached for unsupported URL")

    monkeypatch.setattr(client, "_create_task", _should_not_call)
    result = client.extract_figures("https://github.com/a/b/raw/main/paper.pdf")
    assert result == []


def test_parse_figures_from_api_response():
    client = MineruClient(api_key="test-key")
    data = {
        "figures": [
            {
                "url": "https://cdn.mineru.net/fig1.png",
                "caption": "Figure 1: System overview and architecture",
                "page": 1,
                "width": 800,
                "height": 600,
            },
            {
                "url": "https://cdn.mineru.net/fig2.png",
                "caption": "Figure 2: Ablation results",
                "page": 5,
                "width": 400,
                "height": 300,
            },
            {
                "url": "",
                "caption": "Missing URL",
                "page": 3,
            },
        ]
    }
    figures = client._parse_figures(data)
    assert len(figures) == 2
    assert figures[0].url == "https://cdn.mineru.net/fig1.png"
    assert figures[0].page == 1
    assert figures[0].area == 800 * 600
    assert figures[1].caption == "Figure 2: Ablation results"


def test_parse_figures_from_images_key():
    """API may return 'images' instead of 'figures'."""
    client = MineruClient(api_key="test-key")
    data = {
        "images": [
            {
                "image_url": "https://cdn.mineru.net/img1.png",
                "caption": "Fig. 1: Overview",
                "page_number": 2,
                "width": 600,
                "height": 400,
            },
        ]
    }
    figures = client._parse_figures(data)
    assert len(figures) == 1
    assert figures[0].url == "https://cdn.mineru.net/img1.png"
    assert figures[0].page == 2


def test_parse_figures_from_markdown_zip_refs():
    client = MineruClient(api_key="test-key")
    markdown = """
![](images/fig1.jpg)
Figure 1: System overview

![](https://cdn.mineru.net/fig2.png)
Fig. 2: Attention map
""".strip()
    zip_url = "https://cdn-mineru.example.com/result.zip"

    figures = client._parse_figures_from_markdown(markdown, zip_url=zip_url)

    assert len(figures) == 2
    assert figures[0].url == f"{zip_url}#/images/fig1.jpg"
    assert figures[0].caption == "Figure 1: System overview"
    assert figures[1].url == "https://cdn.mineru.net/fig2.png"
    assert figures[1].caption == "Fig. 2: Attention map"


def test_identify_main_figure_prefers_figure_1():
    figures = [
        Figure(url="fig2.png", caption="Figure 2: Results table", page=4, width=400, height=300),
        Figure(
            url="fig1.png",
            caption="Figure 1: System architecture overview",
            page=1,
            width=600,
            height=400,
        ),
        Figure(url="fig3.png", caption="Figure 3: Comparison chart", page=6, width=500, height=350),
    ]
    client = MineruClient(api_key="test")
    main = client.identify_main_figure(figures)
    assert main is not None
    assert main.url == "fig1.png"


def test_identify_main_figure_prefers_architecture_keyword():
    figures = [
        Figure(url="table.png", caption="Table of results", page=5, width=400, height=300),
        Figure(url="arch.png", caption="Our proposed framework", page=2, width=500, height=400),
    ]
    client = MineruClient(api_key="test")
    main = client.identify_main_figure(figures)
    assert main is not None
    assert main.url == "arch.png"


def test_identify_main_figure_empty_list():
    client = MineruClient(api_key="test")
    assert client.identify_main_figure([]) is None


def test_identify_main_figure_filters_tiny():
    figures = [
        Figure(url="icon.png", caption="", page=1, width=20, height=20),
        Figure(url="main.png", caption="overview", page=1, width=600, height=400),
    ]
    client = MineruClient(api_key="test")
    main = client.identify_main_figure(figures)
    assert main is not None
    assert main.url == "main.png"


def test_figure_area():
    f = Figure(url="x.png", width=100, height=200)
    assert f.area == 20000


def test_extract_figures_for_report():
    """Test the pipeline-level figure extraction function."""
    from paperbot.application.workflows.dailypaper import extract_figures_for_report

    report = {
        "queries": [
            {
                "top_items": [
                    {"title": "Paper A", "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf"},
                    {"title": "Paper B"},  # No PDF URL
                ]
            }
        ]
    }
    # Without API key, report is returned unchanged
    result = extract_figures_for_report(report, api_key="")
    assert result is report  # Same object, not copied

    # With API key but no actual API call (will fail gracefully)
    result = extract_figures_for_report(report, api_key="fake-key", max_items=1)
    # Should return a copy even if extraction fails (graceful fallback)
    assert isinstance(result, dict)


def test_extract_figures_uses_local_cache(tmp_path, monkeypatch):
    url = "https://arxiv.org/pdf/2401.00001.pdf"
    client = MineruClient(api_key="test-key", cache_dir=str(tmp_path), cache_ttl_seconds=3600)

    calls = {"count": 0}

    def _fake_call_extract(pdf_url: str):
        calls["count"] += 1
        return [
            Figure(
                url="https://cdn.mineru.net/fig-main.png",
                caption="Figure 1: Overview",
                page=1,
                width=800,
                height=600,
            )
        ]

    monkeypatch.setattr(client, "_call_extract", _fake_call_extract)

    first = client.extract_figures(url)
    second = client.extract_figures(url)

    assert calls["count"] == 1
    assert len(first) == 1
    assert len(second) == 1
    assert second[0].url == "https://cdn.mineru.net/fig-main.png"
