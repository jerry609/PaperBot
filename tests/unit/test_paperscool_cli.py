import json

from paperbot.presentation.cli import main as cli_main


class _FakeWorkflow:
    def run(self, *, queries, sources, branches, top_k_per_query, show_per_branch):
        return {
            "source": "papers.cool",
            "fetched_at": "2026-02-09T00:00:00+00:00",
            "sources": sources,
            "queries": [
                {
                    "raw_query": queries[0],
                    "normalized_query": "icl compression",
                    "tokens": ["icl", "compression"],
                    "total_hits": 1,
                    "items": [],
                }
            ],
            "items": [],
            "summary": {
                "unique_items": 1,
                "total_query_hits": 1,
                "top_titles": ["UniICL"],
                "source_breakdown": {sources[0]: 1},
                "query_highlights": [
                    {
                        "raw_query": queries[0],
                        "normalized_query": "icl compression",
                        "hit_count": 1,
                        "top_title": "UniICL",
                        "top_keywords": ["icl", "compression"],
                    }
                ],
            },
        }


def test_cli_topic_search_parser_flags():
    parser = cli_main.create_parser()
    args = parser.parse_args(
        [
            "topic-search",
            "-q",
            "ICL压缩",
            "--source",
            "papers_cool",
            "--branch",
            "arxiv",
            "--top-k",
            "3",
            "--show",
            "20",
            "--json",
        ]
    )

    assert args.command == "topic-search"
    assert args.queries == ["ICL压缩"]
    assert args.sources == ["papers_cool"]
    assert args.branches == ["arxiv"]
    assert args.top_k == 3
    assert args.show == 20
    assert args.json is True


def test_cli_topic_search_json_output(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_create_topic_search_workflow", lambda: _FakeWorkflow())

    exit_code = cli_main.run_cli(["topic-search", "-q", "ICL压缩", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["source"] == "papers.cool"
    assert payload["summary"]["unique_items"] == 1


def test_cli_daily_paper_json_output(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_create_topic_search_workflow", lambda: _FakeWorkflow())

    exit_code = cli_main.run_cli(["daily-paper", "-q", "ICL压缩", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["report"]["title"] == "DailyPaper Digest"
    assert "# DailyPaper Digest" in payload["markdown"]


def test_cli_daily_paper_parser_with_llm_flags():
    parser = cli_main.create_parser()
    args = parser.parse_args(
        [
            "daily-paper",
            "-q",
            "ICL压缩",
            "--with-llm",
            "--llm-feature",
            "summary",
            "--llm-feature",
            "trends",
        ]
    )

    assert args.with_llm is True
    assert args.llm_features == ["summary", "trends"]


def test_cli_daily_paper_json_output_with_llm(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_create_topic_search_workflow", lambda: _FakeWorkflow())
    monkeypatch.setattr(
        cli_main,
        "enrich_daily_paper_report",
        lambda report, llm_features: {**report, "llm_analysis": {"enabled": True, "features": llm_features}},
    )

    exit_code = cli_main.run_cli(
        ["daily-paper", "-q", "ICL压缩", "--json", "--with-llm", "--llm-feature", "summary"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "llm_analysis" in payload["report"]


def test_cli_daily_paper_parser_with_judge_flags():
    parser = cli_main.create_parser()
    args = parser.parse_args(
        [
            "daily-paper",
            "-q",
            "ICL压缩",
            "--with-judge",
            "--judge-runs",
            "2",
            "--judge-max-items",
            "6",
        ]
    )

    assert args.with_judge is True
    assert args.judge_runs == 2
    assert args.judge_max_items == 6


def test_cli_daily_paper_json_output_with_judge(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_create_topic_search_workflow", lambda: _FakeWorkflow())

    def _fake_judge(report, max_items_per_query, n_runs):
        report = dict(report)
        report["judge"] = {
            "enabled": True,
            "max_items_per_query": max_items_per_query,
            "n_runs": n_runs,
            "recommendation_count": {"must_read": 1, "worth_reading": 0, "skim": 0, "skip": 0},
        }
        return report

    monkeypatch.setattr(cli_main, "apply_judge_scores_to_report", _fake_judge)

    exit_code = cli_main.run_cli(
        [
            "daily-paper",
            "-q",
            "ICL压缩",
            "--json",
            "--with-judge",
            "--judge-runs",
            "2",
            "--judge-max-items",
            "4",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["report"]["judge"]["enabled"] is True
