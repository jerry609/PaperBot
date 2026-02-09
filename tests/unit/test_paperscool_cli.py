import json

from paperbot.presentation.cli import main as cli_main


class _FakeWorkflow:
    def run(self, *, queries, branches, top_k_per_query, show_per_branch):
        return {
            "source": "papers.cool",
            "fetched_at": "2026-02-09T00:00:00+00:00",
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
