from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.stores.author_store import AuthorStore
from paperbot.infrastructure.stores.paper_store import PaperStore


def test_upsert_author_deduplicates_by_name(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'author-store.db'}"
    store = AuthorStore(db_url=db_url)

    first = store.upsert_author(name="Yoshua Bengio")
    second = store.upsert_author(name="  yoshua   bengio  ")

    assert first["id"] == second["id"]
    assert second["author_id"].startswith("name:")
    assert second["slug"] == "yoshua-bengio"


def test_replace_and_get_paper_authors(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'paper-authors.db'}"
    paper_store = PaperStore(db_url=db_url)
    author_store = AuthorStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "paper_id": "1706.03762",
            "url": "https://arxiv.org/abs/1706.03762",
        },
        source_hint="arxiv",
    )

    linked = author_store.replace_paper_authors(
        paper_id=int(paper["id"]),
        authors=[
            {"name": "Ashish Vaswani", "is_corresponding": True},
            "Noam Shazeer",
        ],
    )

    assert len(linked) == 2
    assert linked[0]["name"] == "Ashish Vaswani"
    assert linked[0]["is_corresponding"] is True
    assert linked[1]["author_order"] == 1

    reloaded = author_store.get_paper_authors(paper_id=int(paper["id"]))
    assert [row["name"] for row in reloaded] == ["Ashish Vaswani", "Noam Shazeer"]


def test_get_author_and_list_authors_return_persisted_rows(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'author-read.db'}"
    store = AuthorStore(db_url=db_url)

    bengio = store.upsert_author(name="Yoshua Bengio")
    lecun = store.upsert_author(name="Yann LeCun")

    loaded = store.get_author(int(bengio["id"]))
    assert loaded is not None
    assert loaded["name"] == "Yoshua Bengio"

    authors = store.list_authors(limit=10, offset=0)
    assert [row["name"] for row in authors] == ["Yann LeCun", "Yoshua Bengio"]
    assert {int(row["id"]) for row in authors} == {int(bengio["id"]), int(lecun["id"])}
