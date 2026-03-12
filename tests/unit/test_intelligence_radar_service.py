from datetime import datetime, timedelta

from paperbot.infrastructure.services.intelligence_radar_service import (
    IntelligenceRadarService,
    _find_matches,
    _score_signal,
    _signal_sort_value,
    _utcnow,
)


def test_find_matches_is_case_insensitive_across_multiple_texts():
    matches = _find_matches(
        ["RAG", "Alice Zhang", "org/rag-agent"],
        [
            "Latest updates on rag systems",
            "Maintainer Alice Zhang posted a release note",
            "See org/rag-agent for the code",
        ],
    )

    assert matches == ["RAG", "Alice Zhang", "org/rag-agent"]


def test_build_match_reasons_surfaces_keyword_delta_author_and_repo():
    service = object.__new__(IntelligenceRadarService)

    reasons = service._build_match_reasons(
        {
            "keyword_hits": ["rag"],
            "metric_delta": 4,
            "author_matches": ["Alice Zhang"],
            "repo_matches": ["org/rag-agent"],
            "metric_name": "mentions/24h",
            "metric_value": 12,
        }
    )

    assert reasons == [
        "keyword: rag",
        "delta: +4",
        "author: Alice Zhang",
        "repo: org/rag-agent",
    ]


def test_score_signal_adds_breakdown_for_matches_and_freshness():
    score, breakdown = _score_signal(
        source="github",
        kind="repo_issue_heat",
        metric_value=12,
        metric_delta=5,
        published_at=_utcnow() - timedelta(hours=2),
        keyword_hits=["rag"],
        author_matches=["Alice Zhang"],
        repo_matches=["org/rag-agent"],
    )

    assert score > 0
    assert breakdown["keyword"] > 0
    assert breakdown["author"] > 0
    assert breakdown["repo"] > 0
    assert breakdown["freshness"] > 0
    assert breakdown["total"] == score


def test_signal_sort_value_supports_delta_keyword_and_time_modes():
    row = {
        "metric_delta": 7,
        "score": 42.5,
        "keyword_hits": ["rag"],
        "repo_full_name": "org/rag-agent",
        "published_at": (_utcnow() - timedelta(hours=1)).isoformat(),
        "detected_at": (_utcnow() - timedelta(minutes=20)).isoformat(),
        "source": "github",
    }

    assert _signal_sort_value(row, sort_by="delta") == 7
    assert _signal_sort_value(row, sort_by="keyword") == "rag"
    assert _signal_sort_value(row, sort_by="repo") == "org/rag-agent"
    assert _signal_sort_value(row, sort_by="published_at") > 0
    assert _signal_sort_value(row, sort_by="detected_at") > 0


def test_needs_refresh_accepts_naive_latest_detected_at():
    class _NaiveStore:
        def latest_detected_at(self, *, user_id: str = "default") -> datetime:
            return datetime.utcnow()

    service = object.__new__(IntelligenceRadarService)
    service._store = _NaiveStore()

    assert service.needs_refresh(user_id="default", max_age_minutes=45) is False
