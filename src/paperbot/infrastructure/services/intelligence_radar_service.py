from __future__ import annotations

from math import log1p
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import requests
from sqlalchemy import desc, select

from paperbot.infrastructure.api_clients.github_client import GitHubRadarClient
from paperbot.infrastructure.api_clients.x_client import XRecentSearchClient
from paperbot.infrastructure.connectors.hf_daily_papers_connector import (
    HFDailyPaperRecord,
    HFDailyPapersConnector,
)
from paperbot.infrastructure.services.subscription_service import SubscriptionService
from paperbot.infrastructure.stores.intelligence_store import IntelligenceStore
from paperbot.infrastructure.stores.models import PaperRepoModel
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_KEYWORDS = [
    "llm agents",
    "rag",
    "ai security",
    "long context",
]
DEFAULT_REDDIT_SUBREDDITS = ["MachineLearning", "LocalLLaMA", "OpenAI"]
DEFAULT_GITHUB_REPOS = [
    "langchain-ai/langchain",
    "microsoft/autogen",
    "huggingface/transformers",
]
USER_AGENT = "PaperBot/1.0 (+https://github.com/Color2333/PaperBot)"


@dataclass
class RadarProfile:
    keywords: List[str]
    scholar_names: List[str]
    watch_repos: List[str]
    subreddits: List[str]


@dataclass(frozen=True)
class RadarScorePolicy:
    metric_scale: float
    delta_scale: float
    freshness_hours: int
    source_confidence: float


_DEFAULT_SCORE_POLICY = RadarScorePolicy(
    metric_scale=10.0,
    delta_scale=5.0,
    freshness_hours=48,
    source_confidence=0.7,
)

_RADAR_SCORE_POLICIES: Dict[tuple[str, str], RadarScorePolicy] = {
    ("reddit", "keyword_spike"): RadarScorePolicy(14.0, 8.0, 24, 0.72),
    ("reddit", "comment_spike"): RadarScorePolicy(18.0, 10.0, 24, 0.75),
    ("github", "repo_activity"): RadarScorePolicy(12.0, 8.0, 48, 0.88),
    ("github", "repo_release"): RadarScorePolicy(1.0, 1.0, 24 * 30, 0.92),
    ("github", "repo_issue_heat"): RadarScorePolicy(12.0, 6.0, 24 * 7, 0.84),
    ("huggingface", "paper_buzz"): RadarScorePolicy(30.0, 12.0, 24 * 3, 0.8),
    ("twitter_x", "keyword_spike"): RadarScorePolicy(16.0, 8.0, 24, 0.62),
}


class IntelligenceRadarService:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._store = IntelligenceStore(self.db_url, auto_create_schema=False)
        self._research_store = SqlAlchemyResearchStore(self.db_url)
        self._provider = SessionProvider(self.db_url)
        self._hf_daily = HFDailyPapersConnector(cache_ttl_s=300.0)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._timeout_s = float(os.getenv("PAPERBOT_INTELLIGENCE_TIMEOUT_S", "15"))
        self._github_client = GitHubRadarClient(
            token=os.getenv("GITHUB_TOKEN") or None,
            session=self._session,
            timeout_s=self._timeout_s,
        )
        self._x_client = XRecentSearchClient(
            bearer_token=os.getenv("PAPERBOT_X_BEARER_TOKEN") or None,
            session=self._session,
            timeout_s=self._timeout_s,
        )

    def list_feed(
        self,
        *,
        user_id: str = "default",
        limit: int = 8,
        source: Optional[str] = None,
        keyword: Optional[str] = None,
        repo: Optional[str] = None,
        sort_by: str = "score",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        candidate_limit = max(int(limit), 50)
        rows = self._store.list_events(user_id=user_id, limit=candidate_limit, max_age_days=14)
        source_filter = str(source or "").strip().lower()
        keyword_filter = str(keyword or "").strip().lower()
        repo_filter = str(repo or "").strip().lower()

        filtered_rows: List[Dict[str, Any]] = []
        for row in rows:
            if source_filter and source_filter != str(row.get("source") or "").strip().lower():
                continue
            if keyword_filter and not _matches_signal_filter(
                keyword_filter,
                row.get("keyword_hits") or [],
                [row.get("title"), row.get("summary")],
            ):
                continue
            if repo_filter and not _matches_signal_filter(
                repo_filter,
                row.get("repo_matches") or [],
                [row.get("repo_full_name"), row.get("title"), row.get("summary")],
            ):
                continue
            row["match_reasons"] = self._build_match_reasons(row)
            filtered_rows.append(row)

        reverse = str(sort_order or "desc").strip().lower() != "asc"
        filtered_rows.sort(key=lambda row: _signal_sort_value(row, sort_by=sort_by), reverse=reverse)
        return filtered_rows[: max(1, int(limit))]

    def needs_refresh(self, *, user_id: str = "default", max_age_minutes: int = 45) -> bool:
        latest = _parse_datetime(self._store.latest_detected_at(user_id=user_id))
        if latest is None:
            return True
        return latest <= _utcnow() - timedelta(minutes=max(5, int(max_age_minutes)))

    def latest_refresh(self, *, user_id: str = "default") -> Optional[str]:
        latest = _parse_datetime(self._store.latest_detected_at(user_id=user_id))
        return latest.isoformat() if latest else None

    def build_profile(self, *, user_id: str = "default") -> RadarProfile:
        keywords: List[str] = []
        scholar_names: List[str] = []

        try:
            tracks = self._research_store.list_tracks(user_id=user_id, include_archived=False, limit=12)
        except Exception:
            tracks = []

        active_track = next((track for track in tracks if track.get("is_active")), None)
        track_pool = [active_track] if active_track else []
        track_pool.extend(track for track in tracks if track is not active_track)
        for track in track_pool:
            keywords.extend(track.get("keywords") or [])
            keywords.extend(track.get("methods") or [])

        try:
            subscription_service = SubscriptionService(
                config_path=os.getenv("PAPERBOT_SUBSCRIPTIONS_PATH") or None
            )
            for row in subscription_service.get_scholar_configs():
                name = str(row.get("name") or "").strip()
                if name:
                    scholar_names.append(name)
                keywords.extend(row.get("alert_keywords") or [])
                keywords.extend(row.get("keywords") or [])
        except Exception:
            pass

        fallback_keywords = _parse_csv_env(
            "PAPERBOT_INTELLIGENCE_KEYWORDS",
            ",".join(DEFAULT_KEYWORDS),
        )
        watch_repos = self._load_watch_repos()
        if not watch_repos:
            watch_repos = _parse_csv_env(
                "PAPERBOT_INTELLIGENCE_GITHUB_REPOS",
                ",".join(DEFAULT_GITHUB_REPOS),
            )
        subreddits = _parse_csv_env(
            "PAPERBOT_INTELLIGENCE_REDDIT_SUBREDDITS",
            ",".join(DEFAULT_REDDIT_SUBREDDITS),
        )

        merged_keywords = _dedupe_preserve_order(
            [keyword for keyword in keywords if str(keyword).strip()] + fallback_keywords
        )
        if not merged_keywords:
            merged_keywords = list(DEFAULT_KEYWORDS)

        return RadarProfile(
            keywords=merged_keywords[:6],
            scholar_names=_dedupe_preserve_order(scholar_names)[:12],
            watch_repos=_dedupe_preserve_order(watch_repos)[:5],
            subreddits=_dedupe_preserve_order(subreddits)[:8],
        )

    def refresh(self, *, user_id: str = "default") -> Dict[str, Any]:
        profile = self.build_profile(user_id=user_id)
        detected_at = _utcnow()
        events: List[Dict[str, Any]] = []
        events.extend(self._collect_reddit_events(user_id=user_id, profile=profile))
        events.extend(self._collect_reddit_comment_events(user_id=user_id, profile=profile))
        events.extend(self._collect_github_events(user_id=user_id, profile=profile))
        events.extend(self._collect_github_issue_events(user_id=user_id, profile=profile))
        events.extend(self._collect_hf_events(user_id=user_id, profile=profile))
        events.extend(self._collect_x_events(user_id=user_id, profile=profile))

        persisted: List[Dict[str, Any]] = []
        for event in events:
            row = self._store.upsert_event(
                user_id=user_id,
                external_id=event["external_id"],
                source=event["source"],
                source_label=event["source_label"],
                kind=event["kind"],
                title=event["title"],
                summary=event["summary"],
                url=event.get("url") or "",
                repo_full_name=event.get("repo_full_name") or "",
                author_name=event.get("author_name") or "",
                keyword_hits=event.get("keyword_hits") or [],
                author_matches=event.get("author_matches") or [],
                repo_matches=event.get("repo_matches") or [],
                metric_name=event.get("metric_name") or "",
                metric_value=int(event.get("metric_value") or 0),
                metric_delta=int(event.get("metric_delta") or 0),
                score=float(event.get("score") or 0.0),
                published_at=event.get("published_at"),
                detected_at=detected_at,
                payload=event.get("payload") or {},
            )
            row["match_reasons"] = self._build_match_reasons(row)
            persisted.append(row)

        return {
            "refreshed_at": detected_at.isoformat(),
            "count": len(persisted),
            "keywords": profile.keywords,
            "watch_repos": profile.watch_repos,
            "subreddits": profile.subreddits,
        }

    def _collect_reddit_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        allowed_subreddits = {item.lower() for item in profile.subreddits if item}
        results: List[Dict[str, Any]] = []
        max_workers = min(4, max(1, len(profile.keywords)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._fetch_reddit_keyword_summary, keyword, allowed_subreddits): keyword
                for keyword in profile.keywords[:4]
            }
            for future in as_completed(future_map):
                try:
                    row = future.result()
                except Exception:
                    continue
                if row is None:
                    continue

                keyword = row["keyword"]
                entry_titles = [entry["title"] for entry in row["entries"]]
                author_matches = _find_matches(profile.scholar_names, entry_titles)
                repo_matches = _find_matches(profile.watch_repos, entry_titles)
                external_id = f"reddit:keyword:{_slugify(keyword)}"
                previous = self._store.get_event(user_id=user_id, external_id=external_id)
                metric_value = len(row["entries"])
                previous_value = int(previous.get("metric_value") or 0) if previous else 0
                metric_delta = metric_value - previous_value
                subreddits = row["subreddits"]
                top_entry = row["entries"][0]
                score, breakdown = _score_signal(
                    source="reddit",
                    kind="keyword_spike",
                    metric_value=metric_value,
                    metric_delta=metric_delta,
                    published_at=row["latest_published_at"],
                    keyword_hits=[keyword],
                    author_matches=author_matches,
                    repo_matches=repo_matches,
                )
                summary = (
                    f"24h mentions: {metric_value} across {', '.join(f'r/{name}' for name in subreddits[:3])}. "
                    f"Top post: {top_entry['title']}."
                )
                results.append(
                    {
                        "external_id": external_id,
                        "source": "reddit",
                        "source_label": "Reddit Search",
                        "kind": "keyword_spike",
                        "title": f"Reddit spike: {keyword}",
                        "summary": summary,
                        "url": top_entry["link"],
                        "keyword_hits": [keyword],
                        "author_matches": author_matches,
                        "repo_matches": repo_matches,
                        "metric_name": "mentions/24h",
                        "metric_value": metric_value,
                        "metric_delta": metric_delta,
                        "score": score,
                        "published_at": row["latest_published_at"],
                        "payload": {
                            "subreddits": subreddits,
                            "top_post": top_entry,
                            "entries": row["entries"][:5],
                            "score_breakdown": breakdown,
                        },
                    }
                )
        return results

    def _collect_reddit_comment_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        allowed_subreddits = {item.lower() for item in profile.subreddits if item}
        results: List[Dict[str, Any]] = []
        for keyword in profile.keywords[:3]:
            try:
                row = self._fetch_reddit_keyword_summary(keyword, allowed_subreddits)
            except Exception:
                row = None
            if row is None:
                continue

            comments: List[Dict[str, Any]] = []
            for entry in row["entries"][:3]:
                post_id = _extract_reddit_post_id(entry.get("link"))
                if not post_id:
                    continue
                try:
                    comments.extend(self._fetch_reddit_comments(post_id))
                except Exception:
                    continue

            if not comments:
                continue

            comment_texts = [comment.get("body") or "" for comment in comments]
            author_matches = _find_matches(profile.scholar_names, comment_texts)
            repo_matches = _find_matches(profile.watch_repos, comment_texts)
            external_id = f"reddit:comments:{_slugify(keyword)}"
            previous = self._store.get_event(user_id=user_id, external_id=external_id)
            metric_value = len(comments)
            previous_value = int(previous.get("metric_value") or 0) if previous else 0
            metric_delta = metric_value - previous_value
            top_comment = max(comments, key=lambda comment: int(comment.get("score") or 0))
            score, breakdown = _score_signal(
                source="reddit",
                kind="comment_spike",
                metric_value=metric_value,
                metric_delta=metric_delta,
                published_at=top_comment.get("created_at"),
                keyword_hits=[keyword],
                author_matches=author_matches,
                repo_matches=repo_matches,
            )
            results.append(
                {
                    "external_id": external_id,
                    "source": "reddit",
                    "source_label": "Reddit Comments",
                    "kind": "comment_spike",
                    "title": f"Reddit comments: {keyword}",
                    "summary": (
                        f"24h comment hits: {metric_value}. "
                        f"Top comment: {_truncate_text(top_comment.get('body') or '', limit=110)}"
                    ),
                    "url": top_comment.get("link")
                    or (row["entries"][0].get("link") if row["entries"] else ""),
                    "keyword_hits": [keyword],
                    "author_matches": author_matches,
                    "repo_matches": repo_matches,
                    "metric_name": "comments/24h",
                    "metric_value": metric_value,
                    "metric_delta": metric_delta,
                    "score": score,
                    "published_at": top_comment.get("created_at"),
                    "payload": {
                        "comments": comments[:6],
                        "threads": row["entries"][:3],
                        "score_breakdown": breakdown,
                    },
                }
            )
        return results

    def _collect_github_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not profile.watch_repos:
            return results

        max_workers = min(4, max(1, len(profile.watch_repos) * 2))
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for repo in profile.watch_repos[:4]:
                tasks.append((repo, "commits", executor.submit(self._fetch_github_feed, repo, "commits")))
                tasks.append((repo, "releases", executor.submit(self._fetch_github_feed, repo, "releases")))

            grouped: Dict[str, Dict[str, Any]] = {}
            for repo, feed_type, future in tasks:
                try:
                    grouped.setdefault(repo, {})[feed_type] = future.result()
                except Exception:
                    grouped.setdefault(repo, {})[feed_type] = []

        for repo, payload in grouped.items():
            commits = payload.get("commits") or []
            recent_commits = [
                entry for entry in commits if _is_recent(entry.get("published_at"), hours=48)
            ]
            if recent_commits:
                external_id = f"github:{repo}:commits-48h"
                previous = self._store.get_event(user_id=user_id, external_id=external_id)
                metric_value = len(recent_commits)
                previous_value = int(previous.get("metric_value") or 0) if previous else 0
                metric_delta = metric_value - previous_value
                commit_titles = [entry.get("title") or "" for entry in recent_commits]
                keyword_hits = _find_matches(profile.keywords, [repo] + commit_titles)
                latest_commit = recent_commits[0]
                score, breakdown = _score_signal(
                    source="github",
                    kind="repo_activity",
                    metric_value=metric_value,
                    metric_delta=metric_delta,
                    published_at=latest_commit.get("published_at"),
                    keyword_hits=keyword_hits,
                    author_matches=[],
                    repo_matches=[repo],
                )
                results.append(
                    {
                        "external_id": external_id,
                        "source": "github",
                        "source_label": "GitHub Feed",
                        "kind": "repo_activity",
                        "title": f"GitHub activity: {repo}",
                        "summary": (
                            f"48h commits: {metric_value}. Latest: "
                            f"{latest_commit.get('title') or 'Untitled commit'}."
                        ),
                        "url": latest_commit.get("link") or f"https://github.com/{repo}",
                        "repo_full_name": repo,
                        "keyword_hits": keyword_hits,
                        "author_matches": [],
                        "repo_matches": [repo],
                        "metric_name": "commits/48h",
                        "metric_value": metric_value,
                        "metric_delta": metric_delta,
                        "score": score,
                        "published_at": latest_commit.get("published_at"),
                        "payload": {"commits": recent_commits[:5], "score_breakdown": breakdown},
                    }
                )

            releases = payload.get("releases") or []
            if releases:
                latest_release = releases[0]
                release_ts = latest_release.get("published_at")
                if _is_recent(release_ts, hours=24 * 30):
                    external_id = str(latest_release.get("id") or f"github:{repo}:release:latest")
                    previous = self._store.get_event(user_id=user_id, external_id=external_id)
                    keyword_hits = _find_matches(profile.keywords, [repo, latest_release.get("title") or ""])
                    score, breakdown = _score_signal(
                        source="github",
                        kind="repo_release",
                        metric_value=1,
                        metric_delta=0 if previous else 1,
                        published_at=release_ts,
                        keyword_hits=keyword_hits,
                        author_matches=[],
                        repo_matches=[repo],
                    )
                    results.append(
                        {
                            "external_id": external_id,
                            "source": "github",
                            "source_label": "GitHub Release",
                            "kind": "repo_release",
                            "title": f"{repo} release: {latest_release.get('title') or 'latest'}",
                            "summary": "New release detected for a watched repository with possible track relevance.",
                            "url": latest_release.get("link") or f"https://github.com/{repo}/releases",
                            "repo_full_name": repo,
                            "keyword_hits": keyword_hits,
                            "author_matches": [],
                            "repo_matches": [repo],
                            "metric_name": "release",
                            "metric_value": 1,
                            "metric_delta": 0 if previous else 1,
                            "score": score,
                            "published_at": release_ts,
                            "payload": {"release": latest_release, "score_breakdown": breakdown},
                        }
                    )
        return results

    def _collect_github_issue_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not profile.watch_repos:
            return results

        for repo in profile.watch_repos[:4]:
            try:
                issues = self._github_client.list_recent_issues(repo, per_page=20)
            except Exception:
                issues = []
            recent_issues = [issue for issue in issues if _is_recent(issue.get("updated_at"), hours=24 * 7)]
            if not recent_issues:
                continue

            issue_texts = [
                f"{issue.get('title') or ''}\n{issue.get('body') or ''}\n{issue.get('author') or ''}"
                for issue in recent_issues
            ]
            keyword_hits = _find_matches(profile.keywords, [repo] + issue_texts)
            author_matches = _find_matches(profile.scholar_names, issue_texts)
            external_id = f"github:{repo}:issues-7d"
            previous = self._store.get_event(user_id=user_id, external_id=external_id)
            metric_value = len(recent_issues)
            previous_value = int(previous.get("metric_value") or 0) if previous else 0
            metric_delta = metric_value - previous_value
            top_issue = recent_issues[0]
            score, breakdown = _score_signal(
                source="github",
                kind="repo_issue_heat",
                metric_value=metric_value,
                metric_delta=metric_delta,
                published_at=top_issue.get("updated_at"),
                keyword_hits=keyword_hits,
                author_matches=author_matches,
                repo_matches=[repo],
            )
            results.append(
                {
                    "external_id": external_id,
                    "source": "github",
                    "source_label": "GitHub Issues",
                    "kind": "repo_issue_heat",
                    "title": f"GitHub issues: {repo}",
                    "summary": (
                        f"7d issues: {metric_value}. "
                        f"Top thread: {top_issue.get('title') or 'Untitled issue'}."
                    ),
                    "url": top_issue.get("html_url") or f"https://github.com/{repo}/issues",
                    "repo_full_name": repo,
                    "keyword_hits": keyword_hits,
                    "author_matches": author_matches,
                    "repo_matches": [repo],
                    "metric_name": "issues/7d",
                    "metric_value": metric_value,
                    "metric_delta": metric_delta,
                    "score": score,
                    "published_at": top_issue.get("updated_at"),
                    "payload": {"issues": recent_issues[:6], "score_breakdown": breakdown},
                }
            )
        return results

    def _collect_hf_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        max_workers = min(4, max(1, len(profile.keywords)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._hf_daily.search, query=keyword, max_results=2, page_size=50, max_pages=1): keyword
                for keyword in profile.keywords[:4]
            }
            for future in as_completed(future_map):
                keyword = future_map[future]
                try:
                    records = future.result()
                except Exception:
                    records = []
                for record in records:
                    bucket = aggregated.setdefault(
                        record.paper_id,
                        {
                            "record": record,
                            "keyword_hits": [],
                        },
                    )
                    bucket["keyword_hits"].append(keyword)

        results: List[Dict[str, Any]] = []
        for paper_id, payload in aggregated.items():
            record = payload["record"]
            keyword_hits = _dedupe_preserve_order(payload["keyword_hits"])
            external_id = f"hf_daily:{paper_id}"
            previous = self._store.get_event(user_id=user_id, external_id=external_id)
            metric_value = int(record.upvotes or 0)
            previous_value = int(previous.get("metric_value") or 0) if previous else 0
            metric_delta = metric_value - previous_value
            author_matches = _match_authors(profile.scholar_names, record)
            score, breakdown = _score_signal(
                source="huggingface",
                kind="paper_buzz",
                metric_value=metric_value,
                metric_delta=metric_delta,
                published_at=_parse_datetime(record.submitted_on_daily_at)
                or _parse_datetime(record.published_at),
                keyword_hits=keyword_hits,
                author_matches=author_matches,
                repo_matches=[],
            )
            author_line = ", ".join(record.authors[:2]) if record.authors else "unknown authors"
            results.append(
                {
                    "external_id": external_id,
                    "source": "huggingface",
                    "source_label": "HF Daily Papers",
                    "kind": "paper_buzz",
                    "title": record.title,
                    "summary": f"HF Daily Papers upvotes: {metric_value}. Authors: {author_line}.",
                    "url": record.paper_url or record.external_url,
                    "author_name": record.authors[0] if record.authors else "",
                    "keyword_hits": keyword_hits,
                    "author_matches": author_matches,
                    "repo_matches": [],
                    "metric_name": "upvotes",
                    "metric_value": metric_value,
                    "metric_delta": metric_delta,
                    "score": score,
                    "published_at": _parse_datetime(record.submitted_on_daily_at) or _parse_datetime(record.published_at),
                    "payload": {
                        "authors": record.authors,
                        "external_url": record.external_url,
                        "pdf_url": record.pdf_url,
                        "summary": record.summary,
                        "ai_keywords": record.ai_keywords,
                        "score_breakdown": breakdown,
                    },
                }
            )
        return results

    def _collect_x_events(
        self,
        *,
        user_id: str,
        profile: RadarProfile,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not self._x_client.enabled:
            return results

        max_workers = min(4, max(1, len(profile.keywords)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._x_client.search_recent, keyword, 10): keyword
                for keyword in profile.keywords[:4]
            }
            for future in as_completed(future_map):
                keyword = future_map[future]
                try:
                    tweets = future.result()
                except Exception:
                    tweets = []
                if not tweets:
                    continue

                texts = [
                    f"{tweet.get('text') or ''}\n{tweet.get('author_name') or ''}\n{tweet.get('username') or ''}"
                    for tweet in tweets
                ]
                author_matches = _find_matches(profile.scholar_names, texts)
                repo_matches = _find_matches(profile.watch_repos, texts)
                external_id = f"twitter_x:keyword:{_slugify(keyword)}"
                previous = self._store.get_event(user_id=user_id, external_id=external_id)
                metric_value = len(tweets)
                previous_value = int(previous.get("metric_value") or 0) if previous else 0
                metric_delta = metric_value - previous_value
                latest = tweets[0]
                score, breakdown = _score_signal(
                    source="twitter_x",
                    kind="keyword_spike",
                    metric_value=metric_value,
                    metric_delta=metric_delta,
                    published_at=latest.get("created_at"),
                    keyword_hits=[keyword],
                    author_matches=author_matches,
                    repo_matches=repo_matches,
                )
                results.append(
                    {
                        "external_id": external_id,
                        "source": "twitter_x",
                        "source_label": "X Recent Search",
                        "kind": "keyword_spike",
                        "title": f"X spike: {keyword}",
                        "summary": (
                            f"24h posts: {metric_value}. "
                            f"Top post by @{latest.get('username') or 'unknown'}: "
                            f"{_truncate_text(latest.get('text') or '', limit=110)}"
                        ),
                        "url": latest.get("url") or "",
                        "author_name": latest.get("author_name") or "",
                        "keyword_hits": [keyword],
                        "author_matches": author_matches,
                        "repo_matches": repo_matches,
                        "metric_name": "posts/24h",
                        "metric_value": metric_value,
                        "metric_delta": metric_delta,
                        "score": score,
                        "published_at": latest.get("created_at"),
                        "payload": {"posts": tweets[:6], "score_breakdown": breakdown},
                    }
                )
        return results

    def _fetch_reddit_keyword_summary(
        self,
        keyword: str,
        allowed_subreddits: set[str],
    ) -> Optional[Dict[str, Any]]:
        url = f"https://www.reddit.com/search.rss?q={quote_plus(keyword)}&sort=new&t=day"
        text = self._fetch_text(url)
        if not text:
            return None
        entries = self._parse_atom_entries(text)
        normalized_entries = []
        seen_links = set()
        for entry in entries:
            subreddit = _extract_subreddit(entry)
            if allowed_subreddits and subreddit and subreddit.lower() not in allowed_subreddits:
                continue
            link = entry.get("link") or ""
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            normalized_entries.append(
                {
                    "title": entry.get("title") or "Untitled post",
                    "link": link,
                    "subreddit": subreddit or "reddit.com",
                    "published_at": entry.get("published_at"),
                }
            )
        if not normalized_entries:
            return None
        subreddits = _dedupe_preserve_order(
            [entry["subreddit"] for entry in normalized_entries if entry.get("subreddit")]
        )
        return {
            "keyword": keyword,
            "entries": normalized_entries[:8],
            "subreddits": subreddits,
            "latest_published_at": normalized_entries[0].get("published_at"),
        }

    def _fetch_github_feed(self, repo: str, feed_type: str) -> List[Dict[str, Any]]:
        if feed_type == "releases":
            url = f"https://github.com/{repo}/releases.atom"
        else:
            url = f"https://github.com/{repo}/commits.atom"
        text = self._fetch_text(url)
        if not text:
            return []
        return self._parse_atom_entries(text)

    def _fetch_text(self, url: str) -> str:
        response = self._session.get(url, timeout=self._timeout_s)
        response.raise_for_status()
        return response.text

    def _fetch_reddit_comments(self, post_id: str) -> List[Dict[str, Any]]:
        response = self._session.get(
            f"https://www.reddit.com/comments/{post_id}.json",
            params={"limit": 20, "depth": 2, "sort": "top"},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return []

        post_children = ((payload[0] or {}).get("data") or {}).get("children") or []
        default_link = ""
        if post_children:
            post_data = (post_children[0] or {}).get("data") or {}
            permalink = str(post_data.get("permalink") or "").strip()
            if permalink:
                default_link = f"https://www.reddit.com{permalink}"

        comment_children = ((payload[1] or {}).get("data") or {}).get("children") or []
        return _flatten_reddit_comment_nodes(comment_children, default_link=default_link)[:30]

    def _parse_atom_entries(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        entries: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            published_raw = (
                entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
                or entry.findtext("atom:published", default="", namespaces=ATOM_NS)
                or ""
            )
            published_at = _parse_datetime(published_raw)
            author = (
                entry.findtext("atom:author/atom:name", default="", namespaces=ATOM_NS) or ""
            ).strip()
            summary = (
                entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
                or entry.findtext("atom:content", default="", namespaces=ATOM_NS)
                or ""
            )
            link = ""
            for link_tag in entry.findall("atom:link", ATOM_NS):
                href = str(link_tag.attrib.get("href") or "").strip()
                rel = str(link_tag.attrib.get("rel") or "alternate").strip().lower()
                if href and rel in {"alternate", ""}:
                    link = href
                    break
            categories = [
                str(tag.attrib.get("term") or "").strip() for tag in entry.findall("atom:category", ATOM_NS)
            ]
            entries.append(
                {
                    "id": entry_id or link or title,
                    "title": title,
                    "link": link,
                    "author": author,
                    "summary": _strip_html(summary),
                    "published_at": published_at,
                    "categories": [value for value in categories if value],
                }
            )
        return entries

    def _load_watch_repos(self) -> List[str]:
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(PaperRepoModel)
                    .where(PaperRepoModel.full_name != "")
                    .order_by(desc(PaperRepoModel.stars), desc(PaperRepoModel.synced_at))
                    .limit(5)
                )
                .scalars()
                .all()
            )
            return _dedupe_preserve_order(
                [str(row.full_name or "").strip() for row in rows if str(row.full_name or "").strip()]
            )

    def _build_match_reasons(self, row: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        for keyword in row.get("keyword_hits") or []:
            reasons.append(f"keyword: {keyword}")
        if int(row.get("metric_delta") or 0) > 0:
            reasons.append(f"delta: +{int(row.get('metric_delta') or 0)}")
        for author in row.get("author_matches") or []:
            reasons.append(f"author: {author}")
        for repo in row.get("repo_matches") or []:
            reasons.append(f"repo: {repo}")
        metric_name = str(row.get("metric_name") or "").strip()
        metric_value = int(row.get("metric_value") or 0)
        if metric_name and not reasons:
            reasons.append(f"{metric_name}: {metric_value}")
        return reasons[:4]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _score_signal(
    *,
    source: str,
    kind: str,
    metric_value: int,
    metric_delta: int,
    published_at: Any,
    keyword_hits: List[str],
    author_matches: List[str],
    repo_matches: List[str],
) -> tuple[float, Dict[str, float]]:
    policy = _RADAR_SCORE_POLICIES.get((source, kind), _DEFAULT_SCORE_POLICY)
    safe_metric_value = max(int(metric_value or 0), 0)
    safe_metric_delta = max(int(metric_delta or 0), 0)

    metric_component = min(
        26.0,
        (log1p(safe_metric_value) / log1p(max(policy.metric_scale, 1.0) + 1.0)) * 26.0,
    )
    delta_component = min(
        22.0,
        (log1p(safe_metric_delta) / log1p(max(policy.delta_scale, 1.0) + 1.0)) * 22.0,
    )

    freshness_component = 0.0
    published_dt = _parse_datetime(published_at)
    if published_dt is not None:
        age_hours = max((_utcnow() - published_dt).total_seconds() / 3600.0, 0.0)
        freshness_ratio = max(0.0, 1.0 - (age_hours / max(float(policy.freshness_hours), 1.0)))
        freshness_component = 12.0 * freshness_ratio

    keyword_component = min(10.0, len(keyword_hits or []) * 4.0)
    author_component = min(12.0, len(author_matches or []) * 6.0)
    repo_component = min(10.0, len(repo_matches or []) * 5.0)
    confidence_component = min(max(policy.source_confidence, 0.0), 1.0) * 8.0

    total = min(
        100.0,
        metric_component
        + delta_component
        + freshness_component
        + keyword_component
        + author_component
        + repo_component
        + confidence_component,
    )
    breakdown = {
        "metric": round(metric_component, 2),
        "delta": round(delta_component, 2),
        "freshness": round(freshness_component, 2),
        "keyword": round(keyword_component, 2),
        "author": round(author_component, 2),
        "repo": round(repo_component, 2),
        "source_confidence": round(confidence_component, 2),
        "total": round(total, 2),
    }
    return round(total, 2), breakdown


def _matches_signal_filter(
    filter_value: str,
    primary_values: Iterable[str],
    fallback_texts: Iterable[Any],
) -> bool:
    needle = str(filter_value or "").strip().lower()
    if not needle:
        return True

    for value in primary_values:
        cleaned = str(value or "").strip().lower()
        if cleaned and (needle in cleaned or cleaned in needle):
            return True

    combined = "\n".join(str(value or "") for value in fallback_texts).lower()
    return needle in combined


def _signal_sort_value(row: Dict[str, Any], *, sort_by: str) -> Any:
    normalized_sort = str(sort_by or "score").strip().lower()
    if normalized_sort == "delta":
        return int(row.get("metric_delta") or 0)
    if normalized_sort == "score":
        return float(row.get("score") or 0.0)
    if normalized_sort == "source":
        return str(row.get("source_label") or row.get("source") or "").lower()
    if normalized_sort == "keyword":
        keyword_hits = row.get("keyword_hits") or []
        if keyword_hits:
            return str(keyword_hits[0] or "").lower()
        return str(row.get("title") or "").lower()
    if normalized_sort == "repo":
        repo_name = str(row.get("repo_full_name") or "").strip()
        if repo_name:
            return repo_name.lower()
        repo_matches = row.get("repo_matches") or []
        if repo_matches:
            return str(repo_matches[0] or "").lower()
        return ""
    if normalized_sort in {"published_at", "freshness"}:
        published_at = _parse_datetime(row.get("published_at") or row.get("detected_at"))
        return published_at.timestamp() if published_at else 0.0
    if normalized_sort == "detected_at":
        detected_at = _parse_datetime(row.get("detected_at"))
        return detected_at.timestamp() if detected_at else 0.0
    return float(row.get("score") or 0.0)


def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    results: List[str] = []
    seen = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_recent(value: Any, *, hours: int) -> bool:
    parsed = _parse_datetime(value)
    if parsed is None:
        return False
    return parsed >= _utcnow() - timedelta(hours=max(1, int(hours)))


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-") or "signal"


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_subreddit(entry: Dict[str, Any]) -> str:
    for category in entry.get("categories") or []:
        cleaned = str(category or "").strip().replace("r/", "")
        if cleaned and cleaned.lower() != "reddit.com":
            return cleaned
    link = str(entry.get("link") or "")
    match = re.search(r"reddit\.com/r/([^/]+)/", link, re.IGNORECASE)
    return match.group(1) if match else ""


def _find_matches(candidates: Iterable[str], texts: Iterable[str]) -> List[str]:
    haystack = "\n".join(str(text or "") for text in texts).lower()
    results: List[str] = []
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if not cleaned:
            continue
        if cleaned.lower() in haystack:
            results.append(cleaned)
    return _dedupe_preserve_order(results)


def _match_authors(candidates: Iterable[str], record: HFDailyPaperRecord) -> List[str]:
    haystack = [record.title, record.summary] + list(record.authors or [])
    return _find_matches(candidates, haystack)


def _extract_reddit_post_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"/comments/([a-z0-9]+)/", text, re.IGNORECASE)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-z0-9]{4,12}", text, re.IGNORECASE):
        return text.lower()
    return ""


def _flatten_reddit_comment_nodes(
    nodes: List[Dict[str, Any]],
    *,
    default_link: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node in nodes:
        if str((node or {}).get("kind") or "") != "t1":
            continue

        data = (node or {}).get("data") or {}
        body = _strip_html(data.get("body_html") or data.get("body") or "")
        created_raw = data.get("created_utc")
        if isinstance(created_raw, (int, float)):
            created_at = datetime.fromtimestamp(created_raw, tz=timezone.utc)
        else:
            created_at = _parse_datetime(created_raw)
        permalink = str(data.get("permalink") or "").strip()
        comment_link = f"https://www.reddit.com{permalink}" if permalink else default_link

        if body and created_at and _is_recent(created_at, hours=24):
            results.append(
                {
                    "id": str(data.get("id") or ""),
                    "body": body,
                    "author": str(data.get("author") or ""),
                    "score": int(data.get("score") or 0),
                    "created_at": created_at.isoformat(),
                    "link": comment_link,
                }
            )

        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_nodes = ((replies.get("data") or {}).get("children") or [])
            results.extend(_flatten_reddit_comment_nodes(reply_nodes, default_link=default_link))
    return results


def _truncate_text(value: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return f"{cleaned[: limit - 3].rstrip()}..."
