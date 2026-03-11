from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class XRecentSearchClient:
    def __init__(
        self,
        *,
        bearer_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout_s: float = 15.0,
    ):
        self._bearer_token = str(bearer_token or "").strip()
        self._session = session or requests.Session()
        self._timeout_s = float(timeout_s)

    @property
    def enabled(self) -> bool:
        return bool(self._bearer_token)

    def search_recent(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = str(keyword or "").strip()
        if not self.enabled or not query:
            return []

        params = {
            "query": f"({query}) -is:retweet",
            "max_results": max(10, min(int(limit or 10), 100)),
            "expansions": "author_id",
            "tweet.fields": "created_at,public_metrics,text,author_id,lang",
            "user.fields": "name,username",
        }
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
        }

        try:
            response = self._session.get(
                "https://api.x.com/2/tweets/search/recent",
                params=params,
                headers=headers,
                timeout=self._timeout_s,
            )
            if response.status_code in {401, 403, 404, 429}:
                return []
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        data = payload.get("data") or []
        includes = payload.get("includes") or {}
        users = {
            str(user.get("id") or ""): user
            for user in includes.get("users") or []
            if isinstance(user, dict)
        }

        results: List[Dict[str, Any]] = []
        for tweet in data:
            if not isinstance(tweet, dict):
                continue
            author = users.get(str(tweet.get("author_id") or ""), {})
            username = str(author.get("username") or "").strip()
            tweet_id = str(tweet.get("id") or "").strip()
            url = f"https://x.com/{username}/status/{tweet_id}" if username and tweet_id else ""
            results.append(
                {
                    "id": tweet_id,
                    "text": str(tweet.get("text") or ""),
                    "created_at": tweet.get("created_at"),
                    "author_id": str(tweet.get("author_id") or ""),
                    "author_name": str(author.get("name") or ""),
                    "username": username,
                    "url": url,
                    "public_metrics": dict(tweet.get("public_metrics") or {}),
                }
            )
        return results