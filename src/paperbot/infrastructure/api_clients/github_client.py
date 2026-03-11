from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class GitHubRadarClient:
    def __init__(
        self,
        *,
        token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout_s: float = 15.0,
    ):
        self._token = str(token or "").strip()
        self._session = session or requests.Session()
        self._timeout_s = float(timeout_s)

    def list_recent_issues(self, repo: str, *, per_page: int = 20) -> List[Dict[str, Any]]:
        repo_name = str(repo or "").strip()
        if not repo_name or "/" not in repo_name:
            return []

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            response = self._session.get(
                f"https://api.github.com/repos/{repo_name}/issues",
                params={
                    "state": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": max(1, min(int(per_page), 100)),
                },
                headers=headers,
                timeout=self._timeout_s,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        if not isinstance(payload, list):
            return []

        issues: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("pull_request"):
                continue
            user = item.get("user") or {}
            issues.append(
                {
                    "id": str(item.get("id") or ""),
                    "number": int(item.get("number") or 0),
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("body") or ""),
                    "author": str(user.get("login") or ""),
                    "html_url": str(item.get("html_url") or ""),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "comments": int(item.get("comments") or 0),
                }
            )
        return issues