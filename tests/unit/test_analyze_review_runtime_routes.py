from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from paperbot.api import main as api_main


def _parse_sse_payloads(raw_text: str):
    payloads = []
    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


@dataclass
class _AnalyzeResult:
    summary: str
    key_contributions: list[str]
    methodology: str
    strengths: list[str]
    weaknesses: list[str]


@dataclass
class _ReviewResult:
    summary: str
    contributions: list[str]
    strengths: list[str]
    weaknesses: list[str]
    novelty_score: float
    decision: str


class _FakeResearchAgent:
    def __init__(self, _config):
        pass

    async def analyze_paper(self, *, title: str, abstract: str):
        return _AnalyzeResult(
            summary=f"summary::{title}",
            key_contributions=["k1"],
            methodology=f"m::{abstract[:8]}",
            strengths=["s1"],
            weaknesses=["w1"],
        )


class _FakeReviewerAgent:
    def __init__(self, _config):
        pass

    async def review(self, *, title: str, abstract: str):
        return _ReviewResult(
            summary=f"review::{title}",
            contributions=["c1"],
            strengths=["s1"],
            weaknesses=["w1"],
            novelty_score=4.2,
            decision="accept",
        )


async def _request(path: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_main.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(path, json=payload)


@pytest.mark.asyncio
async def test_analyze_route_uses_runtime_contract(monkeypatch):
    import paperbot.agents.research as research_pkg

    monkeypatch.setattr(research_pkg, "ResearchAgent", _FakeResearchAgent)

    resp = await _request(
        "/api/analyze",
        {"title": "UniICL", "abstract": "context compression"},
    )

    assert resp.status_code == 200
    payloads = _parse_sse_payloads(resp.text)
    assert payloads
    assert payloads[-1]["type"] == "result"
    assert payloads[-1]["data"]["summary"] == "summary::UniICL"
    assert payloads[-1]["envelope"]["workflow"] == "analyze"
    assert payloads[-1]["envelope"]["trace_id"]


@pytest.mark.asyncio
async def test_review_route_uses_runtime_contract(monkeypatch):
    import paperbot.agents.review as review_pkg

    monkeypatch.setattr(review_pkg, "ReviewerAgent", _FakeReviewerAgent)

    resp = await _request(
        "/api/review",
        {"title": "VL-Cache", "abstract": "kv cache"},
    )

    assert resp.status_code == 200
    payloads = _parse_sse_payloads(resp.text)
    assert payloads
    assert payloads[-1]["type"] == "result"
    assert payloads[-1]["data"]["recommendation"] == "accept"
    assert payloads[-1]["envelope"]["workflow"] == "review"
    assert payloads[-1]["envelope"]["trace_id"]
