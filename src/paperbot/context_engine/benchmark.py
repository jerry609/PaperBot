from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig
from paperbot.context_engine.track_router import TrackRouter, TrackRouterConfig

_TOKEN_RX = re.compile(r"[a-zA-Z0-9_+.-]+")


@dataclass
class ContextBenchmarkCase:
    case_id: str
    query: str
    query_type: str = "generic"
    stage: str = "survey"
    user_id: str = "bench-user"
    active_track_id: Optional[int] = None
    track_id: Optional[int] = None
    include_cross_track: bool = False
    paper_id: Optional[str] = None
    context_token_budget: Optional[int] = None
    expected_layers: Dict[str, bool] = field(default_factory=dict)
    expected_token_guard: bool = False
    expected_router_track_id: Optional[int] = None
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    global_memories: List[Dict[str, Any]] = field(default_factory=list)
    track_memories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    paper_memories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tasks_by_track: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    milestones_by_track: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class _FixtureResearchStore:
    def __init__(self, case: ContextBenchmarkCase):
        self._tracks = [copy.deepcopy(track) for track in case.tracks]
        self._tracks_by_id = {int(track["id"]): copy.deepcopy(track) for track in case.tracks}
        self._tasks_by_track = {
            str(track_id): [copy.deepcopy(item) for item in items]
            for track_id, items in case.tasks_by_track.items()
        }
        self._milestones_by_track = {
            str(track_id): [copy.deepcopy(item) for item in items]
            for track_id, items in case.milestones_by_track.items()
        }
        self._active_track_id = int(case.active_track_id) if case.active_track_id else None
        self._context_run_id = 0

    def get_active_track(self, *, user_id: str):
        del user_id
        if self._active_track_id is None:
            return None
        return copy.deepcopy(self._tracks_by_id.get(self._active_track_id))

    def get_track(self, *, user_id: str, track_id: int):
        del user_id
        return copy.deepcopy(self._tracks_by_id.get(int(track_id)))

    def list_tracks(self, *, user_id: str, include_archived: bool = False, limit: int = 50):
        del user_id, include_archived
        return [copy.deepcopy(track) for track in self._tracks[:limit]]

    def list_tasks(self, *, user_id: str, track_id: int, limit: int):
        del user_id
        return [copy.deepcopy(item) for item in self._tasks_by_track.get(str(track_id), [])[:limit]]

    def list_milestones(self, *, user_id: str, track_id: int, limit: int):
        del user_id
        items = self._milestones_by_track.get(str(track_id), [])[:limit]
        return [copy.deepcopy(item) for item in items]

    def list_paper_feedback_ids(self, *, user_id: str, track_id: int, action: str):
        del user_id, track_id, action
        return set()

    def list_paper_feedback(self, *, user_id: str, track_id: int, limit: int):
        del user_id, track_id, limit
        return []

    def create_context_run(self, **kwargs):
        del kwargs
        self._context_run_id += 1
        return {"id": self._context_run_id}

    def get_track_embedding(self, *, track_id: int, model: str):
        del track_id, model
        return None

    def upsert_track_embedding(
        self,
        *,
        track_id: int,
        model: str,
        profile_text: str,
        embedding: Sequence[float],
    ) -> None:
        del track_id, model, profile_text, embedding
        return None


class _FixtureMemoryStore:
    def __init__(self, case: ContextBenchmarkCase):
        self._global_memories = [copy.deepcopy(item) for item in case.global_memories]
        self._track_memories = {
            str(scope_id): [copy.deepcopy(item) for item in items]
            for scope_id, items in case.track_memories.items()
        }
        self._paper_memories = {
            str(scope_id): [copy.deepcopy(item) for item in items]
            for scope_id, items in case.paper_memories.items()
        }

    def list_memories(
        self,
        *,
        user_id: str,
        limit: int,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_pending: bool = False,
        include_deleted: bool = False,
    ):
        del user_id, include_pending, include_deleted
        if scope_type == "global":
            items = self._global_memories
        elif scope_type == "track":
            items = self._track_memories.get(str(scope_id or ""), [])
        elif scope_type == "paper":
            items = self._paper_memories.get(str(scope_id or ""), [])
        else:
            items = []
        return [copy.deepcopy(item) for item in items[:limit]]

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        min_score: float = 0.0,
        candidate_multiplier: int = 4,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        half_life_days: float = 30.0,
    ):
        del user_id, candidate_multiplier, mmr_enabled, mmr_lambda, half_life_days
        if scope_type == "track":
            items = self._track_memories.get(str(scope_id or ""), [])
        elif scope_type == "paper":
            items = self._paper_memories.get(str(scope_id or ""), [])
        elif scope_type == "global":
            items = self._global_memories
        else:
            items = []
        return self._rank_items(items, query=query, limit=limit, min_score=min_score)

    def search_memories_batch(
        self,
        *,
        user_id: str,
        query: str,
        scope_ids: Sequence[str],
        scope_type: str,
        limit_per_scope: int,
        min_score: float = 0.0,
        candidate_multiplier: int = 4,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        half_life_days: float = 30.0,
    ):
        del user_id, candidate_multiplier, mmr_enabled, mmr_lambda, half_life_days
        if scope_type != "track":
            return {}
        batch: Dict[str, List[Dict[str, Any]]] = {}
        for scope_id in scope_ids:
            hits = self._rank_items(
                self._track_memories.get(str(scope_id), []),
                query=query,
                limit=limit_per_scope,
                min_score=min_score,
            )
            if hits:
                batch[str(scope_id)] = hits
        return batch

    def touch_usage(self, *, item_ids: Sequence[int], actor_id: str):
        del item_ids, actor_id
        return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RX.findall(text or "") if token.strip()}

    def _rank_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        query: str,
        limit: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        ranked: List[Dict[str, Any]] = []
        for original in items:
            item = copy.deepcopy(original)
            text = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("content") or ""),
                    " ".join(str(tag) for tag in (item.get("tags") or []) if str(tag)),
                ]
            )
            overlap = len(query_tokens & self._tokenize(text))
            base_score = float(item.get("score_bias") or 0.0)
            score = base_score + float(overlap)
            if score < float(min_score) or score <= 0.0:
                continue
            item["score"] = score
            ranked.append(item)

        ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), int(item.get("id") or 0)))
        return ranked[:limit]


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if isinstance(payload, dict):
        rows = payload.get("cases") or []
    else:
        rows = payload
    return list(rows)


def load_context_benchmark_cases(path: str | Path) -> List[ContextBenchmarkCase]:
    rows = _read_rows(Path(path))
    cases: List[ContextBenchmarkCase] = []
    for row in rows:
        expected = row.get("expected") or {}
        state = row.get("state") or {}
        cases.append(
            ContextBenchmarkCase(
                case_id=str(row.get("case_id") or "").strip(),
                query=str(row.get("query") or "").strip(),
                query_type=str(row.get("query_type") or "generic").strip(),
                stage=str(row.get("stage") or "survey").strip(),
                user_id=str(row.get("user_id") or "bench-user").strip(),
                active_track_id=(
                    int(row["active_track_id"])
                    if row.get("active_track_id") not in (None, "")
                    else None
                ),
                track_id=int(row["track_id"]) if row.get("track_id") not in (None, "") else None,
                include_cross_track=bool(row.get("include_cross_track", False)),
                paper_id=str(row.get("paper_id") or "").strip() or None,
                context_token_budget=(
                    int(row["context_token_budget"])
                    if row.get("context_token_budget") not in (None, "")
                    else None
                ),
                expected_layers={
                    "layer0_profile": bool((expected.get("layers") or {}).get("layer0_profile")),
                    "layer1_track": bool((expected.get("layers") or {}).get("layer1_track")),
                    "layer2_query": bool((expected.get("layers") or {}).get("layer2_query")),
                    "layer3_paper": bool((expected.get("layers") or {}).get("layer3_paper")),
                },
                expected_token_guard=bool(expected.get("token_guard", False)),
                expected_router_track_id=(
                    int(expected["router_track_id"])
                    if expected.get("router_track_id") not in (None, "")
                    else None
                ),
                tracks=[copy.deepcopy(track) for track in (state.get("tracks") or [])],
                global_memories=[
                    copy.deepcopy(item) for item in (state.get("global_memories") or [])
                ],
                track_memories={
                    str(track_id): [copy.deepcopy(item) for item in items]
                    for track_id, items in (state.get("track_memories") or {}).items()
                },
                paper_memories={
                    str(scope_id): [copy.deepcopy(item) for item in items]
                    for scope_id, items in (state.get("paper_memories") or {}).items()
                },
                tasks_by_track={
                    str(track_id): [copy.deepcopy(item) for item in items]
                    for track_id, items in (state.get("tasks_by_track") or {}).items()
                },
                milestones_by_track={
                    str(track_id): [copy.deepcopy(item) for item in items]
                    for track_id, items in (state.get("milestones_by_track") or {}).items()
                },
            )
        )
    return cases


def _layer_presence(pack: Dict[str, Any]) -> Dict[str, bool]:
    progress = pack.get("progress_state") or {}
    return {
        "layer0_profile": bool(pack.get("user_prefs") or []),
        "layer1_track": bool((progress.get("tasks") or []) or (progress.get("milestones") or [])),
        "layer2_query": bool(
            (pack.get("relevant_memories") or []) or (pack.get("cross_track_memories") or [])
        ),
        "layer3_paper": bool(pack.get("paper_memories") or []),
    }


def _precision_recall(
    expected_layers: Dict[str, bool],
    actual_layers: Dict[str, bool],
) -> Dict[str, float]:
    expected_positive = {name for name, enabled in expected_layers.items() if enabled}
    actual_positive = {name for name, enabled in actual_layers.items() if enabled}
    if not expected_positive and not actual_positive:
        return {"precision": 1.0, "recall": 1.0}
    if not actual_positive:
        return {"precision": 0.0, "recall": 0.0}
    if not expected_positive:
        return {"precision": 0.0, "recall": 1.0}

    tp = len(expected_positive & actual_positive)
    return {
        "precision": tp / float(len(actual_positive)),
        "recall": tp / float(len(expected_positive)),
    }


def evaluate_context_case(case: ContextBenchmarkCase, pack: Dict[str, Any]) -> Dict[str, Any]:
    actual_layers = _layer_presence(pack)
    layer_scores = _precision_recall(case.expected_layers, actual_layers)
    routing = pack.get("routing") or {}
    suggestion = routing.get("suggestion") or {}
    suggested_track_id = suggestion.get("track_id")
    token_guard_enabled = bool((routing.get("token_guard") or {}).get("enabled"))

    router_evaluable = case.expected_router_track_id is not None
    router_covered = bool(router_evaluable and suggested_track_id is not None)
    router_correct = bool(
        router_evaluable and int(suggested_track_id or 0) == int(case.expected_router_track_id or 0)
    )

    return {
        "case_id": case.case_id,
        "query": case.query,
        "query_type": case.query_type,
        "stage": case.stage,
        "expected_layers": dict(case.expected_layers),
        "actual_layers": actual_layers,
        "layer_precision": float(layer_scores["precision"]),
        "layer_recall": float(layer_scores["recall"]),
        "token_guard_expected": bool(case.expected_token_guard),
        "token_guard_enabled": token_guard_enabled,
        "token_guard_correct": float(token_guard_enabled == case.expected_token_guard),
        "router_evaluable": router_evaluable,
        "router_expected_track_id": case.expected_router_track_id,
        "router_suggested_track_id": int(suggested_track_id) if suggested_track_id else None,
        "router_covered": float(router_covered) if router_evaluable else None,
        "router_correct": float(router_correct) if router_evaluable else None,
        "context_layers": dict(pack.get("context_layers") or {}),
    }


def _aggregate_rows(case_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not case_results:
        return {
            "case_count": 0.0,
            "layer_precision": 0.0,
            "layer_recall": 0.0,
            "token_guard_accuracy": 0.0,
            "token_guard_trigger_rate": 0.0,
            "router_evaluable_cases": 0.0,
            "router_coverage": 1.0,
            "router_accuracy": 1.0,
        }

    n = float(len(case_results))
    router_rows = [row for row in case_results if row.get("router_evaluable")]
    router_n = float(len(router_rows))
    coverage = (
        sum(float(row.get("router_covered") or 0.0) for row in router_rows) / router_n
        if router_rows
        else 1.0
    )
    accuracy = (
        sum(float(row.get("router_correct") or 0.0) for row in router_rows) / router_n
        if router_rows
        else 1.0
    )
    return {
        "case_count": n,
        "layer_precision": sum(float(row["layer_precision"]) for row in case_results) / n,
        "layer_recall": sum(float(row["layer_recall"]) for row in case_results) / n,
        "token_guard_accuracy": sum(float(row["token_guard_correct"]) for row in case_results) / n,
        "token_guard_trigger_rate": sum(
            float(bool(row.get("token_guard_enabled"))) for row in case_results
        )
        / n,
        "router_evaluable_cases": router_n,
        "router_coverage": coverage,
        "router_accuracy": accuracy,
    }


def aggregate_context_benchmark_results(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    by_query_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in case_results:
        by_stage.setdefault(str(row.get("stage") or "survey"), []).append(row)
        by_query_type.setdefault(str(row.get("query_type") or "generic"), []).append(row)

    return {
        "overall": _aggregate_rows(case_results),
        "by_stage": {name: _aggregate_rows(rows) for name, rows in sorted(by_stage.items())},
        "by_query_type": {
            name: _aggregate_rows(rows) for name, rows in sorted(by_query_type.items())
        },
    }


async def run_context_benchmark(cases: Sequence[ContextBenchmarkCase]) -> Dict[str, Any]:
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        research_store = _FixtureResearchStore(case)
        memory_store = _FixtureMemoryStore(case)
        router_config = TrackRouterConfig(
            use_embeddings=False,
            min_switch_score=0.05,
            min_margin=0.01,
        )
        track_router = TrackRouter(
            research_store=research_store,
            memory_store=memory_store,
            config=router_config,
        )
        engine = ContextEngine(
            research_store=research_store,
            memory_store=memory_store,
            search_service=None,
            track_router=track_router,
            config=ContextEngineConfig(
                offline=True,
                paper_limit=0,
                personalized=False,
                stage=case.stage,
                context_token_budget=case.context_token_budget,
                track_router=router_config,
            ),
        )
        pack = await engine.build_context_pack(
            user_id=case.user_id,
            query=case.query,
            track_id=case.track_id,
            include_cross_track=case.include_cross_track,
            paper_id=case.paper_id,
        )
        case_results.append(evaluate_context_case(case, pack))

    return {
        "cases": case_results,
        "summary": aggregate_context_benchmark_results(case_results),
        "config": {"case_count": len(cases)},
    }


def format_context_benchmark_report(result: Dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    overall = summary.get("overall") or {}
    lines = [
        "Context Engine Benchmark",
        f"Cases: {int(result.get('config', {}).get('case_count', 0))}",
        (
            f"Overall: layer_precision={float(overall.get('layer_precision', 0.0)):.3f} | "
            f"layer_recall={float(overall.get('layer_recall', 0.0)):.3f} | "
            f"token_guard_accuracy={float(overall.get('token_guard_accuracy', 0.0)):.3f} | "
            f"router_coverage={float(overall.get('router_coverage', 0.0)):.3f} | "
            f"router_accuracy={float(overall.get('router_accuracy', 0.0)):.3f}"
        ),
        "By stage:",
    ]

    for name, metrics in (summary.get("by_stage") or {}).items():
        lines.append(
            (
                f"  - {name}: layer_precision={float(metrics.get('layer_precision', 0.0)):.3f}, "
                f"token_guard_accuracy={float(metrics.get('token_guard_accuracy', 0.0)):.3f}, "
                f"router_accuracy={float(metrics.get('router_accuracy', 0.0)):.3f}"
            )
        )

    lines.append("By query_type:")
    for name, metrics in (summary.get("by_query_type") or {}).items():
        lines.append(
            (
                f"  - {name}: layer_precision={float(metrics.get('layer_precision', 0.0)):.3f}, "
                f"token_guard_accuracy={float(metrics.get('token_guard_accuracy', 0.0)):.3f}, "
                f"router_accuracy={float(metrics.get('router_accuracy', 0.0)):.3f}"
            )
        )
    return "\n".join(lines)


__all__ = [
    "ContextBenchmarkCase",
    "aggregate_context_benchmark_results",
    "evaluate_context_case",
    "format_context_benchmark_report",
    "load_context_benchmark_cases",
    "run_context_benchmark",
]
