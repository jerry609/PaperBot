from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore


@dataclass(frozen=True)
class MemoryPerformanceConfig:
    sizes: List[int]
    query_count: int = 25
    batch_size: int = 5000
    seed: int = 42
    search_limit: int = 8


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil((float(p) / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_row(
    *,
    row_id: int,
    user_id: str,
    scope_type: str,
    scope_id: str | None,
    topic: str,
    token: str,
    now_iso: str,
) -> Dict[str, Any]:
    content = (
        f"performance bench topic {topic} token {token} row {row_id} user {user_id} "
        f"scope {scope_type} {scope_id or 'global'}"
    )
    content_hash = hashlib.sha256(
        f"{scope_type}:{scope_id or ''}:{row_id}:{content}".encode("utf-8")
    ).hexdigest()
    return {
        "id": row_id,
        "user_id": user_id,
        "workspace_id": None,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "kind": "fact",
        "content": content,
        "content_hash": content_hash,
        "confidence": 0.9,
        "status": "approved",
        "supersedes_id": None,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
        "last_used_at": None,
        "use_count": 0,
        "pii_risk": 0,
        "tags_json": json.dumps(["perf", topic, scope_type]),
        "evidence_json": "{}",
        "created_at": now_iso,
        "updated_at": now_iso,
        "deleted_at": None,
        "deleted_reason": "",
        "embedding": None,
        "source_id": None,
    }


def _generate_rows(size: int, *, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed + size)
    rows: List[Dict[str, Any]] = []
    now_iso = _now_iso()
    users = ["perf_user_a", "perf_user_b", "perf_user_c", "perf_user_d"]
    track_ids = [f"track_{index}" for index in range(1, 9)]
    paper_ids = [f"paper_{index}" for index in range(1, 9)]
    topics = [f"topic_{index}" for index in range(1, 33)]

    for row_id in range(1, size + 1):
        user_id = users[(row_id - 1) % len(users)]
        topic = topics[(row_id - 1) % len(topics)]
        token = f"kw_{rng.randint(1, 128)}"
        bucket = row_id % 10
        if bucket < 2:
            scope_type = "global"
            scope_id = None
        elif bucket < 7:
            scope_type = "track"
            scope_id = track_ids[(row_id - 1) % len(track_ids)]
        else:
            scope_type = "paper"
            scope_id = paper_ids[(row_id - 1) % len(paper_ids)]
        rows.append(
            _memory_row(
                row_id=row_id,
                user_id=user_id,
                scope_type=scope_type,
                scope_id=scope_id,
                topic=topic,
                token=token,
                now_iso=now_iso,
            )
        )
    return rows


def seed_sqlite_memory_store(
    store: SqlAlchemyMemoryStore,
    *,
    size: int,
    seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    rows = _generate_rows(size, seed=seed)
    db_path = Path(str(store.db_url).replace("sqlite:///", ""))

    started = time.perf_counter()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("DELETE FROM memory_items")
        try:
            conn.execute("DELETE FROM memory_items_fts")
        except Exception:
            pass

        insert_sql = (
            "INSERT INTO memory_items ("
            "id, user_id, workspace_id, scope_type, scope_id, kind, content, content_hash, "
            "confidence, status, supersedes_id, expires_at, last_used_at, use_count, pii_risk, "
            "tags_json, evidence_json, created_at, updated_at, deleted_at, deleted_reason, embedding, source_id"
            ") VALUES ("
            ":id, :user_id, :workspace_id, :scope_type, :scope_id, :kind, :content, :content_hash, "
            ":confidence, :status, :supersedes_id, :expires_at, :last_used_at, :use_count, :pii_risk, "
            ":tags_json, :evidence_json, :created_at, :updated_at, :deleted_at, :deleted_reason, :embedding, :source_id"
            ")"
        )
        for start in range(0, len(rows), max(1, int(batch_size))):
            conn.executemany(insert_sql, rows[start : start + max(1, int(batch_size))])
        conn.commit()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "rows_seeded": size,
        "seed_time_ms": elapsed_ms,
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
    }


def _measure_calls(fn, *, count: int) -> List[float]:
    latencies: List[float] = []
    for _ in range(max(1, int(count))):
        started = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def _latency_summary(latencies: Sequence[float]) -> Dict[str, float]:
    return {
        "count": float(len(latencies)),
        "avg_ms": sum(float(value) for value in latencies) / max(1, len(latencies)),
        "p50_ms": percentile(latencies, 50.0),
        "p95_ms": percentile(latencies, 95.0),
        "p99_ms": percentile(latencies, 99.0),
    }


def run_memory_performance_benchmark(config: MemoryPerformanceConfig) -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    rng = random.Random(config.seed)

    for size in config.sizes:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            store = SqlAlchemyMemoryStore(
                db_url=f"sqlite:///{db_path}",
                embedding_provider=False,
            )
            seed_info = seed_sqlite_memory_store(
                store,
                size=int(size),
                seed=config.seed,
                batch_size=config.batch_size,
            )

            topics = [f"topic_{index}" for index in range(1, 33)]
            track_ids = [f"track_{index}" for index in range(1, 9)]
            paper_ids = [f"paper_{index}" for index in range(1, 9)]

            def search_unscoped() -> None:
                topic = rng.choice(topics)
                store.search_memories(
                    user_id="perf_user_a",
                    query=f"performance bench {topic}",
                    limit=config.search_limit,
                )

            def search_track_scoped() -> None:
                topic = rng.choice(topics)
                scope_id = rng.choice(track_ids)
                store.search_memories(
                    user_id="perf_user_a",
                    query=f"performance bench {topic}",
                    limit=config.search_limit,
                    scope_type="track",
                    scope_id=scope_id,
                )

            def search_batch_track() -> None:
                topic = rng.choice(topics)
                store.search_memories_batch(
                    user_id="perf_user_a",
                    query=f"performance bench {topic}",
                    scope_ids=track_ids[:4],
                    scope_type="track",
                    limit_per_scope=config.search_limit,
                )

            def search_batch_paper() -> None:
                topic = rng.choice(topics)
                store.search_memories_batch(
                    user_id="perf_user_a",
                    query=f"performance bench {topic}",
                    scope_ids=paper_ids[:4],
                    scope_type="paper",
                    limit_per_scope=config.search_limit,
                )

            # warmup
            search_unscoped()
            search_track_scoped()
            search_batch_track()
            search_batch_paper()

            report = {
                "size": int(size),
                "seed": seed_info,
                "search_unscoped": _latency_summary(
                    _measure_calls(search_unscoped, count=config.query_count)
                ),
                "search_track_scoped": _latency_summary(
                    _measure_calls(search_track_scoped, count=config.query_count)
                ),
                "search_batch_track": _latency_summary(
                    _measure_calls(search_batch_track, count=config.query_count)
                ),
                "search_batch_paper": _latency_summary(
                    _measure_calls(search_batch_paper, count=config.query_count)
                ),
            }
            reports.append(report)
        finally:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass

    return {
        "config": {
            "sizes": [int(size) for size in config.sizes],
            "query_count": int(config.query_count),
            "batch_size": int(config.batch_size),
            "seed": int(config.seed),
            "search_limit": int(config.search_limit),
        },
        "reports": reports,
    }


__all__ = [
    "MemoryPerformanceConfig",
    "percentile",
    "run_memory_performance_benchmark",
    "seed_sqlite_memory_store",
]
