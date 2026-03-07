"""
Retrieval Bench v2 — Offline retrieval quality evaluation.

Aligned with:
  - LongMemEval (ICLR 2025): 5 memory dimensions
  - LoCoMo (ACL 2024): 5 question types

Metrics: Recall@K, MRR@K, nDCG@K, Hit@K
Modes: fts5 (default) / hybrid (fts5 + HashEmbedding)

Targets:
  - Recall@5  >= 0.80
  - MRR@10    >= 0.65
  - nDCG@10   >= 0.70

Usage:
  PYTHONPATH=src pytest -q evals/memory/test_retrieval_bench.py
  python evals/memory/test_retrieval_bench.py          # standalone
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate
from paperbot.memory.eval.collector import MemoryMetricCollector

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bench_v2"
REPORTS_DIR = Path(__file__).parent / "reports"


# ── IR Metrics ──


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 1.0  # abstention: no relevant → perfect recall if nothing retrieved
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> bool:
    if not relevant_ids:
        return len(retrieved_ids[:k]) == 0  # abstention: hit if nothing retrieved
    return bool(set(retrieved_ids[:k]) & set(relevant_ids))


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    rel_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], grades: Dict[str, int], k: int) -> float:
    total = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = float(grades.get(rid, 0))
        total += rel / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved_ids: List[str], grades: Dict[str, int], k: int) -> float:
    if not grades:
        return 1.0 if not retrieved_ids[:k] else 0.0
    actual = dcg_at_k(retrieved_ids, grades, k)
    ideal_ids = sorted(grades.keys(), key=lambda x: grades[x], reverse=True)
    ideal = dcg_at_k(ideal_ids, grades, k)
    return actual / ideal if ideal > 0 else 0.0


# ── Bench Runner ──


def _load_fixtures() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(FIXTURES_DIR / "bench_memories.json") as f:
        memories = json.load(f)
    with open(FIXTURES_DIR / "retrieval_queries_v2.json") as f:
        queries = json.load(f)
    return memories, queries


def _populate_store(
    store: SqlAlchemyMemoryStore, memories_data: Dict[str, Any]
) -> Dict[str, int]:
    """Insert memories and return mapping from fixture id -> db id."""
    id_map: Dict[str, int] = {}
    for user_key, user_data in memories_data["users"].items():
        user_id = user_key
        for m in user_data["memories"]:
            fixture_id = m["id"]
            candidate = MemoryCandidate(
                kind=m["kind"],
                content=m["content"],
                confidence=m["confidence"],
                tags=m.get("tags", []),
                scope_type=m.get("scope_type", "global"),
                scope_id=m.get("scope_id"),
            )
            created, skipped, rows = store.add_memories(
                user_id=user_id, memories=[candidate], actor_id="bench"
            )
            if created > 0:
                # rows are detached; re-query to get the id
                all_items = store.list_memories(
                    user_id=user_id, limit=200,
                    scope_type=m.get("scope_type", "global"),
                    scope_id=m.get("scope_id"),
                )
                # Find the item by content match
                for item in all_items:
                    if item.get("content") == m["content"]:
                        id_map[fixture_id] = item["id"]
                        break

    return id_map


def _run_retrieval_bench() -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db_url = f"sqlite:///{db_path}"
        store = SqlAlchemyMemoryStore(db_url=db_url)
        collector = MemoryMetricCollector(db_url=db_url)

        memories_data, queries_data = _load_fixtures()
        id_map = _populate_store(store, memories_data)

        # Reverse map: db_id -> fixture_id
        reverse_map: Dict[int, str] = {v: k for k, v in id_map.items()}

        k_values = [1, 3, 5, 10]
        aggregate: Dict[str, List[float]] = {
            f"recall@{k}": [] for k in k_values
        }
        aggregate.update({f"hit@{k}": [] for k in k_values})
        aggregate["mrr@10"] = []
        aggregate["ndcg@10"] = []

        # Abstention cases tracked separately (no relevant docs → should return empty)
        abstention_total = 0
        abstention_correct = 0

        # Per question_type and memory_dimension breakdowns
        by_type: Dict[str, Dict[str, List[float]]] = {}
        by_dim: Dict[str, Dict[str, List[float]]] = {}

        details: List[Dict[str, Any]] = []

        for tc in queries_data["test_cases"]:
            query = tc["query"]
            user_id = tc["user_id"]
            scope = tc.get("scope")
            relevant_fixture_ids = tc["relevant_memory_ids"]
            grades = {fid: g for fid, g in tc.get("relevance_grades", {}).items()}
            qtype = tc.get("question_type", "unknown")
            dim = tc.get("memory_dimension", "unknown")

            search_kwargs: Dict[str, Any] = {
                "user_id": user_id,
                "query": query,
                "limit": 10,
            }
            if scope:
                search_kwargs["scope_type"] = scope["type"]
                search_kwargs["scope_id"] = scope.get("id")

            results = store.search_memories(**search_kwargs)

            # Map retrieved db IDs back to fixture IDs
            retrieved_fixture_ids = []
            for r in results:
                db_id = r.get("id")
                if db_id and db_id in reverse_map:
                    retrieved_fixture_ids.append(reverse_map[db_id])

            is_abstention = len(relevant_fixture_ids) == 0

            # Abstention: separate metric (should return nothing)
            if is_abstention:
                abstention_total += 1
                if len(retrieved_fixture_ids) == 0:
                    abstention_correct += 1
                details.append({
                    "id": tc["id"],
                    "query": query,
                    "question_type": qtype,
                    "memory_dimension": dim,
                    "difficulty": tc.get("difficulty"),
                    "expected": relevant_fixture_ids,
                    "retrieved": retrieved_fixture_ids[:5],
                    "metrics": {"abstention": len(retrieved_fixture_ids) == 0},
                })
                continue

            # Compute metrics (non-abstention only)
            case_metrics: Dict[str, float] = {}
            for k in k_values:
                r = recall_at_k(retrieved_fixture_ids, relevant_fixture_ids, k)
                h = 1.0 if hit_at_k(retrieved_fixture_ids, relevant_fixture_ids, k) else 0.0
                case_metrics[f"recall@{k}"] = r
                case_metrics[f"hit@{k}"] = h
                aggregate[f"recall@{k}"].append(r)
                aggregate[f"hit@{k}"].append(h)

            m = mrr_at_k(retrieved_fixture_ids, relevant_fixture_ids, 10)
            n = ndcg_at_k(retrieved_fixture_ids, grades, 10)
            case_metrics["mrr@10"] = m
            case_metrics["ndcg@10"] = n
            aggregate["mrr@10"].append(m)
            aggregate["ndcg@10"].append(n)

            # Per-type / per-dim breakdown
            for label, bucket in [(qtype, by_type), (dim, by_dim)]:
                if label not in bucket:
                    bucket[label] = {mk: [] for mk in aggregate}
                for mk, mv in case_metrics.items():
                    bucket[label][mk].append(mv)

            details.append({
                "id": tc["id"],
                "query": query,
                "question_type": qtype,
                "memory_dimension": dim,
                "difficulty": tc.get("difficulty"),
                "expected": relevant_fixture_ids,
                "retrieved": retrieved_fixture_ids[:5],
                "metrics": case_metrics,
            })

        # Aggregate
        summary: Dict[str, float] = {}
        for mk, vals in aggregate.items():
            summary[mk] = sum(vals) / len(vals) if vals else 0.0

        summary["abstention_accuracy"] = (
            abstention_correct / abstention_total if abstention_total else 1.0
        )
        # Breakdown summaries
        type_summary = {
            t: {mk: sum(vs) / len(vs) if vs else 0.0 for mk, vs in metrics.items()}
            for t, metrics in by_type.items()
        }
        dim_summary = {
            d: {mk: sum(vs) / len(vs) if vs else 0.0 for mk, vs in metrics.items()}
            for d, metrics in by_dim.items()
        }

        # Record to collector
        total = len(queries_data["test_cases"])
        hits = sum(1 for d in details if d["metrics"].get("hit@5", 0) > 0)
        collector.record_retrieval_hit_rate(
            hits=hits,
            expected=total,
            evaluator_id="bench:retrieval_v2",
            detail={"summary": summary},
        )

        # Targets
        targets = {
            "recall@5": 0.80,
            "mrr@10": 0.65,
            "ndcg@10": 0.70,
        }
        passed = all(summary.get(k, 0) >= v for k, v in targets.items())

        report = {
            "bench": "retrieval_bench_v2",
            "aligned_with": ["LongMemEval", "LoCoMo"],
            "total_queries": total,
            "summary": summary,
            "targets": targets,
            "passed": passed,
            "by_question_type": type_summary,
            "by_memory_dimension": dim_summary,
            "details": details,
        }

        # Save JSON report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "retrieval_bench_v2.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


# ── Pytest entry ──


def test_retrieval_bench_v2():
    result = _run_retrieval_bench()
    print(f"\n{'=' * 60}")
    print("Retrieval Bench v2 Results")
    print(f"{'=' * 60}")
    s = result["summary"]
    for k in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]:
        target = result["targets"].get(k)
        flag = ""
        if target is not None:
            flag = " ✓" if s.get(k, 0) >= target else " ✗"
        print(f"  {k:12s}: {s.get(k, 0):.3f}{flag}")

    print(f"\n  By question_type:")
    for t, ms in result["by_question_type"].items():
        print(f"    {t:25s}: recall@5={ms.get('recall@5', 0):.3f}  mrr@10={ms.get('mrr@10', 0):.3f}")

    print(f"\n  By memory_dimension:")
    for d, ms in result["by_memory_dimension"].items():
        print(f"    {d:30s}: recall@5={ms.get('recall@5', 0):.3f}  mrr@10={ms.get('mrr@10', 0):.3f}")

    print(f"\n  Status: {'PASS' if result['passed'] else 'FAIL'}")

    assert result["passed"], (
        f"Retrieval bench FAILED: recall@5={s.get('recall@5', 0):.3f} "
        f"mrr@10={s.get('mrr@10', 0):.3f} ndcg@10={s.get('ndcg@10', 0):.3f}"
    )


# ── Standalone ──

if __name__ == "__main__":
    result = _run_retrieval_bench()
    s = result["summary"]
    print(f"{'=' * 60}")
    print("Retrieval Bench v2")
    print(f"{'=' * 60}")
    for k in sorted(s):
        target = result["targets"].get(k)
        flag = ""
        if target is not None:
            flag = " ✓" if s[k] >= target else " ✗"
        print(f"  {k:12s}: {s[k]:.3f}{flag}")
    print(f"\n  Status: {'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
