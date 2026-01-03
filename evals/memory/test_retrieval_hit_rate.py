"""
Retrieval Hit Rate Test (Acceptance Criteria)

Target: >= 80% - Relevant memories should be retrieved when needed.

This test:
1. Loads sample memories from fixtures
2. Runs queries from fixtures
3. Checks if expected memory kinds are retrieved
4. Calculates hit rate
5. Records the metric
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate
from paperbot.memory.eval.collector import MemoryMetricCollector


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixtures():
    """Load test fixtures."""
    with open(FIXTURES_DIR / "sample_memories.json") as f:
        memories_data = json.load(f)

    with open(FIXTURES_DIR / "retrieval_queries.json") as f:
        queries_data = json.load(f)

    return memories_data, queries_data


def run_retrieval_hit_rate_test() -> dict:
    """
    Run retrieval hit rate test and return results.

    Returns:
        dict with keys: passed, total_queries, hits, hit_rate, details
    """
    # Use temp database for isolation
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db_url = f"sqlite:///{db_path}"
        store = SqlAlchemyMemoryStore(db_url=db_url)
        collector = MemoryMetricCollector(db_url=db_url)

        # Load fixtures
        memories_data, queries_data = load_fixtures()
        user_id = memories_data["user_id"]

        # Step 1: Insert sample memories
        candidates = [
            MemoryCandidate(
                kind=m["kind"],
                content=m["content"],
                confidence=m["confidence"],
                tags=m.get("tags", []),
            )
            for m in memories_data["memories"]
        ]

        store.add_memories(
            user_id=user_id,
            memories=candidates,
            actor_id="test",
        )

        # Step 2: Run queries and check hits
        total_queries = len(queries_data["test_cases"])
        hits = 0
        details = []

        for test_case in queries_data["test_cases"]:
            query = test_case["query"]
            expected_kinds = set(test_case["expected_kinds"])

            # Search
            results = store.search_memories(
                user_id=user_id,
                query=query,
                limit=10,
            )

            # Check if expected kinds are found
            result_kinds = {r.get("kind") for r in results}
            found_expected = bool(expected_kinds & result_kinds)

            if found_expected:
                hits += 1
                status = "HIT"
            else:
                status = "MISS"

            detail = {
                "id": test_case["id"],
                "query": query,
                "expected_kinds": list(expected_kinds),
                "result_kinds": list(result_kinds),
                "status": status,
            }
            details.append(detail)
            print(f"  {test_case['id']}: {status} (expected {expected_kinds}, got {result_kinds})")

        # Step 3: Calculate hit rate
        hit_rate = hits / total_queries if total_queries > 0 else 0.0

        # Step 4: Record metric
        collector.record_retrieval_hit_rate(
            hits=hits,
            expected=total_queries,
            evaluator_id="test:retrieval_hit_rate",
            detail={"test_cases": details},
        )

        passed = hit_rate >= 0.80  # Target: >= 80%

        return {
            "passed": passed,
            "total_queries": total_queries,
            "hits": hits,
            "hit_rate": hit_rate,
            "details": details,
        }

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    print("=" * 60)
    print("Retrieval Hit Rate Test")
    print("Target: >= 80% (relevant memories should be retrieved)")
    print("=" * 60)

    result = run_retrieval_hit_rate_test()

    print(f"\nResults:")
    print(f"  Total queries: {result['total_queries']}")
    print(f"  Hits: {result['hits']}")
    print(f"  Hit rate: {result['hit_rate']:.1%}")
    print(f"  Target: >= 80%")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
