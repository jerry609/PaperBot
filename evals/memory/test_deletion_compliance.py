"""
Deletion Compliance Test (Acceptance Criteria)

Target: 100% - Deleted items must NEVER be retrieved.

This test:
1. Creates test memories
2. Soft-deletes some items
3. Attempts to retrieve them via search
4. Verifies deleted items are not returned
5. Records the metric
"""

import os
import sys
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate
from paperbot.memory.eval.collector import MemoryMetricCollector


def run_deletion_compliance_test() -> dict:
    """
    Run deletion compliance test and return results.

    Returns:
        dict with keys: passed, deleted_count, retrieved_count, compliance_rate
    """
    # Use temp database for isolation
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db_url = f"sqlite:///{db_path}"
        store = SqlAlchemyMemoryStore(db_url=db_url)
        collector = MemoryMetricCollector(db_url=db_url)

        user_id = "test_deletion_user"

        # Step 1: Create test memories
        test_memories = [
            MemoryCandidate(kind="profile", content="Test user name is Alice", confidence=0.85),
            MemoryCandidate(kind="preference", content="Prefers markdown format", confidence=0.75),
            MemoryCandidate(kind="project", content="Working on test project", confidence=0.80),
            MemoryCandidate(kind="goal", content="Goal is to test deletion", confidence=0.70),
            MemoryCandidate(kind="fact", content="Test fact that will be deleted", confidence=0.65),
        ]

        created, skipped, _rows = store.add_memories(
            user_id=user_id,
            memories=test_memories,
            actor_id="test",
        )

        # Query to get the created memory IDs (rows are detached from session)
        all_items = store.list_memories(user_id=user_id, limit=100)
        item_ids = [item["id"] for item in all_items]

        # Step 2: Soft-delete some items (last 2 items)
        deleted_ids = []
        for i, item_id in enumerate(item_ids):
            if i >= 3:  # Delete last 2 items
                store.soft_delete_item(
                    user_id=user_id,
                    item_id=item_id,
                    actor_id="test",
                    reason="Test deletion",
                )
                deleted_ids.append(item_id)

        deleted_count = len(deleted_ids)

        # Step 3: Attempt to retrieve ALL items via search
        # Use broad queries that should match deleted items if they were returned
        queries = ["test", "goal", "deletion", "fact", "Alice", "project"]

        retrieved_deleted_count = 0
        for query in queries:
            results = store.search_memories(
                user_id=user_id,
                query=query,
                limit=100,
            )
            # Check if any deleted ID appears in results
            result_ids = {r.get("id") for r in results}
            for deleted_id in deleted_ids:
                if deleted_id in result_ids:
                    retrieved_deleted_count += 1
                    print(f"FAIL: Deleted item {deleted_id} was retrieved for query '{query}'")

        # Step 4: Also check list_memories
        all_items = store.list_memories(user_id=user_id, limit=100)
        for item in all_items:
            if item.get("id") in deleted_ids:
                retrieved_deleted_count += 1
                print(f"FAIL: Deleted item {item.get('id')} was returned in list_memories")

        # Step 5: Calculate compliance
        # Compliance = 1.0 means no deleted items were retrieved
        compliance_rate = 1.0 if retrieved_deleted_count == 0 else 0.0

        # Record metric
        collector.record_deletion_compliance(
            deleted_retrieved_count=retrieved_deleted_count,
            deleted_total_count=deleted_count,
            evaluator_id="test:deletion_compliance",
            detail={
                "deleted_ids": deleted_ids,
                "queries_tested": queries,
            },
        )

        passed = compliance_rate == 1.0

        return {
            "passed": passed,
            "deleted_count": deleted_count,
            "retrieved_count": retrieved_deleted_count,
            "compliance_rate": compliance_rate,
        }

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    print("=" * 60)
    print("P0 Deletion Compliance Test")
    print("Target: 100% (deleted items must never be retrieved)")
    print("=" * 60)

    result = run_deletion_compliance_test()

    print(f"\nResults:")
    print(f"  Deleted items: {result['deleted_count']}")
    print(f"  Retrieved (should be 0): {result['retrieved_count']}")
    print(f"  Compliance rate: {result['compliance_rate']:.1%}")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
