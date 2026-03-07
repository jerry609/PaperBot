"""
Scope Isolation Test (P0 Acceptance Criteria)

Target: 0% leakage - no memory from another user or another scope should appear.

This test:
1. Builds a deterministic user × scope matrix
2. Exercises search_memories(), list_memories(), and search_memories_batch()
3. Fails on any cross-user or cross-scope leakage
4. Verifies same-user global memories stay visible on unscoped queries
5. Records cross_user_leak_rate and cross_scope_leak_rate
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.eval.collector import MemoryMetricCollector
from paperbot.memory.schema import MemoryCandidate


USERS = ["isolation_user_a", "isolation_user_b"]
TRACK_SCOPE_IDS = ["track_1", "track_2"]
PAPER_SCOPE_IDS = ["paper_x", "paper_y"]
COMMON_QUERY = "scopebench isolation sentinel"


def _make_candidate(*, marker: str, scope_type: str, scope_id: Optional[str]) -> MemoryCandidate:
    scope_desc = scope_id or "global"
    return MemoryCandidate(
        kind="fact",
        content=(
            f"{COMMON_QUERY} unique_marker_{marker} "
            f"owned by {marker} in {scope_type}:{scope_desc}"
        ),
        confidence=0.90,
        tags=["scopebench", "isolation", scope_type, scope_desc],
        scope_type=scope_type,
        scope_id=scope_id,
    )


def _seed_scope_matrix(store: SqlAlchemyMemoryStore) -> Dict[Tuple[str, str, Optional[str]], int]:
    created: Dict[Tuple[str, str, Optional[str]], int] = {}

    for user_id in USERS:
        _, _, rows = store.add_memories(
            user_id=user_id,
            memories=[
                _make_candidate(marker=f"{user_id}_global", scope_type="global", scope_id=None)
            ],
            actor_id="scopebench",
        )
        created[(user_id, "global", None)] = rows[0].id

        for scope_id in TRACK_SCOPE_IDS:
            _, _, rows = store.add_memories(
                user_id=user_id,
                memories=[
                    _make_candidate(
                        marker=f"{user_id}_{scope_id}",
                        scope_type="track",
                        scope_id=scope_id,
                    )
                ],
                actor_id="scopebench",
            )
            created[(user_id, "track", scope_id)] = rows[0].id

        for scope_id in PAPER_SCOPE_IDS:
            _, _, rows = store.add_memories(
                user_id=user_id,
                memories=[
                    _make_candidate(
                        marker=f"{user_id}_{scope_id}",
                        scope_type="paper",
                        scope_id=scope_id,
                    )
                ],
                actor_id="scopebench",
            )
            created[(user_id, "paper", scope_id)] = rows[0].id

    return created


def _describe_item(item: Dict[str, Any]) -> str:
    return (
        f"id={item.get('id')} user={item.get('user_id')} "
        f"scope={item.get('scope_type')}:{item.get('scope_id')}"
    )


def _collect_leaks(
    *,
    results: Sequence[Dict[str, Any]],
    requested_user: str,
    allowed_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    allow = set(int(item_id) for item_id in allowed_ids)
    cross_user = [item for item in results if str(item.get("user_id") or "") != requested_user]
    cross_scope = [item for item in results if int(item.get("id") or 0) not in allow]
    return cross_user, cross_scope


def _log_failures(
    *,
    path: str,
    requested_user: str,
    requested_scope_type: Optional[str],
    requested_scope_id: Optional[str],
    requested_scope_ids: Optional[Sequence[str]],
    cross_user: Sequence[Dict[str, Any]],
    cross_scope: Sequence[Dict[str, Any]],
    missing_required: Sequence[int],
) -> None:
    scope_desc = (
        f"{requested_scope_type}:{requested_scope_id}"
        if requested_scope_id is not None
        else str(requested_scope_type)
    )
    if requested_scope_ids:
        scope_desc = f"{requested_scope_type}:{','.join(requested_scope_ids)}"
    for item in cross_user:
        print(
            f"FAIL [{path}] cross-user leak for user={requested_user} scope={scope_desc}: "
            f"{_describe_item(item)}"
        )
    for item in cross_scope:
        print(
            f"FAIL [{path}] cross-scope leak for user={requested_user} scope={scope_desc}: "
            f"{_describe_item(item)}"
        )
    if missing_required:
        print(
            f"FAIL [{path}] missing expected same-user items for user={requested_user} scope={scope_desc}: "
            f"missing_ids={list(missing_required)}"
        )


def run_scope_isolation_test() -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db_url = f"sqlite:///{db_path}"
        store = SqlAlchemyMemoryStore(db_url=db_url, embedding_provider=False)
        collector = MemoryMetricCollector(db_url=db_url)
        created = _seed_scope_matrix(store)

        total_checks = 0
        cross_user_leak_checks = 0
        cross_scope_leak_checks = 0
        visibility_failures = 0
        details: List[Dict[str, Any]] = []

        def record_check(
            *,
            path: str,
            requested_user: str,
            requested_scope_type: Optional[str],
            requested_scope_id: Optional[str],
            requested_scope_ids: Optional[Sequence[str]],
            results: Sequence[Dict[str, Any]],
            allowed_ids: Sequence[int],
            required_ids: Optional[Sequence[int]] = None,
        ) -> None:
            nonlocal total_checks, cross_user_leak_checks, cross_scope_leak_checks, visibility_failures

            total_checks += 1
            cross_user, cross_scope = _collect_leaks(
                results=results,
                requested_user=requested_user,
                allowed_ids=allowed_ids,
            )
            result_ids = {int(item.get("id") or 0) for item in results if item.get("id")}
            required = {int(item_id) for item_id in (required_ids or []) if item_id}
            missing_required = sorted(required - result_ids)

            if cross_user:
                cross_user_leak_checks += 1
            if cross_scope:
                cross_scope_leak_checks += 1
            if missing_required:
                visibility_failures += 1

            if cross_user or cross_scope or missing_required:
                _log_failures(
                    path=path,
                    requested_user=requested_user,
                    requested_scope_type=requested_scope_type,
                    requested_scope_id=requested_scope_id,
                    requested_scope_ids=requested_scope_ids,
                    cross_user=cross_user,
                    cross_scope=cross_scope,
                    missing_required=missing_required,
                )

            details.append(
                {
                    "path": path,
                    "requested_user": requested_user,
                    "requested_scope_type": requested_scope_type,
                    "requested_scope_id": requested_scope_id,
                    "requested_scope_ids": list(requested_scope_ids or []),
                    "result_ids": sorted(result_ids),
                    "allowed_ids": sorted(int(item_id) for item_id in allowed_ids),
                    "required_ids": sorted(required),
                    "cross_user_leaks": [_describe_item(item) for item in cross_user],
                    "cross_scope_leaks": [_describe_item(item) for item in cross_scope],
                    "missing_required": missing_required,
                }
            )

        for user_id in USERS:
            global_id = created[(user_id, "global", None)]
            all_same_user_ids = [
                item_id
                for (owner, _scope_type, _scope_id), item_id in created.items()
                if owner == user_id
            ]

            search_all = store.search_memories(user_id=user_id, query=COMMON_QUERY, limit=100)
            record_check(
                path="search_memories:unscoped",
                requested_user=user_id,
                requested_scope_type=None,
                requested_scope_id=None,
                requested_scope_ids=None,
                results=search_all,
                allowed_ids=all_same_user_ids,
                required_ids=[global_id],
            )

            list_all = store.list_memories(user_id=user_id, limit=100)
            record_check(
                path="list_memories:unscoped",
                requested_user=user_id,
                requested_scope_type=None,
                requested_scope_id=None,
                requested_scope_ids=None,
                results=list_all,
                allowed_ids=all_same_user_ids,
                required_ids=[global_id],
            )

            global_results = store.search_memories(
                user_id=user_id,
                query=COMMON_QUERY,
                limit=100,
                scope_type="global",
            )
            record_check(
                path="search_memories:global",
                requested_user=user_id,
                requested_scope_type="global",
                requested_scope_id=None,
                requested_scope_ids=None,
                results=global_results,
                allowed_ids=[global_id],
                required_ids=[global_id],
            )

            for scope_id in TRACK_SCOPE_IDS:
                expected_id = created[(user_id, "track", scope_id)]
                track_results = store.search_memories(
                    user_id=user_id,
                    query=COMMON_QUERY,
                    limit=100,
                    scope_type="track",
                    scope_id=scope_id,
                )
                record_check(
                    path="search_memories:track",
                    requested_user=user_id,
                    requested_scope_type="track",
                    requested_scope_id=scope_id,
                    requested_scope_ids=None,
                    results=track_results,
                    allowed_ids=[expected_id],
                    required_ids=[expected_id],
                )

                track_list = store.list_memories(
                    user_id=user_id,
                    limit=100,
                    scope_type="track",
                    scope_id=scope_id,
                )
                record_check(
                    path="list_memories:track",
                    requested_user=user_id,
                    requested_scope_type="track",
                    requested_scope_id=scope_id,
                    requested_scope_ids=None,
                    results=track_list,
                    allowed_ids=[expected_id],
                    required_ids=[expected_id],
                )

            for scope_id in PAPER_SCOPE_IDS:
                expected_id = created[(user_id, "paper", scope_id)]
                paper_results = store.search_memories(
                    user_id=user_id,
                    query=COMMON_QUERY,
                    limit=100,
                    scope_type="paper",
                    scope_id=scope_id,
                )
                record_check(
                    path="search_memories:paper",
                    requested_user=user_id,
                    requested_scope_type="paper",
                    requested_scope_id=scope_id,
                    requested_scope_ids=None,
                    results=paper_results,
                    allowed_ids=[expected_id],
                    required_ids=[expected_id],
                )

                paper_list = store.list_memories(
                    user_id=user_id,
                    limit=100,
                    scope_type="paper",
                    scope_id=scope_id,
                )
                record_check(
                    path="list_memories:paper",
                    requested_user=user_id,
                    requested_scope_type="paper",
                    requested_scope_id=scope_id,
                    requested_scope_ids=None,
                    results=paper_list,
                    allowed_ids=[expected_id],
                    required_ids=[expected_id],
                )

            track_batch = store.search_memories_batch(
                user_id=user_id,
                query=COMMON_QUERY,
                scope_ids=TRACK_SCOPE_IDS,
                scope_type="track",
                limit_per_scope=10,
            )
            for scope_id in TRACK_SCOPE_IDS:
                record_check(
                    path="search_memories_batch:track",
                    requested_user=user_id,
                    requested_scope_type="track",
                    requested_scope_id=scope_id,
                    requested_scope_ids=TRACK_SCOPE_IDS,
                    results=track_batch.get(scope_id, []),
                    allowed_ids=[created[(user_id, "track", scope_id)]],
                    required_ids=[created[(user_id, "track", scope_id)]],
                )

            paper_batch = store.search_memories_batch(
                user_id=user_id,
                query=COMMON_QUERY,
                scope_ids=PAPER_SCOPE_IDS,
                scope_type="paper",
                limit_per_scope=10,
            )
            for scope_id in PAPER_SCOPE_IDS:
                record_check(
                    path="search_memories_batch:paper",
                    requested_user=user_id,
                    requested_scope_type="paper",
                    requested_scope_id=scope_id,
                    requested_scope_ids=PAPER_SCOPE_IDS,
                    results=paper_batch.get(scope_id, []),
                    allowed_ids=[created[(user_id, "paper", scope_id)]],
                    required_ids=[created[(user_id, "paper", scope_id)]],
                )

        collector.record_scope_isolation(
            cross_user_leak_count=cross_user_leak_checks,
            cross_user_total_checks=total_checks,
            cross_scope_leak_count=cross_scope_leak_checks,
            cross_scope_total_checks=total_checks,
            evaluator_id="test:scope_isolation",
            detail={
                "checks": details,
                "visibility_failures": visibility_failures,
            },
        )

        passed = (
            cross_user_leak_checks == 0
            and cross_scope_leak_checks == 0
            and visibility_failures == 0
        )

        return {
            "passed": passed,
            "total_checks": total_checks,
            "cross_user_leak_checks": cross_user_leak_checks,
            "cross_scope_leak_checks": cross_scope_leak_checks,
            "visibility_failures": visibility_failures,
            "details": details,
        }
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main() -> int:
    print("=" * 60)
    print("P0 Scope Isolation Test")
    print("Target: 0% cross-user / cross-scope leakage")
    print("=" * 60)

    result = run_scope_isolation_test()
    print("\nResults:")
    print(f"  Total checks: {result['total_checks']}")
    print(f"  Cross-user leak checks: {result['cross_user_leak_checks']}")
    print(f"  Cross-scope leak checks: {result['cross_scope_leak_checks']}")
    print(f"  Visibility failures: {result['visibility_failures']}")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
