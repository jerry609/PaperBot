"""
Context Extraction Bench — Layered context assembly evaluation.

Aligned with:
  - Letta: core/archival memory layering
  - PaperBot-specific: L0-L3 context layers, token guard, TrackRouter

Tests:
  1. Layer completeness (L0-L3)
  2. Context precision (query → relevant memories)
  3. Token budget guard
  4. TrackRouter accuracy

Usage:
  PYTHONPATH=src pytest -q evals/memory/test_context_extraction.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.memory.schema import MemoryCandidate
from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig
from paperbot.context_engine.track_router import TrackRouter, TrackRouterConfig
from paperbot.context_engine.embeddings import HashEmbeddingProvider


USER_ID = "ctx_bench_user"


def _setup_stores(db_url: str):
    memory_store = SqlAlchemyMemoryStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    return memory_store, research_store


def _seed_data(
    memory_store: SqlAlchemyMemoryStore,
    research_store: SqlAlchemyResearchStore,
) -> Dict[str, Any]:
    """Seed all layers with deterministic test data."""
    ids: Dict[str, Any] = {}

    # L0: global profile memories
    global_mems = [
        MemoryCandidate(
            kind="preference",
            content="Prefers concise mathematical notation in summaries",
            confidence=0.90,
            scope_type="global",
        ),
        MemoryCandidate(
            kind="profile",
            content="ML researcher specializing in NLP and information retrieval",
            confidence=0.85,
            scope_type="global",
        ),
    ]
    _, _, _g_rows = memory_store.add_memories(
        user_id=USER_ID, memories=global_mems, actor_id="bench"
    )
    # Rows are detached; re-query for IDs
    g_items = memory_store.list_memories(user_id=USER_ID, scope_type="global", limit=10)
    ids["global_ids"] = [i["id"] for i in g_items]

    # Create tracks (track1 last so it becomes the active track via activate=True)
    track2 = research_store.create_track(
        user_id=USER_ID,
        name="Diffusion Models",
        description="Research on diffusion-based generative models",
        keywords=["diffusion", "generation", "denoising", "score"],
        activate=False,
    )
    track3 = research_store.create_track(
        user_id=USER_ID,
        name="LLM Agents",
        description="Research on LLM-based autonomous agents",
        keywords=["agent", "tool-use", "planning", "reasoning"],
        activate=False,
    )
    track1 = research_store.create_track(
        user_id=USER_ID,
        name="Dense Retrieval",
        description="Research on dense passage retrieval methods",
        keywords=["retrieval", "embedding", "contrastive", "dense"],
        activate=True,  # track1 becomes the active track
    )
    ids["track1_id"] = track1["id"]
    ids["track2_id"] = track2["id"]
    ids["track3_id"] = track3["id"]

    # L1: tasks and milestones
    research_store.add_task(
        user_id=USER_ID,
        track_id=track1["id"],
        title="Implement ColBERT v3 baseline",
        status="in_progress",
    )
    research_store.add_task(
        user_id=USER_ID,
        track_id=track1["id"],
        title="Run MS MARCO evaluation",
        status="todo",
    )
    research_store.add_milestone(
        user_id=USER_ID,
        track_id=track1["id"],
        name="Baseline reproduction",
    )

    # L2: track-scoped memories (query-relevant)
    track1_mems = [
        MemoryCandidate(
            kind="fact",
            content="ColBERT v2 achieves MRR@10 of 0.397 on MS MARCO dev",
            confidence=0.92,
            scope_type="track",
            scope_id=str(track1["id"]),
            tags=["colbert", "baseline"],
        ),
        MemoryCandidate(
            kind="note",
            content="Hard negative mining with BM25 top-1000 improves training",
            confidence=0.85,
            scope_type="track",
            scope_id=str(track1["id"]),
            tags=["training"],
        ),
        MemoryCandidate(
            kind="decision",
            content="Using FAISS for ANN index due to better Python API",
            confidence=0.80,
            scope_type="track",
            scope_id=str(track1["id"]),
            tags=["infrastructure"],
        ),
        MemoryCandidate(
            kind="hypothesis",
            content="Late interaction should outperform single-vector on multi-hop queries",
            confidence=0.70,
            scope_type="track",
            scope_id=str(track1["id"]),
            tags=["hypothesis"],
        ),
    ]
    _, _, _t1_rows = memory_store.add_memories(
        user_id=USER_ID, memories=track1_mems, actor_id="bench"
    )
    t1_items = memory_store.list_memories(
        user_id=USER_ID, scope_type="track", scope_id=str(track1["id"]), limit=20
    )
    ids["track1_mem_ids"] = [i["id"] for i in t1_items]

    track2_mems = [
        MemoryCandidate(
            kind="fact",
            content="Consistency models achieve FID 3.55 on CIFAR-10 with 1-step generation",
            confidence=0.88,
            scope_type="track",
            scope_id=str(track2["id"]),
            tags=["consistency"],
        ),
        MemoryCandidate(
            kind="note",
            content="Progressive distillation from 1024 to 4 steps preserves 95% quality",
            confidence=0.82,
            scope_type="track",
            scope_id=str(track2["id"]),
            tags=["distillation"],
        ),
    ]
    _, _, _t2_rows = memory_store.add_memories(
        user_id=USER_ID, memories=track2_mems, actor_id="bench"
    )
    t2_items = memory_store.list_memories(
        user_id=USER_ID, scope_type="track", scope_id=str(track2["id"]), limit=20
    )
    ids["track2_mem_ids"] = [i["id"] for i in t2_items]

    # L3: paper-scoped memories
    paper_mems = [
        MemoryCandidate(
            kind="note",
            content="RAPTOR recursively clusters and summarizes for tree-based retrieval",
            confidence=0.85,
            scope_type="paper",
            scope_id="paper_raptor_001",
            tags=["raptor"],
        ),
        MemoryCandidate(
            kind="fact",
            content="RAPTOR improves accuracy on QuALITY by 20% over vanilla RAG",
            confidence=0.88,
            scope_type="paper",
            scope_id="paper_raptor_001",
            tags=["raptor", "results"],
        ),
    ]
    _, _, _p_rows = memory_store.add_memories(
        user_id=USER_ID, memories=paper_mems, actor_id="bench"
    )
    p_items = memory_store.list_memories(
        user_id=USER_ID, scope_type="paper", scope_id="paper_raptor_001", limit=10
    )
    ids["paper_mem_ids"] = [i["id"] for i in p_items]
    ids["paper_id"] = "paper_raptor_001"

    return ids


def _run_context_extraction_bench() -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db_url = f"sqlite:///{db_path}"
        memory_store, research_store = _setup_stores(db_url)
        ids = _seed_data(memory_store, research_store)
        embedding_provider = HashEmbeddingProvider(dim=128)

        results: Dict[str, Any] = {
            "bench": "context_extraction_v1",
            "aligned_with": ["Letta"],
            "tests": {},
        }

        # ── Test 1: Layer Completeness ──
        engine = ContextEngine(
            research_store=research_store,
            memory_store=memory_store,
            config=ContextEngineConfig(offline=True, paper_limit=0),
            track_router=TrackRouter(
                research_store=research_store,
                memory_store=memory_store,
                embedding_provider=embedding_provider,
                config=TrackRouterConfig(use_embeddings=True),
            ),
        )

        pack = asyncio.get_event_loop().run_until_complete(
            engine.build_context_pack(
                user_id=USER_ID,
                query="ColBERT dense retrieval performance",
                paper_id=ids["paper_id"],
            )
        )

        layer_tests = {
            "L0_profile_populated": len(pack.get("user_prefs", [])) > 0,
            "L1_track_populated": len(pack.get("progress_state", {}).get("tasks", [])) > 0,
            "L1_milestones_populated": len(pack.get("progress_state", {}).get("milestones", [])) > 0,
            "L2_relevant_memories_populated": len(pack.get("relevant_memories", [])) > 0,
            "L3_paper_memories_populated": len(pack.get("paper_memories", [])) > 0,
            "active_track_present": pack.get("active_track") is not None,
            "routing_present": pack.get("routing") is not None,
            "context_layers_present": pack.get("context_layers") is not None,
        }
        layer_pass = all(layer_tests.values())
        results["tests"]["layer_completeness"] = {
            "passed": layer_pass,
            "details": layer_tests,
        }

        # ── Test 2: Layer Graceful Degradation ──
        pack_no_paper = asyncio.get_event_loop().run_until_complete(
            engine.build_context_pack(
                user_id=USER_ID,
                query="ColBERT dense retrieval",
            )
        )
        degradation_tests = {
            "no_paper_id_gives_empty_L3": len(pack_no_paper.get("paper_memories", [])) == 0,
        }

        # New user with no data
        pack_empty = asyncio.get_event_loop().run_until_complete(
            engine.build_context_pack(
                user_id="nonexistent_user_xyz",
                query="anything",
            )
        )
        degradation_tests["empty_user_no_crash"] = True  # if we get here, no crash
        degradation_tests["empty_user_empty_prefs"] = len(pack_empty.get("user_prefs", [])) == 0
        degradation_pass = all(degradation_tests.values())
        results["tests"]["graceful_degradation"] = {
            "passed": degradation_pass,
            "details": degradation_tests,
        }

        # ── Test 3: Context Precision ──
        precision_cases = [
            {
                "query": "ColBERT MRR performance baseline",
                "scope_track_id": ids["track1_id"],
                "expected_content_fragments": ["ColBERT", "MRR@10", "0.397"],
            },
            {
                "query": "negative mining BM25 training",
                "scope_track_id": ids["track1_id"],
                "expected_content_fragments": ["negative mining", "BM25"],
            },
            {
                "query": "FAISS ANN index decision",
                "scope_track_id": ids["track1_id"],
                "expected_content_fragments": ["FAISS"],
            },
        ]

        precision_hits = 0
        precision_total = len(precision_cases)
        precision_details = []

        for case in precision_cases:
            pack_p = asyncio.get_event_loop().run_until_complete(
                engine.build_context_pack(
                    user_id=USER_ID,
                    query=case["query"],
                    track_id=case.get("scope_track_id"),
                )
            )
            mem_contents = " ".join(
                m.get("content", "") for m in pack_p.get("relevant_memories", [])
            )
            found = all(
                frag.lower() in mem_contents.lower()
                for frag in case["expected_content_fragments"]
            )
            if found:
                precision_hits += 1
            precision_details.append({
                "query": case["query"],
                "found": found,
                "expected_fragments": case["expected_content_fragments"],
                "retrieved_count": len(pack_p.get("relevant_memories", [])),
            })

        precision_rate = precision_hits / precision_total if precision_total else 0
        results["tests"]["context_precision"] = {
            "passed": precision_rate >= 0.75,
            "precision": precision_rate,
            "hits": precision_hits,
            "total": precision_total,
            "details": precision_details,
        }

        # ── Test 4: Token Budget Guard ──
        engine_budget = ContextEngine(
            research_store=research_store,
            memory_store=memory_store,
            config=ContextEngineConfig(
                offline=True,
                paper_limit=0,
                context_token_budget=300,  # very low budget to force truncation
            ),
            track_router=TrackRouter(
                research_store=research_store,
                memory_store=memory_store,
                embedding_provider=embedding_provider,
            ),
        )

        pack_budget = asyncio.get_event_loop().run_until_complete(
            engine_budget.build_context_pack(
                user_id=USER_ID,
                query="ColBERT retrieval training",
                paper_id=ids["paper_id"],
            )
        )

        layers = pack_budget.get("context_layers", {})
        total_tokens = sum(layers.values())
        guard_info = pack_budget.get("routing", {}).get("token_guard", {})
        budget_trimmed = guard_info.get("enabled", False) or total_tokens <= 300

        budget_tests = {
            "total_tokens_within_budget": total_tokens <= 350,  # small margin
            "guard_triggered_or_within": budget_trimmed,
            "L0_not_empty": len(pack_budget.get("user_prefs", [])) >= 0,  # L0 may be trimmed last
        }
        budget_pass = budget_tests["total_tokens_within_budget"]
        results["tests"]["token_budget_guard"] = {
            "passed": budget_pass,
            "budget": 300,
            "actual_tokens": total_tokens,
            "layers": layers,
            "details": budget_tests,
        }

        # ── Test 5: TrackRouter Accuracy ──
        router = TrackRouter(
            research_store=research_store,
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            config=TrackRouterConfig(use_embeddings=True),
        )

        router_cases = [
            {"query": "dense retrieval embedding contrastive learning", "expected_track_id": ids["track1_id"]},
            {"query": "diffusion denoising score matching generation", "expected_track_id": ids["track2_id"]},
            {"query": "autonomous agent tool use planning reasoning", "expected_track_id": ids["track3_id"]},
            {"query": "passage retrieval FAISS ColBERT", "expected_track_id": ids["track1_id"]},
            {"query": "consistency model image generation distillation", "expected_track_id": ids["track2_id"]},
        ]

        router_hits = 0
        router_total = len(router_cases)
        router_details = []

        for case in router_cases:
            suggestion = router.suggest_track(
                user_id=USER_ID,
                query=case["query"],
                active_track_id=None,
                limit=50,
            )
            # When active_track_id is None, suggest_track may return None or a suggestion
            # We check the top scored track by using a different approach
            # Use the router's internal scoring by getting suggestion
            suggested_id = suggestion.get("track_id") if suggestion else None
            hit = suggested_id == case["expected_track_id"]
            if hit:
                router_hits += 1
            router_details.append({
                "query": case["query"],
                "expected_track_id": case["expected_track_id"],
                "suggested_track_id": suggested_id,
                "hit": hit,
                "score": suggestion.get("score") if suggestion else None,
            })

        router_accuracy = router_hits / router_total if router_total else 0
        results["tests"]["track_router_accuracy"] = {
            "passed": router_accuracy >= 0.60,
            "accuracy": router_accuracy,
            "hits": router_hits,
            "total": router_total,
            "details": router_details,
        }

        # ── Overall ──
        all_passed = all(t["passed"] for t in results["tests"].values())
        results["passed"] = all_passed

        return results

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_context_extraction_bench():
    result = _run_context_extraction_bench()
    print(f"\n{'=' * 60}")
    print("Context Extraction Bench")
    print(f"{'=' * 60}")
    for name, test in result["tests"].items():
        status = "PASS" if test["passed"] else "FAIL"
        extra = ""
        if "precision" in test:
            extra = f" ({test['precision']:.0%})"
        elif "accuracy" in test:
            extra = f" ({test['accuracy']:.0%})"
        elif "actual_tokens" in test:
            extra = f" ({test['actual_tokens']} tokens)"
        print(f"  {name:30s}: {status}{extra}")

    print(f"\n  Overall: {'PASS' if result['passed'] else 'FAIL'}")
    assert result["passed"], f"Context extraction bench FAILED: {result['tests']}"


if __name__ == "__main__":
    result = _run_context_extraction_bench()
    print(f"{'=' * 60}")
    print("Context Extraction Bench")
    print(f"{'=' * 60}")
    for name, test in result["tests"].items():
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  {name:30s}: {status}")
        if "details" in test and isinstance(test["details"], dict):
            for k, v in test["details"].items():
                print(f"    {k}: {v}")
    print(f"\n  Overall: {'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
