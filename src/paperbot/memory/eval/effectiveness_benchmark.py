from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from paperbot.application.services.llm_service import LLMService
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate


@dataclass(frozen=True)
class EffectivenessMemoryWrite:
    kind: str
    content: str
    confidence: float = 0.9
    tags: List[str] = field(default_factory=list)
    scope_type: Optional[str] = None
    scope_id: Optional[str] = None


@dataclass(frozen=True)
class EffectivenessSession:
    session_id: str
    writes: List[EffectivenessMemoryWrite] = field(default_factory=list)


@dataclass(frozen=True)
class EffectivenessQuestion:
    question_id: str
    query: str
    expected_answer: str
    acceptable_answers: List[str] = field(default_factory=list)
    expected_memory_substrings: List[str] = field(default_factory=list)
    category: str = "fact"
    scope_type: Optional[str] = None
    scope_id: Optional[str] = None


@dataclass(frozen=True)
class EffectivenessCase:
    case_id: str
    user_id: str
    sessions: List[EffectivenessSession] = field(default_factory=list)
    questions: List[EffectivenessQuestion] = field(default_factory=list)


@dataclass
class EffectivenessQuestionResult:
    case_id: str
    question_id: str
    category: str
    query: str
    answer: str
    expected_answer: str
    retrieved_contents: List[str] = field(default_factory=list)
    retrieved_hit: bool = False
    answer_correct: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryAnswerRunner(Protocol):
    def answer(
        self,
        question: EffectivenessQuestion,
        *,
        retrieved_memories: Sequence[Dict[str, Any]],
    ) -> str: ...


class HeuristicMemoryAnswerRunner:
    _STOPWORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "should",
        "be",
        "used",
        "for",
        "of",
        "now",
        "this",
        "paper",
        "current",
    }

    @classmethod
    def answer(
        cls,
        question: EffectivenessQuestion,
        *,
        retrieved_memories: Sequence[Dict[str, Any]],
    ) -> str:
        if not retrieved_memories:
            return "INSUFFICIENT_MEMORY"

        query_tokens = {
            token
            for token in _normalize(question.query).replace("?", "").split()
            if token and token not in cls._STOPWORDS
        }
        scored = []
        for memory in retrieved_memories:
            content = str(memory.get("content") or "").strip()
            content_tokens = set(_normalize(content).replace(":", " ").split())
            overlap = len(query_tokens & content_tokens)
            scored.append((overlap, cls._created_at(memory), memory))

        best_overlap = max(item[0] for item in scored) if scored else 0
        if best_overlap <= 0 and question.category == "abstain":
            return "INSUFFICIENT_MEMORY"

        relevant = (
            [item for item in scored if item[0] == best_overlap] if best_overlap > 0 else scored
        )
        colon_grouped = {}
        for overlap, created_at, memory in relevant:
            content = str(memory.get("content") or "").strip()
            key = content.split(":", 1)[0].strip().lower() if ":" in content else content.lower()
            existing = colon_grouped.get(key)
            if existing is None or created_at > existing[0]:
                colon_grouped[key] = (created_at, content)

        if colon_grouped:
            newest_content = max(colon_grouped.values(), key=lambda item: item[0])[1]
        else:
            newest_content = str(relevant[0][2].get("content") or "").strip()

        if ":" in newest_content:
            return newest_content.split(":", 1)[1].strip()
        return newest_content or "INSUFFICIENT_MEMORY"

    @staticmethod
    def _created_at(memory: Dict[str, Any]) -> datetime:
        raw = memory.get("created_at")
        if isinstance(raw, str) and raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.min


class LLMMemoryAnswerRunner:
    def __init__(self, llm: Optional[LLMService] = None) -> None:
        self._llm = llm or LLMService()
        self._fallback = HeuristicMemoryAnswerRunner()

    def answer(
        self,
        question: EffectivenessQuestion,
        *,
        retrieved_memories: Sequence[Dict[str, Any]],
    ) -> str:
        if not retrieved_memories:
            return "INSUFFICIENT_MEMORY"
        memories = []
        for index, memory in enumerate(retrieved_memories, start=1):
            memories.append(f"{index}. {memory.get('content', '')}")
        system = (
            "Answer the user question using only the provided memories. "
            "If the memories are insufficient or conflicting, reply exactly INSUFFICIENT_MEMORY. "
            "Return only the short final answer."
        )
        user = f"Question: {question.query}\n\n" f"Memories:\n" + "\n".join(memories)
        answer = (self._llm.complete(task_type="reasoning", system=system, user=user) or "").strip()
        return answer or self._fallback.answer(question, retrieved_memories=retrieved_memories)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def load_effectiveness_cases(path: str | Path) -> List[EffectivenessCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: List[EffectivenessCase] = []
    for row in payload:
        sessions = []
        for session_row in row.get("sessions", []) or []:
            writes = []
            for write_row in session_row.get("writes", []) or []:
                writes.append(
                    EffectivenessMemoryWrite(
                        kind=str(write_row.get("kind") or "fact"),
                        content=str(write_row.get("content") or ""),
                        confidence=float(write_row.get("confidence") or 0.9),
                        tags=[str(tag) for tag in write_row.get("tags", []) or []],
                        scope_type=(
                            str(write_row.get("scope_type"))
                            if write_row.get("scope_type")
                            else None
                        ),
                        scope_id=(
                            str(write_row.get("scope_id")) if write_row.get("scope_id") else None
                        ),
                    )
                )
            sessions.append(
                EffectivenessSession(
                    session_id=str(session_row.get("session_id") or ""),
                    writes=writes,
                )
            )
        questions = []
        for question_row in row.get("questions", []) or []:
            questions.append(
                EffectivenessQuestion(
                    question_id=str(question_row.get("question_id") or ""),
                    query=str(question_row.get("query") or ""),
                    expected_answer=str(question_row.get("expected_answer") or ""),
                    acceptable_answers=[
                        str(answer) for answer in question_row.get("acceptable_answers", []) or []
                    ],
                    expected_memory_substrings=[
                        str(item)
                        for item in question_row.get("expected_memory_substrings", []) or []
                    ],
                    category=str(question_row.get("category") or "fact"),
                    scope_type=(
                        str(question_row.get("scope_type"))
                        if question_row.get("scope_type")
                        else None
                    ),
                    scope_id=(
                        str(question_row.get("scope_id")) if question_row.get("scope_id") else None
                    ),
                )
            )
        cases.append(
            EffectivenessCase(
                case_id=str(row.get("case_id") or ""),
                user_id=str(row.get("user_id") or "effectiveness_bench"),
                sessions=sessions,
                questions=questions,
            )
        )
    return cases


def seed_effectiveness_case(store: SqlAlchemyMemoryStore, case: EffectivenessCase) -> None:
    for session in case.sessions:
        memories = [
            MemoryCandidate(
                kind=write.kind,
                content=write.content,
                confidence=write.confidence,
                tags=list(write.tags),
                scope_type=write.scope_type,
                scope_id=write.scope_id,
                status="approved",
            )
            for write in session.writes
        ]
        if memories:
            store.add_memories(
                user_id=case.user_id, memories=memories, actor_id=f"session:{session.session_id}"
            )


def evaluate_effectiveness_question(
    store: SqlAlchemyMemoryStore,
    case: EffectivenessCase,
    question: EffectivenessQuestion,
    *,
    runner: MemoryAnswerRunner,
    top_k: int = 4,
) -> EffectivenessQuestionResult:
    retrieved = store.search_memories(
        user_id=case.user_id,
        query=question.query,
        limit=max(1, int(top_k)),
        scope_type=question.scope_type,
        scope_id=question.scope_id,
    )
    retrieved_contents = [str(item.get("content") or "") for item in retrieved]
    expected_substrings = list(question.expected_memory_substrings or [])
    if (
        not expected_substrings
        and question.expected_answer
        and question.expected_answer != "INSUFFICIENT_MEMORY"
    ):
        expected_substrings = [question.expected_answer]
    retrieved_hit = (
        any(
            _normalize(expected) in _normalize(content)
            for expected in expected_substrings
            for content in retrieved_contents
        )
        if expected_substrings
        else not retrieved_contents
    )

    answer = runner.answer(question, retrieved_memories=retrieved)
    normalized_answer = _normalize(answer)
    acceptable = [_normalize(question.expected_answer)] + [
        _normalize(item) for item in question.acceptable_answers
    ]
    answer_correct = normalized_answer in acceptable

    return EffectivenessQuestionResult(
        case_id=case.case_id,
        question_id=question.question_id,
        category=question.category,
        query=question.query,
        answer=answer,
        expected_answer=question.expected_answer,
        retrieved_contents=retrieved_contents,
        retrieved_hit=retrieved_hit,
        answer_correct=answer_correct,
        metadata={
            "scope_type": question.scope_type,
            "scope_id": question.scope_id,
            "retrieved_count": len(retrieved),
        },
    )


def summarize_effectiveness_results(
    results: Sequence[EffectivenessQuestionResult],
) -> Dict[str, Any]:
    if not results:
        return {
            "question_count": 0,
            "retrieval_hit_rate": 0.0,
            "answer_accuracy": 0.0,
            "temporal_accuracy": 0.0,
            "update_accuracy": 0.0,
            "abstention_accuracy": 0.0,
        }

    def _rate(items: Sequence[EffectivenessQuestionResult], attr: str) -> float:
        if not items:
            return 0.0
        return sum(1.0 if getattr(item, attr) else 0.0 for item in items) / len(items)

    temporal = [item for item in results if item.category == "temporal"]
    updates = [item for item in results if item.category == "update"]
    abstentions = [item for item in results if item.category == "abstain"]
    return {
        "question_count": len(results),
        "retrieval_hit_rate": _rate(results, "retrieved_hit"),
        "answer_accuracy": _rate(results, "answer_correct"),
        "temporal_accuracy": _rate(temporal, "answer_correct"),
        "update_accuracy": _rate(updates, "answer_correct"),
        "abstention_accuracy": _rate(abstentions, "answer_correct"),
    }


def run_effectiveness_benchmark(
    cases: Sequence[EffectivenessCase],
    *,
    runner: MemoryAnswerRunner,
    top_k: int = 4,
) -> Dict[str, Any]:
    all_results: List[EffectivenessQuestionResult] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix=f"effectiveness_{case.case_id}_") as temp_dir:
            db_url = f"sqlite:///{Path(temp_dir) / 'effectiveness.db'}"
            store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)
            seed_effectiveness_case(store, case)
            for question in case.questions:
                all_results.append(
                    evaluate_effectiveness_question(
                        store,
                        case,
                        question,
                        runner=runner,
                        top_k=top_k,
                    )
                )

    return {
        "config": {
            "cases": len(cases),
            "top_k": max(1, int(top_k)),
            "runner": runner.__class__.__name__,
        },
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in all_results],
        "summary": summarize_effectiveness_results(all_results),
    }
