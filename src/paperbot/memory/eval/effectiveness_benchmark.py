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
        "user",
        "track",
        "does",
        "do",
        "did",
    }
    _PAST_CUES = {"first", "initial", "original", "earlier", "previous", "prior"}
    _LATEST_CUES = {"now", "current", "currently", "latest", "final", "updated"}

    @classmethod
    def answer(
        cls,
        question: EffectivenessQuestion,
        *,
        retrieved_memories: Sequence[Dict[str, Any]],
    ) -> str:
        if not retrieved_memories:
            return "INSUFFICIENT_MEMORY"

        query_text = _normalize(question.query).replace("?", " ")
        query_tokens = {
            token for token in query_text.split() if token and token not in cls._STOPWORDS
        }
        scored = []
        for memory in retrieved_memories:
            content = str(memory.get("content") or "").strip()
            content_tokens = set(_normalize(content).replace(":", " ").split())
            overlap = len(query_tokens & content_tokens)
            scored.append((overlap, cls._created_at(memory), memory))

        best_overlap = max(item[0] for item in scored) if scored else 0
        if question.category == "abstain" and best_overlap <= 1:
            return "INSUFFICIENT_MEMORY"

        relevant = (
            [item for item in scored if item[0] == best_overlap] if best_overlap > 0 else scored
        )
        selection_mode = cls._selection_mode(question, query_tokens)
        selected_content = cls._select_content(
            query_tokens=query_tokens,
            relevant=relevant,
            selection_mode=selection_mode,
        )

        if not selected_content:
            return "INSUFFICIENT_MEMORY"
        if ":" in selected_content:
            return selected_content.split(":", 1)[1].strip()
        return selected_content or "INSUFFICIENT_MEMORY"

    @classmethod
    def _selection_mode(
        cls, question: EffectivenessQuestion, query_tokens: Sequence[str]
    ) -> str:
        category = str(question.category or "").strip().lower()
        token_set = set(query_tokens)
        if category == "temporal_previous" or any(token in cls._PAST_CUES for token in token_set):
            return "oldest"
        if category in {"temporal", "update"} or any(
            token in cls._LATEST_CUES for token in token_set
        ):
            return "newest"
        return "newest"

    @classmethod
    def _select_content(
        cls,
        *,
        query_tokens: Sequence[str],
        relevant: Sequence[tuple[int, datetime, Dict[str, Any]]],
        selection_mode: str,
    ) -> str:
        if not relevant:
            return ""

        grouped: Dict[str, List[tuple[datetime, str]]] = {}
        for _, created_at, memory in relevant:
            content = str(memory.get("content") or "").strip()
            key = cls._prefix_key(content)
            grouped.setdefault(key, []).append((created_at, content))

        if grouped:
            best_score = None
            candidate_items: List[tuple[datetime, str]] = []
            for key, values in grouped.items():
                key_tokens = set(_normalize(key).replace(":", " ").split())
                score = len(key_tokens & set(query_tokens))
                if best_score is None or score > best_score:
                    best_score = score
                    candidate_items = list(values)
                elif score == best_score:
                    candidate_items.extend(values)
        else:
            candidate_items = [
                (created_at, str(memory.get("content") or "").strip())
                for _, created_at, memory in relevant
            ]

        if not candidate_items:
            return ""
        chooser = min if selection_mode == "oldest" else max
        return chooser(candidate_items, key=lambda item: item[0])[1]

    @staticmethod
    def _prefix_key(content: str) -> str:
        return content.split(":", 1)[0].strip().lower() if ":" in content else content.lower()

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

        heuristic_answer = self._fallback.answer(question, retrieved_memories=retrieved_memories)
        ordered_memories = sorted(
            retrieved_memories, key=HeuristicMemoryAnswerRunner._created_at
        )
        memories = []
        for index, memory in enumerate(ordered_memories, start=1):
            created_at = memory.get("created_at") or f"rank_{index}"
            memories.append(
                f"{index}. time={created_at} | content={memory.get('content', '')}"
            )
        system = (
            "Answer the user question using only the provided memories. "
            "Each memory includes a timestamp-like order signal. "
            "If the question asks about now/current/latest, use the newest relevant memory. "
            "If the question asks about first/original/initial/previous, use the earliest relevant memory. "
            "Prefer direct key-value memories over broad recap sentences when both are available. "
            "If the memories are insufficient or conflicting, reply exactly INSUFFICIENT_MEMORY. "
            "Return only the short final answer with no explanation."
        )
        user = f"Question: {question.query}\n\nMemories oldest-to-newest:\n" + "\n".join(memories)
        answer = (self._llm.complete(task_type="reasoning", system=system, user=user) or "").strip()
        if not answer or _normalize(answer) == _normalize("INSUFFICIENT_MEMORY"):
            return heuristic_answer
        return answer


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _signal_tokens(text: str) -> List[str]:
    stopwords = HeuristicMemoryAnswerRunner._STOPWORDS
    return [
        token
        for token in _normalize(text).replace("?", " ").replace(":", " ").split()
        if token and token not in stopwords
    ]


def _signal_overlap(query: str, content: str) -> int:
    return len(set(_signal_tokens(query)) & set(_signal_tokens(content)))


def _match_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in _normalize(text).replace(":", " ").replace(",", " ").split():
        token = raw.strip(" .;:!?()[]{}\"\'`")
        if token:
            tokens.append(token)
    return tokens


def _tokens_equivalent(left: str, right: str) -> bool:
    a = left.strip().lower()
    b = right.strip().lower()
    if not a or not b:
        return False
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    return len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]


def _is_answer_match(answer: str, acceptable_answers: Sequence[str]) -> bool:
    normalized_answer = _normalize(answer)
    if not normalized_answer:
        return False
    normalized_acceptable = [item for item in (_normalize(value) for value in acceptable_answers) if item]
    for expected in normalized_acceptable:
        if normalized_answer == expected or expected in normalized_answer or normalized_answer in expected:
            return True

    answer_tokens = _match_tokens(answer)
    if not answer_tokens:
        return False
    for expected in normalized_acceptable:
        expected_tokens = _match_tokens(expected)
        if not expected_tokens:
            continue
        matched = 0
        for expected_token in expected_tokens:
            if any(_tokens_equivalent(answer_token, expected_token) for answer_token in answer_tokens):
                matched += 1
        if matched / len(expected_tokens) >= 0.75:
            return True
    return False


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
    if question.category == "abstain":
        retrieved_hit = not any(_signal_overlap(question.query, content) >= 2 for content in retrieved_contents)
    else:
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
    acceptable = [question.expected_answer] + list(question.acceptable_answers or [])
    answer_correct = _is_answer_match(answer, acceptable)

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
    def _rate(items: Sequence[EffectivenessQuestionResult], attr: str) -> float:
        if not items:
            return 0.0
        return sum(1.0 if getattr(item, attr) else 0.0 for item in items) / len(items)

    def _bucket_summary(items: Sequence[EffectivenessQuestionResult]) -> Dict[str, Any]:
        return {
            "question_count": len(items),
            "retrieval_hit_rate": _rate(items, "retrieved_hit"),
            "answer_accuracy": _rate(items, "answer_correct"),
        }

    if not results:
        return {
            "question_count": 0,
            "retrieval_hit_rate": 0.0,
            "answer_accuracy": 0.0,
            "temporal_accuracy": 0.0,
            "update_accuracy": 0.0,
            "abstention_accuracy": 0.0,
            "scope_accuracy": 0.0,
            "multi_session_accuracy": 0.0,
            "category_breakdown": {},
            "case_breakdown": {},
        }

    temporal = [item for item in results if item.category in {"temporal", "temporal_previous"}]
    updates = [item for item in results if item.category == "update"]
    abstentions = [item for item in results if item.category == "abstain"]
    scoped = [item for item in results if item.category == "scope"]
    multi_session = [item for item in results if item.category == "multi_session"]

    category_breakdown: Dict[str, Dict[str, Any]] = {}
    for category in sorted({item.category for item in results}):
        bucket = [item for item in results if item.category == category]
        category_breakdown[category] = _bucket_summary(bucket)

    case_breakdown: Dict[str, Dict[str, Any]] = {}
    for case_id in sorted({item.case_id for item in results}):
        bucket = [item for item in results if item.case_id == case_id]
        case_breakdown[case_id] = _bucket_summary(bucket)

    return {
        "question_count": len(results),
        "retrieval_hit_rate": _rate(results, "retrieved_hit"),
        "answer_accuracy": _rate(results, "answer_correct"),
        "temporal_accuracy": _rate(temporal, "answer_correct"),
        "update_accuracy": _rate(updates, "answer_correct"),
        "abstention_accuracy": _rate(abstentions, "answer_correct"),
        "scope_accuracy": _rate(scoped, "answer_correct"),
        "multi_session_accuracy": _rate(multi_session, "answer_correct"),
        "category_breakdown": category_breakdown,
        "case_breakdown": case_breakdown,
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
