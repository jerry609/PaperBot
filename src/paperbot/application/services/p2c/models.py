from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import uuid4


class StageName(str, Enum):
    LITERATURE_DISTILL = "literature_distill"
    BLUEPRINT_EXTRACT = "blueprint_extract"
    ENVIRONMENT_EXTRACT = "environment_extract"
    SPEC_EXTRACT = "spec_extract"
    ROADMAP_PLANNING = "roadmap_planning"
    SUCCESS_CRITERIA = "success_criteria"


class PaperType(str, Enum):
    EXPERIMENTAL = "experimental"
    THEORETICAL = "theoretical"
    SURVEY = "survey"
    BENCHMARK = "benchmark"
    SYSTEM = "system"


Depth = Literal["fast", "standard", "deep"]
Source = Literal["manual", "recommendation", "collection"]
Difficulty = Literal["low", "medium", "high"]
EvidenceType = Literal["paper_span", "table", "figure", "code_snippet", "metadata"]


@dataclass
class GenerateContextRequest:
    """P2C pipeline entry request."""

    paper_id: str
    user_id: str = "default"
    project_id: Optional[str] = None
    track_id: Optional[int] = None
    depth: Depth = "standard"
    source: Source = "manual"


@dataclass
class PaperIdentity:
    paper_id: str = ""
    title: str = ""
    year: int = 0
    authors: List[str] = field(default_factory=list)
    identifiers: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskCheckpoint:
    id: str
    title: str
    description: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    estimated_difficulty: Difficulty = "medium"


@dataclass
class EvidenceLink:
    type: EvidenceType
    ref: str
    supports: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ConfidenceScores:
    overall: float = 0.0
    literature: float = 0.0
    blueprint: float = 0.0
    environment: float = 0.0
    spec: float = 0.0
    roadmap: float = 0.0
    metrics: float = 0.0


@dataclass
class ExtractionObservation:
    """
    Structured extraction record inspired by observation-style memory models.
    """

    id: str
    stage: str
    type: str
    title: str
    narrative: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceLink] = field(default_factory=list)
    confidence: float = 0.0
    concepts: List[str] = field(default_factory=list)

    def to_compact(self, max_tokens: int = 100) -> str:
        max_chars = max(1, max_tokens) * 4
        narrative = self.narrative.strip()
        if len(narrative) > max_chars:
            narrative = narrative[: max_chars - 3].rstrip() + "..."
        return f"[{self.type}] {self.title} (conf={self.confidence:.0%}): {narrative}"

    def to_full(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "type": self.type,
            "title": self.title,
            "narrative": self.narrative,
            "structured_data": self.structured_data,
            "evidence": [asdict(item) for item in self.evidence],
            "confidence": self.confidence,
            "concepts": self.concepts,
        }


def new_observation_id() -> str:
    return f"obs_{uuid4().hex[:12]}"


def new_context_pack_id() -> str:
    return f"ctxp_{uuid4().hex[:12]}"


@dataclass
class ReproContextPack:
    context_pack_id: str
    version: str = "v2"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    paper: PaperIdentity = field(default_factory=PaperIdentity)
    paper_type: PaperType = PaperType.EXPERIMENTAL
    objective: str = ""
    observations: List[ExtractionObservation] = field(default_factory=list)
    task_roadmap: List[TaskCheckpoint] = field(default_factory=list)
    confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    warnings: List[str] = field(default_factory=list)

    def get_by_stage(self, stage: str) -> List[ExtractionObservation]:
        return [obs for obs in self.observations if obs.stage == stage]

    def get_by_type(self, type_name: str) -> List[ExtractionObservation]:
        return [obs for obs in self.observations if obs.type == type_name]

    def get_by_concept(self, concept: str) -> List[ExtractionObservation]:
        return [obs for obs in self.observations if concept in obs.concepts]

    def to_compact_context(self, max_tokens: int = 2000) -> str:
        concept_priority = [
            "core_method",
            "architecture",
            "hyperparameter",
            "environment",
            "gotcha",
        ]

        lines = [f"# Reproduction Context: {self.paper.title} ({self.paper.year})"]
        if self.objective:
            lines.append(f"Objective: {self.objective}")

        budget = max(1, max_tokens) * 4
        used_ids: set[str] = set()

        for concept in concept_priority:
            candidates = sorted(
                self.get_by_concept(concept),
                key=lambda item: item.confidence,
                reverse=True,
            )
            for obs in candidates:
                if obs.id in used_ids:
                    continue
                compact = obs.to_compact(max_tokens=80)
                if len(compact) > budget:
                    return "\n".join(lines)
                lines.append(compact)
                used_ids.add(obs.id)
                budget -= len(compact)

        remaining = sorted(self.observations, key=lambda item: item.confidence, reverse=True)
        for obs in remaining:
            if obs.id in used_ids:
                continue
            compact = obs.to_compact(max_tokens=60)
            if len(compact) > budget:
                break
            lines.append(compact)
            budget -= len(compact)

        return "\n".join(lines)

    def to_execution_prompt(
        self, executor: Literal["claude_code", "codex", "local"] = "claude_code"
    ) -> str:
        prompt = self.to_compact_context(max_tokens=3000)
        if self.task_roadmap:
            prompt += "\n\n## Roadmap\n"
            for step in self.task_roadmap:
                acceptance = (
                    ", ".join(step.acceptance_criteria) if step.acceptance_criteria else "none"
                )
                prompt += f"- {step.id}: {step.title} (acceptance: {acceptance})\n"

        if executor == "codex":
            prompt += "\nFocus on concrete file-level steps and verifiable checkpoints.\n"
        elif executor == "claude_code":
            prompt += "\nExecute step-by-step and surface blockers early.\n"
        return prompt


@dataclass
class RawPaperData:
    paper_id: str
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    identifiers: Dict[str, str] = field(default_factory=dict)
    full_text: Optional[str] = None
    source_adapter: str = ""


@dataclass
class NormalizedInput:
    paper: PaperIdentity
    abstract: str = ""
    full_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    section_offsets: Dict[str, tuple[int, int]] = field(default_factory=dict)
    user_memory: Optional[str] = None
    project_context: Optional[str] = None

    @property
    def method_section(self) -> str:
        return self.sections.get("method", "")


@dataclass
class StageResult:
    observations: List[ExtractionObservation] = field(default_factory=list)
    roadmap: List[TaskCheckpoint] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def stage_mean_confidence(items: Sequence[ExtractionObservation]) -> float:
    if not items:
        return 0.0
    return float(sum(max(0.0, min(1.0, item.confidence)) for item in items) / len(items))
