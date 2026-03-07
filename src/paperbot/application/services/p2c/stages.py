from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .evidence import EvidenceLinker, calibrate_confidence
from .models import (
    ExtractionObservation,
    StageName,
    StageResult,
    TaskCheckpoint,
    new_observation_id,
)
from .prompts import (
    blueprint_extract_prompt,
    environment_extract_prompt,
    literature_distill_prompt,
    roadmap_planning_prompt,
    spec_extract_prompt,
    success_criteria_prompt,
)

if TYPE_CHECKING:
    from paperbot.application.services.llm_service import LLMService

logger = logging.getLogger(__name__)
_evidence_linker = EvidenceLinker()


@dataclass
class StageInput:
    title: str
    abstract: str
    full_text: str
    sections: Dict[str, str]
    user_memory: Optional[str] = None
    project_context: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Parse a JSON array from LLM output, tolerating markdown fences."""
    text = (raw or "").strip()
    if not text:
        return None
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return None
    except json.JSONDecodeError:
        pass
    # Try to find array brackets
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _llm_complete_async(llm: "LLMService", system: str, user: str, task_type: str) -> str:
    """Call LLMService.complete() — sync method, run in thread for async context."""
    return llm.complete(task_type=task_type, system=system, user=user)


def _obs_from_dict(
    raw: Dict[str, Any], *, stage: str, default_type: str = "method"
) -> ExtractionObservation:
    """Build an ExtractionObservation from an LLM-returned dict."""
    return ExtractionObservation(
        id=new_observation_id(),
        stage=stage,
        type=str(raw.get("type", default_type)),
        title=str(raw.get("title", "")),
        narrative=str(raw.get("narrative", "")),
        structured_data=raw.get("structured_data") or {},
        confidence=float(raw.get("confidence", 0.7)),
        concepts=raw.get("concepts") or [],
    )


# ---------------------------------------------------------------------------
# Heuristic helpers (kept for fallback)
# ---------------------------------------------------------------------------

def _first_sentence(text: str, fallback: str = "") -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return fallback
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return (parts[0] if parts else cleaned).strip() or fallback


def _detect_architecture(text: str) -> str:
    lower = text.lower()
    if "transformer" in lower:
        return "transformer"
    if "diffusion" in lower:
        return "diffusion"
    if "graph neural" in lower or "gnn" in lower:
        return "gnn"
    if "convolution" in lower or "cnn" in lower:
        return "cnn"
    if "lstm" in lower or "rnn" in lower:
        return "rnn"
    return "unknown"


# ===================================================================
# Stage classes — each tries LLM first, falls back to heuristic
# ===================================================================


class LiteratureDistillStage:
    name = StageName.LITERATURE_DISTILL.value

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("LiteratureDistillStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = literature_distill_prompt(
            data.title, data.abstract, data.sections, user_memory=data.user_memory
        )
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "extraction")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)
        observations = [_obs_from_dict(item, stage=self.name, default_type="method") for item in items]
        return StageResult(observations=observations)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        narrative = _first_sentence(
            data.abstract,
            fallback="No abstract summary detected; review the original manuscript.",
        )
        evidence = _evidence_linker.section_anchor(
            data.abstract,
            section="abstract",
            supports=["problem_definition"],
        )
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="method",
            title="Problem and core idea",
            narrative=narrative,
            structured_data={
                "problem_definition": narrative,
                "paper_title": data.title,
            },
            evidence=evidence,
            confidence=calibrate_confidence(0.55, evidence),
            concepts=["core_method"],
        )
        return StageResult(observations=[obs])


class BlueprintExtractStage:
    name = StageName.BLUEPRINT_EXTRACT.value

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("BlueprintExtractStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = blueprint_extract_prompt(data.title, data.abstract, data.sections)
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "extraction")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)
        observations = [
            _obs_from_dict(item, stage=self.name, default_type="architecture") for item in items
        ]
        return StageResult(observations=observations)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        source_text = f"{data.title}\n{data.abstract}\n{data.sections.get('method', '')}"
        architecture = _detect_architecture(source_text)
        evidence = (
            _evidence_linker.find_keyword_evidence(
                source_text,
                keywords=[architecture],
                section="method",
                supports=["architecture_type"],
                max_links=1,
            )
            if architecture != "unknown"
            else []
        )
        confidence = calibrate_confidence(0.6 if architecture != "unknown" else 0.3, evidence)
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="architecture",
            title=f"Architecture: {architecture}",
            narrative=f"Detected main architecture pattern as {architecture}.",
            structured_data={"architecture_type": architecture},
            evidence=evidence,
            confidence=confidence,
            concepts=["architecture"],
        )
        warnings: List[str] = []
        if architecture != "unknown" and not evidence:
            warnings.append("Blueprint extraction lacks direct evidence span for architecture.")
        return StageResult(observations=[obs], warnings=warnings)


class EnvironmentExtractStage:
    name = StageName.ENVIRONMENT_EXTRACT.value

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("EnvironmentExtractStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = environment_extract_prompt(data.title, data.abstract, data.sections)
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "extraction")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)
        observations = [
            _obs_from_dict(item, stage=self.name, default_type="environment") for item in items
        ]
        return StageResult(observations=observations)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        text = f"{data.abstract}\n{data.full_text}".lower()
        framework = (
            "pytorch"
            if "pytorch" in text or "torch" in text
            else "tensorflow" if "tensorflow" in text else "unknown"
        )
        python_version = "3.10" if framework != "unknown" else "3.11"
        evidence = (
            _evidence_linker.find_keyword_evidence(
                text,
                keywords=[framework],
                section="full_text",
                supports=["framework"],
                max_links=1,
            )
            if framework != "unknown"
            else []
        )
        confidence = calibrate_confidence(
            0.5 if framework != "unknown" else 0.35, evidence, required=True
        )
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="environment",
            title=f"Runtime env ({framework})",
            narrative=f"Inferred runtime stack: python {python_version}, framework {framework}.",
            structured_data={
                "python_version": python_version,
                "framework": framework,
            },
            evidence=evidence,
            confidence=confidence,
            concepts=["environment"],
        )
        warnings: List[str] = []
        if framework == "unknown":
            warnings.append("Environment extraction could not find explicit framework evidence.")
        return StageResult(observations=[obs], warnings=warnings)


class SpecExtractStage:
    name = StageName.SPEC_EXTRACT.value

    _regex_map = {
        "learning_rate": re.compile(
            r"(?:learning rate|lr)\s*[:=]?\s*([0-9.]+e-?\d+|0?\.\d+)",
            re.IGNORECASE,
        ),
        "batch_size": re.compile(r"batch size\s*[:=]?\s*(\d+)", re.IGNORECASE),
        "epochs": re.compile(r"(?:epoch|epochs)\s*[:=]?\s*(\d+)", re.IGNORECASE),
    }

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("SpecExtractStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = spec_extract_prompt(data.title, data.abstract, data.sections)
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "extraction")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)
        observations = [
            _obs_from_dict(item, stage=self.name, default_type="hyperparameter") for item in items
        ]
        return StageResult(observations=observations)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        blob = f"{data.abstract}\n{data.full_text}"
        extracted: Dict[str, str] = {}
        evidence = []
        for key, pattern in self._regex_map.items():
            match = pattern.search(blob)
            if match:
                extracted[key] = match.group(1)
                evidence.append(
                    _evidence_linker.from_match(
                        section="full_text",
                        supports=[key],
                        start=match.start(1),
                        end=match.end(1),
                    )
                )

        if not extracted:
            return StageResult(
                warnings=["Spec extraction found no explicit hyperparameters in text."],
            )

        narrative = ", ".join(f"{k}={v}" for k, v in extracted.items())
        confidence = calibrate_confidence(0.62, evidence, required=True)
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="hyperparameter",
            title="Key hyperparameters",
            narrative=f"Extracted hyperparameters: {narrative}.",
            structured_data=extracted,
            evidence=evidence,
            confidence=confidence,
            concepts=["hyperparameter", "gotcha"],
        )
        warnings: List[str] = []
        if not evidence:
            warnings.append("Spec extraction produced values without evidence spans.")
        return StageResult(observations=[obs], warnings=warnings)


class RoadmapPlanningStage:
    name = StageName.ROADMAP_PLANNING.value

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("RoadmapPlanningStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = roadmap_planning_prompt(
            data.title, data.abstract, data.sections, project_context=data.project_context
        )
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "reasoning")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)

        roadmap: List[TaskCheckpoint] = []
        for item in items:
            roadmap.append(
                TaskCheckpoint(
                    id=str(item.get("id", f"T{len(roadmap) + 1}")),
                    title=str(item.get("title", "")),
                    description=str(item.get("description", "")),
                    acceptance_criteria=item.get("acceptance_criteria") or [],
                    depends_on=item.get("depends_on") or [],
                    estimated_difficulty=item.get("estimated_difficulty", "medium"),
                )
            )

        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="roadmap",
            title="Paper-specific reproduction roadmap",
            narrative=f"Generated a {len(roadmap)}-step reproduction roadmap tailored to this paper.",
            structured_data={"task_count": len(roadmap)},
            confidence=0.78,
            concepts=["reproduction_hint"],
        )
        return StageResult(observations=[obs], roadmap=roadmap)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        roadmap = [
            TaskCheckpoint(
                id="T1",
                title="Prepare dataset and loaders",
                acceptance_criteria=["Training split loads without runtime errors"],
                estimated_difficulty="medium",
            ),
            TaskCheckpoint(
                id="T2",
                title="Implement model architecture",
                acceptance_criteria=["Forward pass shape checks pass"],
                depends_on=["T1"],
                estimated_difficulty="high",
            ),
            TaskCheckpoint(
                id="T3",
                title="Reproduce training loop",
                acceptance_criteria=["Loss decreases in smoke run"],
                depends_on=["T2"],
                estimated_difficulty="medium",
            ),
        ]
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="roadmap",
            title="Initial execution roadmap",
            narrative="Generated a three-step baseline roadmap for reproduction.",
            structured_data={"task_count": len(roadmap)},
            confidence=0.58,
            concepts=["reproduction_hint"],
        )
        return StageResult(observations=[obs], roadmap=roadmap)


class SuccessCriteriaStage:
    name = StageName.SUCCESS_CRITERIA.value

    _metric_re = re.compile(
        r"\b(accuracy|f1|bleu|rouge|mrr|ndcg|auc|wer|cer)\b[^\n.]{0,60}",
        re.IGNORECASE,
    )

    def __init__(self, llm: Optional["LLMService"] = None) -> None:
        self._llm = llm

    async def run(self, data: StageInput) -> StageResult:
        if self._llm is not None:
            try:
                return await self._run_llm(data)
            except Exception:  # noqa: BLE001 — network/API errors, graceful degradation
                logger.warning("SuccessCriteriaStage LLM failed, falling back to heuristic")
        return self._run_heuristic(data)

    async def _run_llm(self, data: StageInput) -> StageResult:
        system, user = success_criteria_prompt(data.title, data.abstract, data.sections)
        raw = await asyncio.to_thread(_llm_complete_async, self._llm, system, user, "extraction")
        items = _safe_parse_json_array(raw)
        if not items:
            return self._run_heuristic(data)
        observations = [
            _obs_from_dict(item, stage=self.name, default_type="metric") for item in items
        ]
        return StageResult(observations=observations)

    def _run_heuristic(self, data: StageInput) -> StageResult:
        blob = f"{data.abstract}\n{data.full_text}"
        matches = list(self._metric_re.finditer(blob))
        metrics = sorted({m.group(1).lower() for m in matches})
        evidence = [
            _evidence_linker.from_match(
                section="full_text",
                supports=["metrics"],
                start=match.start(1),
                end=match.end(1),
                confidence=0.81,
            )
            for match in matches[:5]
        ]
        if not metrics:
            metrics = ["accuracy"]

        confidence = calibrate_confidence(0.52, evidence, required=True)
        obs = ExtractionObservation(
            id=new_observation_id(),
            stage=self.name,
            type="metric",
            title="Success criteria",
            narrative=f"Track metrics: {', '.join(metrics)}.",
            structured_data={"metrics": metrics},
            evidence=evidence,
            confidence=confidence,
            concepts=["baseline"],
        )
        warnings: List[str] = []
        if not evidence:
            warnings.append(
                "Success criteria metrics are inferred without explicit metric span evidence."
            )
        return StageResult(observations=[obs], warnings=warnings)
