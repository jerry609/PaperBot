from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

from .evidence import EvidenceLinker, calibrate_confidence
from .models import (
    ExtractionObservation,
    StageName,
    StageResult,
    TaskCheckpoint,
    new_observation_id,
)

_evidence_linker = EvidenceLinker()


@dataclass
class StageInput:
    title: str
    abstract: str
    full_text: str
    sections: Dict[str, str]


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


class LiteratureDistillStage:
    name = StageName.LITERATURE_DISTILL.value

    async def run(self, data: StageInput) -> StageResult:
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

    async def run(self, data: StageInput) -> StageResult:
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
        warnings = []
        if architecture != "unknown" and not evidence:
            warnings.append("Blueprint extraction lacks direct evidence span for architecture.")
        return StageResult(observations=[obs], warnings=warnings)


class EnvironmentExtractStage:
    name = StageName.ENVIRONMENT_EXTRACT.value

    async def run(self, data: StageInput) -> StageResult:
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
        warnings = []
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

    async def run(self, data: StageInput) -> StageResult:
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
        warnings = []
        if not evidence:
            warnings.append("Spec extraction produced values without evidence spans.")
        return StageResult(observations=[obs], warnings=warnings)


class RoadmapPlanningStage:
    name = StageName.ROADMAP_PLANNING.value

    async def run(self, data: StageInput) -> StageResult:
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

    async def run(self, data: StageInput) -> StageResult:
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
        warnings = []
        if not evidence:
            warnings.append(
                "Success criteria metrics are inferred without explicit metric span evidence."
            )
        return StageResult(observations=[obs], warnings=warnings)
