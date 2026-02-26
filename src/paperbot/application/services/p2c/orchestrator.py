from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Protocol

from .input_pipeline import PaperInputRouter, PaperSectionExtractor, PaperTypeClassifier
from .models import (
    ConfidenceScores,
    Depth,
    GenerateContextRequest,
    NormalizedInput,
    PaperType,
    RawPaperData,
    ReproContextPack,
    StageName,
    StageResult,
    new_context_pack_id,
    stage_mean_confidence,
)
from .stages import (
    BlueprintExtractStage,
    EnvironmentExtractStage,
    LiteratureDistillStage,
    RoadmapPlanningStage,
    SpecExtractStage,
    StageInput,
    SuccessCriteriaStage,
)


class P2CStage(Protocol):
    name: str

    async def run(self, data: StageInput) -> StageResult: ...


_STAGE_TO_CONFIDENCE_FIELD = {
    StageName.LITERATURE_DISTILL.value: "literature",
    StageName.BLUEPRINT_EXTRACT.value: "blueprint",
    StageName.ENVIRONMENT_EXTRACT.value: "environment",
    StageName.SPEC_EXTRACT.value: "spec",
    StageName.ROADMAP_PLANNING.value: "roadmap",
    StageName.SUCCESS_CRITERIA.value: "metrics",
}


StageCompleteCallback = Callable[
    [str, List["ExtractionObservation"], List[str]],
    Optional[Awaitable[None]],
]


@dataclass
class OrchestratorConfig:
    default_depth: Depth = "standard"


class ExtractionOrchestrator:
    def __init__(
        self,
        stages: Optional[Iterable[P2CStage]] = None,
        *,
        input_router: Optional[PaperInputRouter] = None,
        section_extractor: Optional[PaperSectionExtractor] = None,
        paper_type_classifier: Optional[PaperTypeClassifier] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        active_stages = list(
            stages
            or [
                LiteratureDistillStage(),
                BlueprintExtractStage(),
                EnvironmentExtractStage(),
                SpecExtractStage(),
                RoadmapPlanningStage(),
                SuccessCriteriaStage(),
            ]
        )
        self._stages: Dict[str, P2CStage] = {stage.name: stage for stage in active_stages}
        self._input_router = input_router or PaperInputRouter()
        self._section_extractor = section_extractor or PaperSectionExtractor()
        self._paper_type_classifier = paper_type_classifier or PaperTypeClassifier()
        self._config = config or OrchestratorConfig()

    @staticmethod
    def resolve_stage_sequence(depth: Depth, paper_type: PaperType) -> List[str]:
        if depth == "fast":
            stage_names = [
                StageName.BLUEPRINT_EXTRACT.value,
                StageName.ENVIRONMENT_EXTRACT.value,
            ]
        else:
            stage_names = [
                StageName.LITERATURE_DISTILL.value,
                StageName.BLUEPRINT_EXTRACT.value,
                StageName.ENVIRONMENT_EXTRACT.value,
                StageName.SPEC_EXTRACT.value,
                StageName.ROADMAP_PLANNING.value,
                StageName.SUCCESS_CRITERIA.value,
            ]

        if paper_type == PaperType.THEORETICAL:
            return [
                s
                for s in stage_names
                if s
                not in {
                    StageName.ENVIRONMENT_EXTRACT.value,
                    StageName.SPEC_EXTRACT.value,
                }
            ]

        if paper_type == PaperType.SURVEY:
            return [StageName.LITERATURE_DISTILL.value, StageName.SUCCESS_CRITERIA.value]

        if paper_type == PaperType.BENCHMARK:
            ordered = [
                StageName.LITERATURE_DISTILL.value,
                StageName.BLUEPRINT_EXTRACT.value,
                StageName.SPEC_EXTRACT.value,
                StageName.SUCCESS_CRITERIA.value,
                StageName.ROADMAP_PLANNING.value,
            ]
            return [name for name in ordered if name in stage_names]

        if paper_type == PaperType.SYSTEM:
            ordered = [
                StageName.LITERATURE_DISTILL.value,
                StageName.ENVIRONMENT_EXTRACT.value,
                StageName.BLUEPRINT_EXTRACT.value,
                StageName.ROADMAP_PLANNING.value,
                StageName.SUCCESS_CRITERIA.value,
            ]
            return [name for name in ordered if name in stage_names]

        return stage_names

    async def run(
        self,
        request: GenerateContextRequest,
        *,
        raw_paper: Optional[RawPaperData] = None,
        normalized_input: Optional[NormalizedInput] = None,
        on_stage_complete: Optional[StageCompleteCallback] = None,
    ) -> ReproContextPack:
        depth = request.depth or self._config.default_depth

        if normalized_input is None:
            if raw_paper is None:
                raw_paper = await self._input_router.fetch(request.paper_id)
            normalized_input = await self._section_extractor.extract(raw_paper)

        paper_type = self._paper_type_classifier.classify(normalized_input)
        pack = ReproContextPack(
            context_pack_id=new_context_pack_id(),
            paper=normalized_input.paper,
            paper_type=paper_type,
            objective=f"Reproduce core claims of {normalized_input.paper.title}.",
        )

        stage_order = self.resolve_stage_sequence(depth, paper_type)
        stage_scores: Dict[str, float] = {}
        stage_input = StageInput(
            title=normalized_input.paper.title,
            abstract=normalized_input.abstract,
            full_text=normalized_input.full_text,
            sections=normalized_input.sections,
        )

        for stage_name in stage_order:
            stage = self._stages.get(stage_name)
            if stage is None:
                pack.warnings.append(f"Stage {stage_name} is not registered.")
                continue

            result = await stage.run(stage_input)
            if result.observations:
                pack.observations.extend(result.observations)
                stage_scores[stage_name] = stage_mean_confidence(result.observations)
            if result.roadmap:
                pack.task_roadmap.extend(result.roadmap)
            if result.warnings:
                pack.warnings.extend(result.warnings)
            await self._emit_stage_complete(
                on_stage_complete,
                stage_name=stage_name,
                observations=result.observations,
                warnings=result.warnings,
            )

        if depth == "deep":
            await self._run_deep_verification(stage_order, stage_input, stage_scores, pack)
            pack.observations.sort(key=lambda item: item.confidence, reverse=True)

        self._apply_confidence(pack.confidence, stage_scores)
        if depth == "deep":
            available_stage_scores = [score for score in stage_scores.values() if score > 0]
            pack.confidence.overall = (
                float(sum(available_stage_scores) / len(available_stage_scores))
                if available_stage_scores
                else 0.0
            )
        elif pack.observations:
            pack.confidence.overall = stage_mean_confidence(pack.observations)
        return pack

    @staticmethod
    def _apply_confidence(confidence: ConfidenceScores, stage_scores: Dict[str, float]) -> None:
        for stage_name, score in stage_scores.items():
            field = _STAGE_TO_CONFIDENCE_FIELD.get(stage_name)
            if field:
                setattr(confidence, field, score)

    async def _run_deep_verification(
        self,
        stage_order: List[str],
        stage_input: StageInput,
        stage_scores: Dict[str, float],
        pack: ReproContextPack,
    ) -> None:
        verification_targets = [
            StageName.BLUEPRINT_EXTRACT.value,
            StageName.SPEC_EXTRACT.value,
            StageName.SUCCESS_CRITERIA.value,
        ]
        for stage_name in verification_targets:
            if stage_name not in stage_order:
                continue
            stage = self._stages.get(stage_name)
            if stage is None:
                continue

            verification = await stage.run(stage_input)
            baseline = stage_scores.get(stage_name, 0.0)
            if not verification.observations:
                if baseline > 0:
                    stage_scores[stage_name] = baseline * 0.85
                    pack.warnings.append(
                        f"Deep verification found no observations for stage {stage_name}; confidence reduced."
                    )
                continue

            verified_score = stage_mean_confidence(verification.observations)
            if baseline <= 0:
                stage_scores[stage_name] = verified_score * 0.95
                continue

            if abs(verified_score - baseline) > 0.1:
                pack.warnings.append(
                    f"Deep verification mismatch on stage {stage_name}: baseline={baseline:.2f}, verified={verified_score:.2f}."
                )

            stage_scores[stage_name] = min(baseline, verified_score) * 0.95

    @staticmethod
    async def _emit_stage_complete(
        callback: Optional[StageCompleteCallback],
        *,
        stage_name: str,
        observations: List["ExtractionObservation"],
        warnings: List[str],
    ) -> None:
        if callback is None:
            return
        result = callback(stage_name, observations, warnings)
        if inspect.isawaitable(result):
            await result
