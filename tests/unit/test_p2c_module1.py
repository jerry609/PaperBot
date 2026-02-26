from __future__ import annotations

import pytest

from paperbot.application.services.p2c import (
    ExtractionObservation,
    ExtractionOrchestrator,
    GenerateContextRequest,
    NormalizedInput,
    PaperIdentity,
    PaperType,
    PaperTypeClassifier,
    RawPaperData,
    ReproContextPack,
)


def test_repro_context_pack_compact_context_prioritizes_core_method():
    pack = ReproContextPack(
        context_pack_id="ctxp_test",
        paper=PaperIdentity(paper_id="p1", title="Test Paper", year=2026),
        objective="Reproduce key claim",
        observations=[
            ExtractionObservation(
                id="obs_a",
                stage="literature_distill",
                type="method",
                title="Core method",
                narrative="A" * 60,
                confidence=0.9,
                concepts=["core_method"],
            ),
            ExtractionObservation(
                id="obs_b",
                stage="spec_extract",
                type="hyperparameter",
                title="Hyperparameters",
                narrative="B" * 60,
                confidence=0.7,
                concepts=["hyperparameter"],
            ),
        ],
    )

    compact = pack.to_compact_context(max_tokens=200)
    assert "Core method" in compact
    assert compact.index("Core method") < compact.index("Hyperparameters")


def test_paper_type_classifier_identifies_theoretical_text():
    classifier = PaperTypeClassifier()
    normalized = NormalizedInput(
        paper=PaperIdentity(paper_id="p2", title="Convergence Theorem for XYZ"),
        abstract="We provide proofs and bounds for optimization stability.",
        full_text="",
        sections={},
    )

    assert classifier.classify(normalized) == PaperType.THEORETICAL


def test_resolve_stage_sequence_respects_depth_and_type():
    seq_fast = ExtractionOrchestrator.resolve_stage_sequence("fast", PaperType.EXPERIMENTAL)
    assert seq_fast == ["blueprint_extract", "environment_extract"]

    seq_theoretical = ExtractionOrchestrator.resolve_stage_sequence(
        "standard", PaperType.THEORETICAL
    )
    assert "environment_extract" not in seq_theoretical
    assert "spec_extract" not in seq_theoretical


@pytest.mark.asyncio
async def test_orchestrator_run_builds_context_pack_from_raw_text():
    orchestrator = ExtractionOrchestrator()
    request = GenerateContextRequest(paper_id="local:test", depth="standard")

    raw_paper = RawPaperData(
        paper_id="local:test",
        title="Transformer Benchmark Study",
        abstract="We introduce a transformer model and report accuracy improvements.",
        year=2025,
        full_text=(
            "Introduction\n"
            "This paper studies benchmark behavior.\n"
            "Method\n"
            "Our model uses transformer blocks. Learning rate 1e-4, batch size 32, epochs 10.\n"
            "Results\n"
            "Accuracy reaches 92.1 on test split.\n"
        ),
        source_adapter="local_file",
    )

    pack = await orchestrator.run(request, raw_paper=raw_paper)

    assert pack.paper.title == "Transformer Benchmark Study"
    assert pack.observations
    assert any(obs.type == "architecture" for obs in pack.observations)
    assert any(obs.type == "metric" for obs in pack.observations)
    assert pack.confidence.overall > 0
