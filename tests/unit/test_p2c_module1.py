from __future__ import annotations

import pytest

from paperbot.application.services.p2c import (
    ArXivAdapter,
    EvidenceLink,
    ExtractionObservation,
    ExtractionOrchestrator,
    GenerateContextRequest,
    NormalizedInput,
    PaperIdentity,
    PaperInputRouter,
    PaperSectionExtractor,
    PaperType,
    PaperTypeClassifier,
    RawPaperData,
    ReproContextPack,
    SemanticScholarAdapter,
    calibrate_confidence,
)
from paperbot.application.services.p2c.stages import (
    SpecExtractStage,
    StageInput,
    SuccessCriteriaStage,
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


@pytest.mark.asyncio
async def test_semantic_scholar_adapter_fetch_maps_metadata_from_client():
    class _FakeS2Client:
        def __init__(self):
            self.calls = []

        async def get_paper(self, paper_id, fields=None):
            self.calls.append((paper_id, fields))
            return {
                "paperId": "abcdef123456",
                "title": "Attention Is All You Need",
                "abstract": "We propose the Transformer architecture.",
                "year": 2017,
                "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
                "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
            }

        async def close(self):
            return None

    fake_client = _FakeS2Client()
    adapter = SemanticScholarAdapter(client=fake_client)
    result = await adapter.fetch("https://doi.org/10.48550/ARXIV.1706.03762")

    assert fake_client.calls
    assert fake_client.calls[0][0] == "DOI:10.48550/arxiv.1706.03762"
    assert result.paper_id == "s2:abcdef123456"
    assert result.title == "Attention Is All You Need"
    assert result.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert result.identifiers["semantic_scholar"] == "abcdef123456"
    assert result.identifiers["doi"] == "10.48550/arxiv.1706.03762"
    assert result.identifiers["arxiv"] == "1706.03762"


@pytest.mark.asyncio
async def test_arxiv_adapter_fetch_maps_metadata_from_atom_feed():
    async def _fake_atom_feed(_: str) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v1</id>
    <title>Attention Is All You Need</title>
    <summary>Transformer based sequence modeling.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <updated>2017-06-12T00:00:00Z</updated>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link rel="alternate" type="text/html" href="https://arxiv.org/abs/1706.03762v1" />
    <link rel="related" type="application/pdf" href="https://arxiv.org/pdf/1706.03762v1.pdf" />
  </entry>
</feed>
"""

    adapter = ArXivAdapter(fetch_atom_xml=_fake_atom_feed)
    result = await adapter.fetch("https://arxiv.org/abs/1706.03762")

    assert result.paper_id == "arxiv:1706.03762v1"
    assert result.title == "Attention Is All You Need"
    assert result.abstract == "Transformer based sequence modeling."
    assert result.year == 2017
    assert result.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert result.identifiers["arxiv"] == "1706.03762v1"


@pytest.mark.asyncio
async def test_input_router_dispatches_to_arxiv_and_semantic_scholar_adapters():
    async def _fake_atom_feed(_: str) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>ArXiv Sample</title>
    <summary>Sample abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-01-01T00:00:00Z</updated>
    <author><name>Author A</name></author>
    <link rel="alternate" type="text/html" href="https://arxiv.org/abs/2401.00001v2" />
  </entry>
</feed>
"""

    class _FakeS2Client:
        async def get_paper(self, paper_id, fields=None):
            if paper_id != "hash123":
                return {}
            return {
                "paperId": "hash123",
                "title": "S2 Sample",
                "abstract": "S2 abstract",
                "year": 2025,
                "authors": [{"name": "Author B"}],
                "externalIds": {},
            }

        async def close(self):
            return None

    router = PaperInputRouter(
        adapters=[
            ArXivAdapter(fetch_atom_xml=_fake_atom_feed),
            SemanticScholarAdapter(client=_FakeS2Client()),
        ]
    )

    arxiv_raw = await router.fetch("arXiv:2401.00001")
    s2_raw = await router.fetch("s2:hash123")

    assert arxiv_raw.source_adapter == "arxiv"
    assert arxiv_raw.title == "ArXiv Sample"
    assert s2_raw.source_adapter == "semantic_scholar"
    assert s2_raw.title == "S2 Sample"


@pytest.mark.asyncio
async def test_orchestrator_stage_callback_emits_stage_payload():
    orchestrator = ExtractionOrchestrator()
    request = GenerateContextRequest(paper_id="local:cb", depth="standard")
    events: list[tuple[str, int, int]] = []

    async def _on_stage_complete(stage_name, observations, warnings):
        events.append((stage_name, len(observations), len(warnings)))

    raw_paper = RawPaperData(
        paper_id="local:cb",
        title="Experimental Transformer Study",
        abstract="We present a transformer model with improved accuracy.",
        year=2026,
        full_text=(
            "Method\n"
            "Transformer blocks with learning rate 1e-4 and batch size 16 for 5 epochs.\n"
            "Results\n"
            "Accuracy improves to 90.2.\n"
        ),
        source_adapter="local_file",
    )

    await orchestrator.run(request, raw_paper=raw_paper, on_stage_complete=_on_stage_complete)

    expected_order = ExtractionOrchestrator.resolve_stage_sequence(
        "standard", PaperType.EXPERIMENTAL
    )
    assert [item[0] for item in events] == expected_order
    assert all(obs_count >= 0 for _, obs_count, _ in events)


@pytest.mark.asyncio
async def test_deep_mode_uses_stricter_confidence_than_standard():
    orchestrator = ExtractionOrchestrator()
    raw_paper = RawPaperData(
        paper_id="local:deep",
        title="Experimental Transformer Study",
        abstract="We present a transformer model with improved accuracy.",
        year=2026,
        full_text=(
            "Method\n"
            "Transformer blocks with learning rate 1e-4 and batch size 16 for 5 epochs.\n"
            "Results\n"
            "Accuracy improves to 90.2.\n"
        ),
        source_adapter="local_file",
    )

    standard = await orchestrator.run(
        GenerateContextRequest(paper_id="local:deep", depth="standard"),
        raw_paper=raw_paper,
    )
    deep = await orchestrator.run(
        GenerateContextRequest(paper_id="local:deep", depth="deep"),
        raw_paper=raw_paper,
    )

    assert deep.confidence.overall < standard.confidence.overall


def test_calibrate_confidence_penalizes_missing_required_evidence():
    with_evidence = calibrate_confidence(
        0.6,
        [
            EvidenceLink(
                type="paper_span",
                ref="method#char:1-10",
                supports=["learning_rate"],
                confidence=0.9,
            )
        ],
        required=True,
    )
    without_evidence = calibrate_confidence(0.6, [], required=True)
    assert with_evidence > without_evidence


@pytest.mark.asyncio
async def test_spec_extract_emits_evidence_links_for_hyperparameters():
    stage = SpecExtractStage()
    result = await stage.run(
        StageInput(
            title="Hyperparam Test",
            abstract="",
            full_text="Learning rate 1e-4, batch size 32, epochs 8.",
            sections={},
        )
    )

    assert result.observations
    obs = result.observations[0]
    assert obs.type == "hyperparameter"
    assert obs.evidence
    assert all(link.supports for link in obs.evidence)


@pytest.mark.asyncio
async def test_success_criteria_confidence_is_lower_without_metric_spans():
    stage = SuccessCriteriaStage()
    with_metrics = await stage.run(
        StageInput(
            title="Metric Rich",
            abstract="",
            full_text="Results show accuracy improves and F1 reaches 0.80.",
            sections={},
        )
    )
    without_metrics = await stage.run(
        StageInput(
            title="Metric Sparse",
            abstract="",
            full_text="Results are promising according to qualitative feedback.",
            sections={},
        )
    )

    with_conf = with_metrics.observations[0].confidence
    without_conf = without_metrics.observations[0].confidence
    assert with_conf > without_conf
    assert without_metrics.warnings


@pytest.mark.asyncio
async def test_section_extractor_handles_numbered_headings_and_offsets():
    extractor = PaperSectionExtractor()
    raw = RawPaperData(
        paper_id="local:sec1",
        title="Sectioned",
        full_text=(
            "1 Introduction\n"
            "Intro text.\n"
            "2 Methodology\n"
            "We use transformer encoder blocks.\n"
            "3 Experiments\n"
            "Evaluation on benchmark datasets.\n"
            "4 Conclusion\n"
            "Final remarks.\n"
        ),
    )

    normalized = await extractor.extract(raw)
    assert "method" in normalized.sections
    assert "results" in normalized.sections
    assert "discussion" in normalized.sections
    start, end = normalized.section_offsets["method"]
    assert start < end
    assert "transformer encoder" in normalized.full_text[start:end].lower()


@pytest.mark.asyncio
async def test_section_extractor_handles_uppercase_and_colon_headings():
    extractor = PaperSectionExtractor()
    raw = RawPaperData(
        paper_id="local:sec2",
        title="Uppercase headings",
        full_text=(
            "I. BACKGROUND\n"
            "Background details.\n"
            "II. METHODS:\n"
            "Method details.\n"
            "III. EVALUATION\n"
            "Evaluation details.\n"
            "IV. FUTURE WORK\n"
            "Discussion details.\n"
        ),
    )

    normalized = await extractor.extract(raw)
    assert "introduction" in normalized.sections
    assert "method" in normalized.sections
    assert "results" in normalized.sections
    assert "discussion" in normalized.sections
    assert all(section in normalized.section_offsets for section in normalized.sections)


@pytest.mark.asyncio
async def test_section_extractor_fallback_segments_noisy_plain_text():
    extractor = PaperSectionExtractor()
    raw = RawPaperData(
        paper_id="local:sec3",
        title="Noisy plain text",
        full_text=(
            "This paper introduction explains motivation and setup. "
            "Our method uses retrieval augmentation with a lightweight transformer. "
            "Experiments show strong results on two benchmarks. "
            "Conclusion discusses limitations and future work."
        ),
    )

    normalized = await extractor.extract(raw)
    assert "method" in normalized.sections
    assert "results" in normalized.sections
    assert normalized.section_offsets.get("method")
