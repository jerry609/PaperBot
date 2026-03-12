from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.services.document_evidence_benchmark import (  # noqa: E402
    format_document_evidence_benchmark_report,
    load_document_evidence_benchmark_fixture,
    run_document_evidence_benchmark,
)
from paperbot.context_engine.embeddings import (  # noqa: E402
    EmbeddingConfig,
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    try_build_default_embedding_provider,
)


def _build_embedding_provider(
    provider_name: str,
    embedding_model: str,
) -> Tuple[Optional[EmbeddingProvider], str]:
    normalized = str(provider_name or "hash").strip().lower()
    config = EmbeddingConfig(model=str(embedding_model or "").strip() or None)
    if normalized == "hash":
        return HashEmbeddingProvider(dim=128), "hash"
    if normalized == "openai":
        return OpenAIEmbeddingProvider(config=config), f"openai:{config.resolve_model()}"
    if normalized == "default":
        provider = try_build_default_embedding_provider(config=config)
        label = provider.__class__.__name__.lower() if provider is not None else "none"
        return provider, label
    if normalized == "none":
        return None, "none"
    raise SystemExit(f"Unsupported embedding provider: {provider_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the document evidence benchmark.")
    parser.add_argument(
        "--fixtures",
        default="evals/fixtures/document_evidence/bench_v1.json",
        help="Path to the document evidence benchmark fixture JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/reports/document_evidence_bench_v1.json",
        help="Optional output path for the benchmark report JSON.",
    )
    parser.add_argument(
        "--modes",
        default="fts_only,embedding_only,hybrid",
        help="Comma-separated retrieval modes to run.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="hash",
        help="Embedding provider for indexing/query evaluation: hash, openai, default, or none.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Optional embedding model override for openai/default providers.",
    )
    parser.add_argument(
        "--fail-under-hybrid-recall",
        type=float,
        default=0.0,
        help="Fail if hybrid recall drops below this threshold.",
    )
    parser.add_argument(
        "--fail-under-hybrid-hit-rate",
        type=float,
        default=0.0,
        help="Fail if hybrid evidence hit rate drops below this threshold.",
    )
    args = parser.parse_args()

    fixture = load_document_evidence_benchmark_fixture(REPO_ROOT / args.fixtures)
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    embedding_provider, provider_label = _build_embedding_provider(
        args.embedding_provider,
        args.embedding_model,
    )
    if embedding_provider is None and any(mode in {"embedding_only", "hybrid"} for mode in modes):
        raise SystemExit(
            "Embedding provider is disabled, but embedding_only/hybrid modes were requested."
        )
    result = run_document_evidence_benchmark(
        fixture,
        modes=modes,
        embedding_provider=embedding_provider,
        provider_label=provider_label,
    )
    print(format_document_evidence_benchmark_report(result))

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    hybrid = (result.get("summary") or {}).get("hybrid", {}).get("overall", {})
    top_k = 5
    for key in hybrid.keys():
        if key.startswith("recall_at_"):
            top_k = int(key.rsplit("_", 1)[-1])
            break
    hybrid_recall = float(hybrid.get(f"recall_at_{top_k}", 0.0))
    hybrid_hit_rate = float(hybrid.get("evidence_hit_rate", 0.0))

    if hybrid_recall < float(args.fail_under_hybrid_recall):
        raise SystemExit(
            f"hybrid recall dropped below threshold: {hybrid_recall:.3f} < {args.fail_under_hybrid_recall:.3f}"
        )
    if hybrid_hit_rate < float(args.fail_under_hybrid_hit_rate):
        raise SystemExit(
            f"hybrid evidence hit rate dropped below threshold: {hybrid_hit_rate:.3f} < {args.fail_under_hybrid_hit_rate:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
