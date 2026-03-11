from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.stores.llm_usage_store import LLMUsageStore


def test_llm_usage_store_records_and_summarizes(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'llm-usage.db'}"
    store = LLMUsageStore(db_url=db_url)

    store.record_usage(
        task_type="summary",
        provider_name="openai",
        model_name="gpt-4o-mini",
        prompt_tokens=120,
        completion_tokens=80,
        estimated_cost_usd=0.0001,
        metadata={"estimated": True},
    )
    store.record_usage(
        task_type="reasoning",
        provider_name="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        prompt_tokens=300,
        completion_tokens=200,
        estimated_cost_usd=0.001,
        metadata={"estimated": True},
    )

    summary = store.summarize(days=7)

    assert summary["totals"]["calls"] == 2
    assert summary["totals"]["total_tokens"] == 700
    assert len(summary["daily"]) >= 1
    assert len(summary["provider_models"]) == 2
    top = summary["provider_models"][0]
    assert top["provider_name"] in {"openai", "anthropic"}
