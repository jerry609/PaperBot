"""Tests for push channel formatters (Telegram, Discord, WeCom, Feishu)."""
from paperbot.infrastructure.push.formatters import (
    get_formatter,
    list_formatters,
)
from paperbot.infrastructure.push.formatters.telegram import TelegramFormatter
from paperbot.infrastructure.push.formatters.discord import DiscordFormatter
from paperbot.infrastructure.push.formatters.wecom import WeComFormatter
from paperbot.infrastructure.push.formatters.feishu import FeishuFormatter


def _sample_report():
    return {
        "title": "DailyPaper Digest",
        "date": "2026-03-02",
        "stats": {"unique_items": 5, "total_query_hits": 12},
        "queries": [
            {
                "normalized_query": "KV cache",
                "total_hits": 3,
                "top_items": [
                    {
                        "title": "FlashKV: Efficient KV Cache Compression",
                        "url": "https://arxiv.org/abs/2601.00001",
                        "score": 9.5,
                        "snippet": "We propose a method to compress KV cache",
                        "judge": {
                            "overall": 4.5,
                            "recommendation": "must_read",
                            "one_line_summary": "Novel KV compression achieving 2x speedup",
                            "relevance": {"score": 5, "rationale": "Direct match"},
                            "novelty": {"score": 4, "rationale": "New approach"},
                            "rigor": {"score": 4, "rationale": ""},
                            "impact": {"score": 5, "rationale": ""},
                            "clarity": {"score": 4, "rationale": ""},
                        },
                        "digest_card": {
                            "highlight": "2x faster inference via KV cache pruning",
                            "method": "Attention-aware pruning with dynamic threshold",
                            "finding": "Maintains 99% accuracy with 50% cache reduction",
                            "tags": ["KV Cache", "Efficiency", "LLM"],
                        },
                        "main_figure": {
                            "url": "https://cdn.example.com/fig1.png",
                            "caption": "System architecture overview",
                        },
                    },
                    {
                        "title": "PagedAttention v2",
                        "url": "https://arxiv.org/abs/2601.00002",
                        "score": 8.0,
                        "snippet": "Improved paged attention for LLM serving",
                        "judge": {
                            "overall": 3.8,
                            "recommendation": "worth_reading",
                            "one_line_summary": "Incremental paged attention improvement",
                        },
                        "digest_card": {
                            "highlight": "15% throughput gain in LLM serving",
                            "method": "Dynamic page allocation",
                            "finding": "Better memory utilization",
                            "tags": ["Serving", "Attention"],
                        },
                    },
                ],
            }
        ],
        "global_top": [],
        "llm_analysis": {
            "daily_insight": "Today's papers focus on efficient inference with KV cache optimization.",
            "features": ["summary", "digest_card"],
        },
    }


# ── Registry tests ────────────────────────────────────────────

def test_list_formatters():
    names = list_formatters()
    assert "telegram" in names
    assert "discord" in names
    assert "wecom" in names
    assert "feishu" in names


def test_get_formatter_returns_correct_type():
    assert isinstance(get_formatter("telegram"), TelegramFormatter)
    assert isinstance(get_formatter("discord"), DiscordFormatter)
    assert isinstance(get_formatter("wecom"), WeComFormatter)
    assert isinstance(get_formatter("feishu"), FeishuFormatter)


def test_get_formatter_unknown_returns_none():
    assert get_formatter("nonexistent") is None


# ── Telegram tests ────────────────────────────────────────────

def test_telegram_format_digest():
    fmt = TelegramFormatter()
    assert fmt.channel_type == "telegram"
    result = fmt.format_digest(_sample_report())

    assert "text" in result
    assert "parse_mode" in result
    assert result["parse_mode"] == "MarkdownV2"
    assert "FlashKV" in result["text"]
    assert "photo_url" in result
    assert result["photo_url"] == "https://cdn.example.com/fig1.png"


def test_telegram_inline_keyboard():
    fmt = TelegramFormatter()
    result = fmt.format_digest(_sample_report())
    assert "inline_keyboard" in result
    assert len(result["inline_keyboard"]) >= 1


def test_telegram_escapes_special_chars():
    fmt = TelegramFormatter()
    report = _sample_report()
    report["queries"][0]["top_items"][0]["title"] = "Test [with] special (chars)"
    result = fmt.format_digest(report)
    # Should not crash and should contain escaped text
    assert "text" in result


# ── Discord tests ─────────────────────────────────────────────

def test_discord_format_digest():
    fmt = DiscordFormatter()
    assert fmt.channel_type == "discord"
    result = fmt.format_digest(_sample_report())

    assert "embeds" in result
    embed = result["embeds"][0]
    assert embed["title"] == "📄 DailyPaper Digest"
    assert embed["color"] == 0x2563EB
    assert len(embed["fields"]) == 2
    assert "FlashKV" in embed["fields"][0]["name"]
    assert "thumbnail" in embed


def test_discord_embed_fields_have_values():
    fmt = DiscordFormatter()
    result = fmt.format_digest(_sample_report())
    for field in result["embeds"][0]["fields"]:
        assert "name" in field
        assert "value" in field
        assert len(field["name"]) <= 256
        assert len(field["value"]) <= 1024


# ── WeCom tests ───────────────────────────────────────────────

def test_wecom_format_digest():
    fmt = WeComFormatter()
    assert fmt.channel_type == "wecom"
    result = fmt.format_digest(_sample_report())

    assert "markdown" in result
    assert "news" in result
    assert "content" in result["markdown"]
    assert "FlashKV" in result["markdown"]["content"]
    assert len(result["news"]["articles"]) > 0


def test_wecom_news_articles_have_required_fields():
    fmt = WeComFormatter()
    result = fmt.format_digest(_sample_report())
    for article in result["news"]["articles"]:
        assert "title" in article
        assert "description" in article
        assert "url" in article


# ── Feishu tests ──────────────────────────────────────────────

def test_feishu_format_digest():
    fmt = FeishuFormatter()
    assert fmt.channel_type == "feishu"
    result = fmt.format_digest(_sample_report())

    assert "interactive" in result
    assert "post" in result

    card = result["interactive"]["card"]
    assert card["header"]["title"]["content"] == "📄 DailyPaper Digest"
    assert len(card["elements"]) > 0

    post = result["post"]
    assert "zh_cn" in post
    assert len(post["zh_cn"]["content"]) > 0


def test_feishu_card_has_action_buttons():
    fmt = FeishuFormatter()
    result = fmt.format_digest(_sample_report())
    card = result["interactive"]["card"]
    # At least one element should have an "extra" button
    has_button = any("extra" in e for e in card["elements"] if isinstance(e, dict))
    assert has_button


# ── Edge case tests ───────────────────────────────────────────

def test_all_formatters_handle_empty_report():
    report = {
        "title": "Empty",
        "date": "2026-03-02",
        "stats": {},
        "queries": [],
        "global_top": [],
    }
    for name in list_formatters():
        fmt = get_formatter(name)
        assert fmt is not None
        result = fmt.format_digest(report)
        assert isinstance(result, dict)


def test_all_formatters_handle_no_judge():
    report = {
        "title": "No Judge",
        "date": "2026-03-02",
        "stats": {"unique_items": 1},
        "queries": [
            {
                "top_items": [
                    {"title": "Basic Paper", "url": "https://example.com", "score": 5.0}
                ]
            }
        ],
        "global_top": [],
    }
    for name in list_formatters():
        fmt = get_formatter(name)
        assert fmt is not None
        result = fmt.format_digest(report)
        assert isinstance(result, dict)
