"""Push formatter registry — maps channel_type to formatter class."""
from __future__ import annotations

from typing import Dict, Optional, Type

from paperbot.infrastructure.push.formatters.base import PushFormatter
from paperbot.infrastructure.push.formatters.discord import DiscordFormatter
from paperbot.infrastructure.push.formatters.feishu import FeishuFormatter
from paperbot.infrastructure.push.formatters.telegram import TelegramFormatter
from paperbot.infrastructure.push.formatters.wecom import WeComFormatter

_REGISTRY: Dict[str, Type[PushFormatter]] = {
    "telegram": TelegramFormatter,
    "discord": DiscordFormatter,
    "wecom": WeComFormatter,
    "feishu": FeishuFormatter,
    "lark": FeishuFormatter,  # alias: Lark is the international brand of Feishu
}


def get_formatter(channel_type: str) -> Optional[PushFormatter]:
    """Get a formatter instance for the given channel type."""
    cls = _REGISTRY.get(channel_type.lower())
    if cls is None:
        return None
    return cls()


def list_formatters() -> list[str]:
    """List all registered channel types."""
    return sorted(_REGISTRY.keys())


__all__ = [
    "PushFormatter",
    "TelegramFormatter",
    "DiscordFormatter",
    "WeComFormatter",
    "FeishuFormatter",
    "get_formatter",
    "list_formatters",
]
