from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..schema import NormalizedMessage


@dataclass(frozen=True)
class ParsedChatLog:
    platform: str
    messages: List[NormalizedMessage]
    metadata: dict

