# src/paperbot/core/collaboration/bus.py
"""
协作消息总线。

- 统一记录 Agent/Host 消息
- 支持轮次、会话ID、阶段标记
- 提供持久化接口，便于后续报告/调试复用
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Callable

from .messages import AgentMessage, MessageType


class CollaborationBus:
    """轻量消息总线，面向多 Agent 协作与主持人引导。"""

    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self._messages: List[AgentMessage] = []
        self._subscribers: List[Callable[[AgentMessage], None]] = []
        self._round_index: int = 0

    # ------------------ 基础操作 ------------------
    def add_message(self, message: AgentMessage) -> None:
        """新增一条消息并通知订阅者。"""
        message.conversation_id = message.conversation_id or self.conversation_id
        if message.round_index == 0:
            message.round_index = self._round_index
        self._messages.append(message)
        for cb in self._subscribers:
            try:
                cb(message)
            except Exception:
                # 订阅器错误不影响主流程
                pass

    def add_host_message(self, content: str, stage: str = "", metadata: Optional[dict] = None) -> AgentMessage:
        """便捷创建主持人消息。"""
        msg = AgentMessage(
            sender="HOST",
            message_type=MessageType.HOST_GUIDANCE,
            content=content,
            metadata=metadata or {},
            conversation_id=self.conversation_id,
            round_index=self._round_index,
            stage=stage,
        )
        self.add_message(msg)
        return msg

    def subscribe(self, callback: Callable[[AgentMessage], None]) -> None:
        """订阅消息流。"""
        self._subscribers.append(callback)

    def next_round(self) -> int:
        """递增轮次，用于主持人发言后推进下一轮。"""
        self._round_index += 1
        return self._round_index

    # ------------------ 事件 ------------------
    def add_stage_event(self, stage: str, event: str, metadata: Optional[dict] = None):
        """记录阶段开始/结束等事件。"""
        msg = AgentMessage.stage_event(stage=stage, event=event, **(metadata or {}))
        self.add_message(msg)

    def add_fallback(self, stage: str, reason: str, metadata: Optional[dict] = None):
        """记录降级/回退事件。"""
        msg = AgentMessage.fallback(stage=stage, reason=reason, **(metadata or {}))
        self.add_message(msg)

    # ------------------ 查询 ------------------
    @property
    def messages(self) -> List[AgentMessage]:
        return list(self._messages)

    def latest_messages(self, limit: int = 20) -> List[AgentMessage]:
        return self._messages[-limit:]

    # ------------------ 持久化 ------------------
    def persist(self, path: Path) -> Path:
        """
        将消息以 JSONL 写入磁盘。
        返回最终写入文件路径。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for msg in self._messages:
                f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
        return path

    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "round_index": self._round_index,
            "messages": [asdict(m) for m in self._messages],
        }

