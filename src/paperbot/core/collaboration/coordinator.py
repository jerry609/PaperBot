# src/paperbot/core/collaboration/coordinator.py
"""
Agent Coordinator for multi-agent collaboration.
Inspired by BettaFish's ForumEngine pattern.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Awaitable

from typing import TYPE_CHECKING
from .messages import AgentMessage, AgentResult, MessageType

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id, make_event

if TYPE_CHECKING:
    from paperbot.application.ports.event_log_port import EventLogPort

logger = logging.getLogger(__name__)


@dataclass
class RegisteredAgent:
    """Info about a registered agent."""
    name: str
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True


class AgentCoordinator:
    """
    Coordinates communication between multiple agents.
    
    Features:
    - Agent registration
    - Message broadcasting
    - Result collection
    - Synthesis of insights
    
    Usage:
        coordinator = AgentCoordinator()
        coordinator.register("ResearchAgent", capabilities=["search", "summarize"])
        coordinator.register("ReviewerAgent", capabilities=["review", "score"])
        
        # Broadcast insight
        coordinator.broadcast(AgentMessage.insight("ResearchAgent", "Found prior art..."))
        
        # Collect and synthesize
        results = coordinator.get_results()
        synthesis = await coordinator.synthesize(results)
    """
    
    def __init__(
        self,
        synthesizer: Optional[Callable[[List[AgentResult]], Awaitable[str]]] = None,
        *,
        event_log: "Optional[EventLogPort]" = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        workflow: str = "",
    ):
        self.agents: Dict[str, RegisteredAgent] = {}
        self.messages: List[AgentMessage] = []
        self.results: List[AgentResult] = []
        self._synthesizer = synthesizer
        self._message_subscribers: List[Callable[[AgentMessage], None]] = []

        # Phase-0 observability: optional event log + trace identifiers.
        self._event_log = event_log
        self._workflow = workflow
        self._run_id = run_id or new_run_id()
        self._trace_id = trace_id or new_trace_id()

    def _emit_event(self, message: AgentMessage) -> None:
        if not self._event_log:
            return
        try:
            evt = make_event(
                run_id=self._run_id,
                trace_id=self._trace_id,
                workflow=self._workflow,
                stage=message.stage or "",
                attempt=message.round_index or 0,
                agent_name=message.sender,
                role="system" if message.sender == "SYSTEM" else "worker",
                type=message.message_type.value,
                payload={
                    "message": message.to_dict(),
                },
                tags={
                    "conversation_id": message.conversation_id,
                },
            )
            self._event_log.append(evt)
        except Exception as e:
            logger.debug(f"Event log emission failed: {e}")
    # ==================== Agent Management ====================

    def register(self, name: str, capabilities: List[str] = None) -> None:
        """Register an agent with the coordinator."""
        self.agents[name] = RegisteredAgent(
            name=name,
            capabilities=capabilities or [],
        )
        logger.info(f"Registered agent: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Unregistered agent: {name}")
    
    def get_agent(self, name: str) -> Optional[RegisteredAgent]:
        """Get agent info by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())
    
    # ==================== Messaging ====================
    
    def broadcast(self, message: AgentMessage) -> None:
        """
        Broadcast a message to all agents.
        Messages are stored and can be subscribed to.
        """
        self.messages.append(message)
        logger.debug(f"[{message.sender}] {message.message_type.value}: {str(message.content)[:100]}")

        # Phase-0: emit unified event envelope (optional).
        self._emit_event(message)
        
        # Notify subscribers
        for subscriber in self._message_subscribers:
            try:
                subscriber(message)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
    
    def subscribe(self, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to all messages."""
        self._message_subscribers.append(callback)
    
    def get_messages(
        self,
        sender: str = None,
        message_type: MessageType = None,
        limit: int = None,
    ) -> List[AgentMessage]:
        """
        Get messages with optional filtering.
        
        Args:
            sender: Filter by sender name
            message_type: Filter by message type
            limit: Max number of messages to return
        """
        filtered = self.messages
        
        if sender:
            filtered = [m for m in filtered if m.sender == sender]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_insights(self) -> List[AgentMessage]:
        """Get all insight messages."""
        return self.get_messages(message_type=MessageType.INSIGHT)
    
    # ==================== Results ====================
    
    def submit_result(self, result: AgentResult) -> None:
        """Submit a result from an agent."""
        self.results.append(result)
        
        # Also broadcast as a message
        msg = AgentMessage.result(
            sender=result.agent_name,
            content={"task": result.task_name, "success": result.success},
        )
        self.broadcast(msg)
    
    def get_results(self, agent_name: str = None) -> List[AgentResult]:
        """Get results, optionally filtered by agent."""
        if agent_name:
            return [r for r in self.results if r.agent_name == agent_name]
        return self.results
    
    def get_successful_results(self) -> List[AgentResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]
    
    # ==================== Synthesis ====================
    
    async def synthesize(self, results: List[AgentResult] = None) -> str:
        """
        Synthesize insights from multiple agent results.
        
        Args:
            results: Results to synthesize (default: all results)
        
        Returns:
            Synthesized summary string
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No results to synthesize."
        
        if self._synthesizer:
            return await self._synthesizer(results)
        
        # Default simple synthesis
        return self._default_synthesis(results)
    
    def _default_synthesis(self, results: List[AgentResult]) -> str:
        """Default synthesis: summarize results."""
        lines = ["## Agent Results Summary\n"]
        
        for r in results:
            status = "✅" if r.success else "❌"
            lines.append(f"- **{r.agent_name}** ({r.task_name}): {status}")
            if r.error:
                lines.append(f"  - Error: {r.error}")
        
        # Add insights
        insights = self.get_insights()
        if insights:
            lines.append("\n## Key Insights\n")
            for i, insight in enumerate(insights[:5], 1):
                content = str(insight.content)[:200]
                lines.append(f"{i}. [{insight.sender}] {content}")
        
        return "\n".join(lines)
    
    # ==================== Utilities ====================
    
    def clear(self) -> None:
        """Clear all messages and results."""
        self.messages.clear()
        self.results.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of coordinator state."""
        return {
            "agents": list(self.agents.keys()),
            "message_count": len(self.messages),
            "result_count": len(self.results),
            "successful_results": len(self.get_successful_results()),
        }

