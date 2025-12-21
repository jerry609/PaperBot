from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator

from paperbot.application.collaboration.message_schema import new_run_id
from paperbot.application.ports.event_log_port import EventLogPort


class ScholarPipeline:
    """
    Stable application-layer boundary for the scholar paper analysis pipeline.

    Phase-0: thin wrapper around `paperbot.core.workflow_coordinator.ScholarWorkflowCoordinator`.
    Later phases can migrate logic into application layer while keeping this API stable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._coordinator = ScholarWorkflowCoordinator(self.config)

    async def analyze_paper(
        self,
        paper: Any,
        scholar_name: Optional[str] = None,
        persist_report: bool = True,
        *,
        event_log: Optional[EventLogPort] = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[Optional[Path], Any, Dict[str, Any]]:
        return await self._coordinator.run_paper_pipeline(
            paper=paper,
            scholar_name=scholar_name,
            persist_report=persist_report,
            event_log=event_log,
            run_id=run_id,
            trace_id=trace_id,
        )

    async def analyze_papers(
        self,
        papers: List[Any],
        scholar_name: Optional[str] = None,
        *,
        event_log: Optional[EventLogPort] = None,
        run_id: Optional[str] = None,
    ) -> List[Tuple[Optional[Path], Any, Dict[str, Any]]]:
        return await self._coordinator.run_batch_pipeline(
            papers=papers,
            scholar_name=scholar_name,
            event_log=event_log,
            run_id=run_id,
        )

    @staticmethod
    def new_run_id() -> str:
        return new_run_id()


