from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import create_settings
from paperbot.infrastructure.exporters import ObsidianReportExporter

router = APIRouter()


class ObsidianReportCitationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    relevant_finding: str = ""
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    id: Optional[str] = None


class ObsidianReportSectionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = ""
    cited_papers: List[ObsidianReportCitationRequest] = Field(default_factory=list)


class ObsidianMethodComparisonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    paper: str = Field(..., min_length=1, max_length=500)
    pros: str = ""
    cons: str = ""


class ObsidianExportReportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    vault_path: Optional[str] = None
    root_dir: Optional[str] = None
    track_name: Optional[str] = Field(default=None, max_length=128)
    workflow_type: str = "research"
    summary: str = ""
    key_insight: str = ""
    sections: List[ObsidianReportSectionRequest] = Field(default_factory=list)
    methods: List[ObsidianMethodComparisonRequest] = Field(default_factory=list)
    trends: str = ""
    future_directions: str = ""
    references: List[ObsidianReportCitationRequest] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ObsidianExportReportResponse(BaseModel):
    vault_path: str
    root_dir: str
    title: str
    note_path: str


@router.post("/obsidian/export-report", response_model=ObsidianExportReportResponse)
def export_obsidian_report(req: ObsidianExportReportRequest) -> ObsidianExportReportResponse:
    settings = create_settings()
    obsidian_config = settings.obsidian

    vault_value = str(req.vault_path or obsidian_config.vault_path or "").strip()
    if not vault_value:
        raise HTTPException(
            status_code=400,
            detail="vault_path is required. Pass it in the request or configure obsidian.vault_path.",
        )

    exporter = ObsidianReportExporter()
    try:
        result = exporter.export_report_note(
            vault_path=Path(vault_value),
            report=req.model_dump(),
            root_dir=str(req.root_dir or obsidian_config.root_dir or "PaperBot"),
            track_moc_filename=getattr(obsidian_config, "track_moc_filename", "_MOC.md"),
            group_tracks_in_folders=getattr(obsidian_config, "group_tracks_in_folders", True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ObsidianExportReportResponse(**result)
