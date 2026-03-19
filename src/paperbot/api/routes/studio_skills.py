from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from paperbot.application.services.studio_skill_catalog import (
    StudioSkillCatalogError,
    get_studio_skill_detail,
    install_studio_skill_repo,
    list_studio_skill_catalog,
    update_studio_skill_repo,
)

router = APIRouter()


class InstallStudioSkillRepoRequest(BaseModel):
    repo_url: str
    repo_ref: Optional[str] = None


class UpdateStudioSkillRepoRequest(BaseModel):
    repo_ref: Optional[str] = None


@router.get("/studio/skills")
async def studio_skills_catalog():
    return list_studio_skill_catalog()


_SKILL_CATALOG_BAD_REQUEST_RESPONSE = {
    400: {"description": "Invalid skill catalog request or repository metadata."}
}


@router.post("/studio/skills/install", responses=_SKILL_CATALOG_BAD_REQUEST_RESPONSE)
async def studio_skill_install(request: InstallStudioSkillRepoRequest):
    try:
        return install_studio_skill_repo(request.repo_url, repo_ref=request.repo_ref)
    except StudioSkillCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/studio/skills/repos/{repo_slug}/update",
    responses=_SKILL_CATALOG_BAD_REQUEST_RESPONSE,
)
async def studio_skill_repo_update(repo_slug: str, request: UpdateStudioSkillRepoRequest):
    try:
        return update_studio_skill_repo(repo_slug, repo_ref=request.repo_ref)
    except StudioSkillCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/studio/skills/{skill_key}")
async def studio_skill_detail(skill_key: str):
    try:
        detail = get_studio_skill_detail(skill_key)
    except StudioSkillCatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return detail.to_payload()
