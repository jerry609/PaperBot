"""
Paper2Code Generation API Route
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.core.abstractions import AgentRunContext
from paperbot.api.auth.dependencies import get_required_user_id

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


class GenCodeRequest(BaseModel):
    user_id: str = "default"
    title: str
    abstract: str
    method_section: Optional[str] = None
    use_orchestrator: bool = True
    use_rag: bool = True
    output_dir: Optional[str] = None


async def gen_code_stream(
    request: GenCodeRequest, *, user_id: str, event_log=None, run_id: str = "", trace_id: str = ""
):
    """Stream code generation progress"""
    try:
        if not run_id:
            run_id = new_run_id()
        if not trace_id:
            trace_id = new_trace_id()

        runtime_context = AgentRunContext(
            run_id=run_id,
            trace_id=trace_id,
            workflow="paper2code",
            agent_name="ReproAgent",
        )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Initializing",
                "message": "Setting up code generation...",
                "run_id": runtime_context.run_id,
                "trace_id": runtime_context.trace_id,
            },
        )

        # Import repro modules
        from ...repro import PaperContext, ReproAgent

        # Create paper context
        paper_context = PaperContext(
            title=request.title,
            abstract=request.abstract,
            method_section=request.method_section,
        )

        yield StreamEvent(
            type="progress",
            data={"phase": "Blueprint", "message": "Distilling paper blueprint..."},
        )

        # Initialize agent
        agent = ReproAgent(
            {
                "use_orchestrator": request.use_orchestrator,
                "use_rag": request.use_rag,
            }
        )

        # Use temp directory if no output specified
        output_dir = Path(request.output_dir) if request.output_dir else Path(tempfile.mkdtemp())

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Planning",
                "message": "Creating implementation plan...",
                "filesGenerated": 0,
                "totalFiles": 5,
            },
        )

        # Run generation
        result = await agent.reproduce_from_paper(
            paper_context,
            output_dir=output_dir,
            user_id=user_id,
            event_log=event_log,
            run_id=run_id,
            trace_id=trace_id,
        )

        # Report file generation progress
        total_files = len(result.generated_files)
        for i, (filename, _) in enumerate(result.generated_files.items()):
            yield StreamEvent(
                type="progress",
                data={
                    "phase": "Generating",
                    "message": f"Writing {filename}...",
                    "currentFile": filename,
                    "filesGenerated": i + 1,
                    "totalFiles": total_files,
                },
            )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Verifying",
                "message": "Running verification checks...",
                "filesGenerated": total_files,
                "totalFiles": total_files,
            },
        )

        # Build result
        files = []
        for filename, content in result.generated_files.items():
            lines = len(content.split("\n")) if content else 0
            purpose = "Generated code"
            if "config" in filename.lower():
                purpose = "Configuration"
            elif "model" in filename.lower():
                purpose = "Neural network model"
            elif "train" in filename.lower():
                purpose = "Training logic"
            elif "data" in filename.lower():
                purpose = "Data loading"
            elif "main" in filename.lower():
                purpose = "Entry point"

            files.append(
                {
                    "name": filename,
                    "lines": lines,
                    "purpose": purpose,
                }
            )

        # Extract blueprint info from plan/spec if available
        blueprint_info = {"architectureType": "unknown", "domain": "unknown"}
        if hasattr(result, "blueprint") and result.blueprint:
            blueprint_info = {
                "architectureType": getattr(result.blueprint, "architecture_type", "unknown"),
                "domain": getattr(result.blueprint, "domain", "unknown"),
            }
        elif result.spec and result.spec.model_type:
            blueprint_info["architectureType"] = result.spec.model_type

        # Check verification status
        verification_passed = False
        if result.verification_results:
            verification_passed = all(v.passed for v in result.verification_results)
        elif result.verification:
            verification_passed = result.verification.get("all_passed", False)

        yield StreamEvent(
            type="result",
            data={
                "success": result.status.value == "completed" or len(result.generated_files) > 0,
                "outputDir": str(output_dir),
                "files": files,
                "blueprint": blueprint_info,
                "verificationPassed": verification_passed,
            },
        )

    except Exception as e:
        yield StreamEvent(type="error", message=str(e))


@router.post("/gen-code")
async def generate_code(
    request: GenCodeRequest,
    http_request: Request,
    user_id: str = Depends(get_required_user_id),
):
    """
    Generate code from paper and stream progress.

    Returns Server-Sent Events with generation updates.
    """
    event_log = getattr(http_request.app.state, "event_log", None)
    run_id = new_run_id()
    trace_id = new_trace_id()

    return StreamingResponse(
        wrap_generator(
            gen_code_stream(request, user_id=user_id, event_log=event_log, run_id=run_id, trace_id=trace_id),
            workflow="gen_code",
            run_id=run_id,
            trace_id=trace_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
