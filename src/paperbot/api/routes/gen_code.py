"""
Paper2Code Generation API Route
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import tempfile

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


class GenCodeRequest(BaseModel):
    title: str
    abstract: str
    method_section: Optional[str] = None
    use_orchestrator: bool = True
    use_rag: bool = True
    output_dir: Optional[str] = None


async def gen_code_stream(request: GenCodeRequest):
    """Stream code generation progress"""
    try:
        yield StreamEvent(
            type="progress",
            data={"phase": "Initializing", "message": "Setting up code generation..."},
        )

        # Import repro modules
        from ...repro import ReproAgent, PaperContext

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
        agent = ReproAgent({
            "use_orchestrator": request.use_orchestrator,
            "use_rag": request.use_rag,
        })

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
        result = await agent.reproduce_from_paper(paper_context, output_dir=output_dir)

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
            lines = len(content.split('\n')) if content else 0
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

            files.append({
                "name": filename,
                "lines": lines,
                "purpose": purpose,
            })

        yield StreamEvent(
            type="result",
            data={
                "success": result.status.value == "completed",
                "outputDir": str(output_dir),
                "files": files,
                "blueprint": {
                    "architectureType": result.blueprint.architecture_type if result.blueprint else "unknown",
                    "domain": result.blueprint.domain if result.blueprint else "unknown",
                },
                "verificationPassed": len(result.verification_results) > 0 and all(
                    v.passed for v in result.verification_results
                ) if result.verification_results else False,
            },
        )

    except Exception as e:
        yield StreamEvent(type="error", message=str(e))


@router.post("/gen-code")
async def generate_code(request: GenCodeRequest):
    """
    Generate code from paper and stream progress.

    Returns Server-Sent Events with generation updates.
    """
    return StreamingResponse(
        wrap_generator(gen_code_stream(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
