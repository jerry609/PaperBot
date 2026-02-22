"""
Studio Chat API Route - Claude CLI Integration for DeepStudio

Spawns Claude CLI as a subprocess and streams responses.
Supports three modes like CodePilot:
- Code: Can edit files, run commands (permissionMode: acceptEdits)
- Plan: Planning only, no execution (permissionMode: plan)
- Ask: Text-only conversation, no tools (permissionMode: default)
"""

import asyncio
import json
import os
import shutil
import subprocess
from typing import List, Optional, Literal, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()

Mode = Literal["Code", "Plan", "Ask"]
Model = Literal["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"]


class PaperContext(BaseModel):
    title: str
    abstract: str
    method_section: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class StudioChatRequest(BaseModel):
    message: str
    mode: Mode = "Code"
    model: Model = "claude-sonnet-4-5"
    paper: Optional[PaperContext] = None
    project_dir: Optional[str] = None
    history: List[ChatMessage] = []
    session_id: Optional[str] = None


def find_claude_cli() -> Optional[str]:
    """Find Claude CLI executable path."""
    # Check common locations
    candidates = [
        shutil.which("claude"),
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def get_mode_flag(mode: Mode) -> str:
    """Map mode to Claude CLI permission flag."""
    if mode == "Code":
        return "--dangerously-skip-permissions"  # Allow all operations
    elif mode == "Plan":
        return "--plan"  # Planning mode - no execution
    else:  # Ask
        return ""  # Default - no tools


def get_model_id(model: Model, for_cli: bool = False) -> str:
    """Map model selection to Claude model ID.

    Args:
        model: The model selection from the UI
        for_cli: If True, returns CLI-friendly alias; if False, returns full model ID for API
    """
    if for_cli:
        # Claude CLI accepts short aliases
        cli_mapping = {
            "claude-sonnet-4-5": "sonnet",
            "claude-opus-4-5": "opus",
            "claude-haiku-4-5": "haiku",
        }
        return cli_mapping.get(model, "sonnet")
    else:
        # Full model IDs for Anthropic API
        api_mapping = {
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250514",
            "claude-opus-4-5": "claude-opus-4-5-20250514",
            "claude-haiku-4-5": "claude-haiku-4-5-20250514",
        }
        return api_mapping.get(model, "claude-sonnet-4-5-20250514")


def build_prompt_with_context(message: str, paper: Optional[PaperContext], mode: Mode) -> str:
    """Build the prompt with paper context if available."""
    parts = []

    if paper:
        parts.append(f"# Paper Context\n**Title:** {paper.title}\n\n**Abstract:** {paper.abstract}")
        if paper.method_section:
            parts.append(f"\n**Method Section:** {paper.method_section}")
        parts.append("\n---\n")

    if mode == "Code":
        parts.append("You are helping implement this research paper as working code. ")
    elif mode == "Plan":
        parts.append("You are creating an implementation plan for this research paper. Do not write code, only plan. ")
    else:
        parts.append("You are answering questions about this research paper. ")

    parts.append(f"\n# User Request\n{message}")

    return "\n".join(parts)


async def stream_claude_cli(request: StudioChatRequest) -> AsyncGenerator[StreamEvent, None]:
    """Stream Claude CLI output as SSE events."""

    claude_path = find_claude_cli()

    if not claude_path:
        yield StreamEvent(
            type="error",
            message="Claude CLI not found. Please install it with: npm install -g @anthropic-ai/claude-code"
        )
        return

    # Build command
    cmd = [claude_path]

    # Add model flag (use CLI alias)
    model_id = get_model_id(request.model, for_cli=True)
    cmd.extend(["--model", model_id])

    # Add mode flag
    mode_flag = get_mode_flag(request.mode)
    if mode_flag:
        cmd.append(mode_flag)

    # Build prompt with context
    prompt = build_prompt_with_context(request.message, request.paper, request.mode)

    # Add the prompt with proper flags
    # Note: --print requires --verbose for stream-json, so we use regular output
    cmd.extend(["--print", prompt])

    yield StreamEvent(
        type="progress",
        data={
            "phase": "Starting",
            "message": f"[{request.mode}] Connecting to Claude CLI...",
            "model": request.model,
        }
    )

    try:
        # Set working directory
        cwd = request.project_dir or os.getcwd()
        if not os.path.isdir(cwd):
            cwd = os.getcwd()

        # Spawn Claude CLI process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, "FORCE_COLOR": "0"},  # Disable ANSI colors
        )

        full_content = ""

        # Stream stdout - read chunks for real-time streaming
        async def read_stream():
            nonlocal full_content
            while True:
                # Read in chunks for smoother streaming
                chunk = await process.stdout.read(100)
                if not chunk:
                    break

                text = chunk.decode("utf-8", errors="replace")

                # Skip ANSI escape codes
                import re
                text = re.sub(r'\x1B\[[0-9;]*[a-zA-Z]', '', text)

                if text:
                    full_content += text
                    yield StreamEvent(
                        type="progress",
                        data={
                            "delta": text,
                            "content": full_content,
                            "mode": request.mode,
                        }
                    )

        async for event in read_stream():
            yield event

        # Wait for process to complete
        await process.wait()

        # Check for errors
        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            if error_msg:
                yield StreamEvent(type="error", message=error_msg)
                return

        # Emit final result
        yield StreamEvent(
            type="result",
            data={
                "content": full_content,
                "mode": request.mode,
                "model": request.model,
            }
        )

    except FileNotFoundError:
        yield StreamEvent(
            type="error",
            message=f"Claude CLI not found at: {claude_path}"
        )
    except Exception as e:
        yield StreamEvent(
            type="error",
            message=f"Claude CLI error: {str(e)}"
        )


async def stream_anthropic_api(request: StudioChatRequest) -> AsyncGenerator[StreamEvent, None]:
    """Fallback: Stream response using Anthropic API directly."""

    yield StreamEvent(
        type="progress",
        data={"phase": "Processing", "message": f"[{request.mode}] Thinking..."},
    )

    try:
        from ...infrastructure.llm.providers.anthropic_provider import AnthropicProvider

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            yield StreamEvent(type="error", message="ANTHROPIC_API_KEY not set")
            return

        model_id = get_model_id(request.model, for_cli=False)
        provider = AnthropicProvider(
            api_key=api_key,
            model_name=model_id,
            max_tokens=8192,
        )

        # Build system prompt based on mode
        if request.mode == "Code":
            system = "You are an expert AI coding assistant. Generate high-quality, well-documented code. Include type hints and docstrings. Wrap code in appropriate markdown code blocks."
        elif request.mode == "Plan":
            system = "You are a research implementation architect. Create detailed implementation plans WITHOUT generating actual code. Focus on architecture, components, and step-by-step planning."
        else:
            system = "You are a helpful research assistant. Answer questions clearly and concisely."

        if request.paper:
            system += f"\n\n# Paper Context\n**Title:** {request.paper.title}\n**Abstract:** {request.paper.abstract}"
            if request.paper.method_section:
                system += f"\n**Method Section:** {request.paper.method_section}"

        messages = [{"role": "system", "content": system}]

        for msg in request.history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": request.message})

        full_content = ""
        async for chunk in provider.stream(messages):
            if chunk.delta:
                full_content += chunk.delta
                yield StreamEvent(
                    type="progress",
                    data={
                        "delta": chunk.delta,
                        "content": full_content,
                        "mode": request.mode,
                    }
                )

        yield StreamEvent(
            type="result",
            data={
                "content": full_content,
                "mode": request.mode,
                "model": request.model,
            }
        )

    except Exception as e:
        yield StreamEvent(type="error", message=f"API error: {str(e)}")


async def studio_chat_stream(request: StudioChatRequest) -> AsyncGenerator[StreamEvent, None]:
    """Stream studio chat response - tries Claude CLI first, falls back to API."""

    # Check if Claude CLI is available
    claude_path = find_claude_cli()

    if claude_path:
        # Use Claude CLI
        async for event in stream_claude_cli(request):
            yield event
    else:
        # Fallback to Anthropic API
        yield StreamEvent(
            type="progress",
            data={
                "phase": "Info",
                "message": "Claude CLI not found, using Anthropic API directly",
            }
        )
        async for event in stream_anthropic_api(request):
            yield event


@router.post("/studio/chat")
async def studio_chat(request: StudioChatRequest):
    """
    Interactive chat for DeepStudio with Claude CLI integration.

    Modes:
    - Code: Generate/modify code (full tool access)
    - Plan: Create implementation plans (no execution)
    - Ask: Answer questions (no tools)

    Returns Server-Sent Events with streaming text.
    """
    return StreamingResponse(
        wrap_generator(studio_chat_stream(request), workflow="studio_chat"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/studio/status")
async def studio_status():
    """Check if Claude CLI is available."""
    claude_path = find_claude_cli()

    if claude_path:
        # Try to get version
        try:
            result = subprocess.run(
                [claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"

        return {
            "claude_cli": True,
            "claude_path": claude_path,
            "claude_version": version,
        }

    return {
        "claude_cli": False,
        "claude_path": None,
        "claude_version": None,
        "fallback": "anthropic_api",
    }


@router.get("/studio/cwd")
async def studio_cwd():
    """Get the current working directory for Claude CLI session."""
    # Default to user's home directory or a sensible default
    default_cwd = os.path.expanduser("~")

    # Try to get the current working directory
    cwd = os.getcwd()

    # Check if we're in a reasonable project directory
    # If we're in the PaperBot source directory, suggest a better location
    if "PaperBot" in cwd or "paperbot" in cwd.lower():
        # Suggest a projects directory instead
        projects_dir = os.path.expanduser("~/Projects")
        if os.path.isdir(projects_dir):
            suggested_cwd = projects_dir
        else:
            suggested_cwd = os.path.expanduser("~/Documents")
    else:
        suggested_cwd = cwd

    return {
        "cwd": suggested_cwd,
        "actual_cwd": cwd,
        "home": default_cwd,
        "source": "system",
    }
