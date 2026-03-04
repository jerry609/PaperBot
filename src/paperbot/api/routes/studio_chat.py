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
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, AsyncGenerator

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
    context_pack_id: Optional[str] = None


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


log = logging.getLogger(__name__)


def _load_runtime_allowed_dirs() -> List[Path]:
    f = Path("data/runbook_allowed_dirs.json")
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    dirs: List[Path] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            try:
                dirs.append(Path(item).resolve())
            except Exception:
                continue
    return dirs


def _allowed_workdir_prefixes() -> List[Path]:
    prefixes: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    try:
        prefixes.append(Path.cwd().resolve())
    except Exception:
        pass

    extra = os.getenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", "").strip()
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p:
                try:
                    prefixes.append(Path(p).expanduser().resolve())
                except Exception:
                    continue

    prefixes.extend(_load_runtime_allowed_dirs())

    unique: List[Path] = []
    seen: set[str] = set()
    for p in prefixes:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _is_under_prefix(path: Path, prefix: Path) -> bool:
    path_real = os.path.realpath(str(path))
    prefix_real = os.path.realpath(str(prefix))
    return path_real == prefix_real or path_real.startswith(prefix_real + os.sep)


def _resolve_cli_project_dir(raw: Optional[str]) -> Path:
    """Resolve and validate project_dir used by studio CLI execution."""
    if not raw:
        return Path.cwd().resolve()

    cleaned = raw.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError("invalid project_dir")

    if cleaned == "~":
        normalized = str(Path.home())
    elif cleaned.startswith("~/"):
        normalized = str(Path.home() / cleaned[2:])
    else:
        normalized = cleaned

    # Normalize to real path and reconstruct from an allowed prefix.
    # This avoids resolving an arbitrary user-controlled path directly.
    if not os.path.isabs(normalized):
        normalized = str((Path.cwd() / normalized).resolve(strict=False))
    normalized_real = os.path.realpath(normalized)

    resolved: Optional[Path] = None
    for prefix in _allowed_workdir_prefixes():
        prefix_real = os.path.realpath(str(prefix))
        if normalized_real == prefix_real:
            resolved = prefix
            break
        if normalized_real.startswith(prefix_real + os.sep):
            suffix = normalized_real[len(prefix_real):].lstrip("/\\")
            candidate = (prefix / suffix).resolve(strict=False) if suffix else prefix
            if _is_under_prefix(candidate, prefix):
                resolved = candidate
                break
            raise ValueError("project_dir is not allowed")

    if resolved is None:
        raise ValueError("project_dir is not allowed")

    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("project_dir must be an existing directory")
    return resolved


def _load_context_pack(pack_id: str) -> Optional[Dict[str, Any]]:
    """Load a context pack from the database by ID."""
    try:
        from ...infrastructure.stores.repro_context_store import SqlAlchemyReproContextStore

        store = SqlAlchemyReproContextStore()
        return store.get(pack_id)
    except Exception as exc:
        log.warning("Failed to load context pack %s: %s", pack_id, exc)
        return None


def _format_context_pack_markdown(pack: Dict[str, Any]) -> str:
    """Format a context pack as a Markdown document for Claude CLI to read."""
    lines: list[str] = []

    lines.append("# Reproduction Context Pack")
    lines.append("")

    # Paper metadata
    paper = pack.get("paper", {})
    if paper:
        lines.append("## Paper")
        if paper.get("title"):
            lines.append(f"**Title:** {paper['title']}")
        if paper.get("authors"):
            authors = paper["authors"]
            if isinstance(authors, list):
                lines.append(f"**Authors:** {', '.join(authors)}")
            else:
                lines.append(f"**Authors:** {authors}")
        if paper.get("year"):
            lines.append(f"**Year:** {paper['year']}")
        if paper.get("arxiv_id"):
            lines.append(f"**arXiv:** {paper['arxiv_id']}")
        if paper.get("doi"):
            lines.append(f"**DOI:** {paper['doi']}")
        lines.append("")

    # Objective
    if pack.get("objective"):
        lines.append("## Objective")
        lines.append(pack["objective"])
        lines.append("")

    # Task roadmap
    roadmap = pack.get("task_roadmap", [])
    if roadmap:
        lines.append("## Task Roadmap")
        lines.append("")
        for i, step in enumerate(roadmap, 1):
            title = step.get("title", f"Step {i}")
            lines.append(f"### Step {i}: {title}")
            if step.get("description"):
                lines.append(step["description"])
            if step.get("acceptance_criteria"):
                lines.append("")
                lines.append("**Acceptance criteria:**")
                criteria = step["acceptance_criteria"]
                if isinstance(criteria, list):
                    for c in criteria:
                        lines.append(f"- {c}")
                else:
                    lines.append(f"- {criteria}")
            lines.append("")

    # Observations
    observations = pack.get("observations", [])
    if observations:
        lines.append("## Observations")
        lines.append("")
        for obs in observations:
            obs_type = obs.get("type", "note")
            title = obs.get("title", "Untitled")
            confidence = obs.get("confidence", 0)
            lines.append(f"### [{obs_type.upper()}] {title} (confidence: {confidence:.0%})")
            if obs.get("content"):
                lines.append(obs["content"])
            elif obs.get("description"):
                lines.append(obs["description"])
            if obs.get("code_snippet"):
                lines.append("")
                lang = obs.get("language", "")
                lines.append(f"```{lang}")
                lines.append(obs["code_snippet"])
                lines.append("```")
            lines.append("")

    # Warnings
    warnings = pack.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _ensure_context_pack_on_disk(pack_id: str, project_dir: str) -> Optional[str]:
    """Load context pack from DB and write CONTEXT.md into project_dir.

    Returns the file path on success, None on failure.
    """
    pack = _load_context_pack(pack_id)
    if pack is None:
        return None

    md = _format_context_pack_markdown(pack)
    target = Path(project_dir) / "CONTEXT.md"
    try:
        target.write_text(md, encoding="utf-8")
        return str(target)
    except OSError as exc:
        log.warning("Failed to write CONTEXT.md to %s: %s", project_dir, exc)
        return None


def _parse_cli_content_blocks(content_blocks: list) -> list[StreamEvent]:
    """Convert Claude CLI assistant content blocks into StreamEvents."""
    events: list[StreamEvent] = []
    for block in content_blocks:
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                events.append(StreamEvent(
                    type="progress",
                    data={"cli_event": "text", "text": text},
                ))

        elif btype == "tool_use":
            events.append(StreamEvent(
                type="progress",
                data={
                    "cli_event": "tool_use",
                    "tool_name": block.get("name", "unknown"),
                    "tool_input": block.get("input", {}),
                    "tool_id": block.get("id", ""),
                },
            ))

        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                events.append(StreamEvent(
                    type="progress",
                    data={"cli_event": "thinking", "text": thinking},
                ))

    return events


def _parse_cli_event(line_data: Dict[str, Any]) -> list[StreamEvent]:
    """Parse a single NDJSON line from `claude -p --output-format stream-json`.

    Claude CLI stream-json emits one JSON object per line:
    - {"type":"assistant","message":{...}} — assistant turn with content blocks
    - {"type":"tool_result","tool_name":"...","content":"..."} — tool output
    - {"type":"result","subtype":"success","result":"...","cost_usd":...} — final
    - {"type":"system",...} — session init (ignored)
    """
    etype = line_data.get("type", "")
    events: list[StreamEvent] = []

    if etype == "assistant":
        msg = line_data.get("message", {})
        content_blocks = msg.get("content", [])
        events.extend(_parse_cli_content_blocks(content_blocks))

    elif etype == "tool_result":
        events.append(StreamEvent(
            type="progress",
            data={
                "cli_event": "tool_result",
                "tool_name": line_data.get("tool_name", ""),
                "content": _truncate(str(line_data.get("content", "")), 2000),
            },
        ))

    elif etype == "result":
        events.append(StreamEvent(
            type="result",
            data={
                "cli_event": "done",
                "result": line_data.get("result", ""),
                "cost_usd": line_data.get("cost_usd"),
                "duration_ms": line_data.get("duration_ms"),
                "num_turns": line_data.get("num_turns"),
            },
        ))

    # Ignore "system" and other meta events
    return events


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


async def stream_claude_cli(request: StudioChatRequest) -> AsyncGenerator[StreamEvent, None]:
    """Stream Claude CLI output as structured SSE events.

    Uses ``--output-format stream-json`` so we get real-time NDJSON events
    (text, tool_use, tool_result) instead of buffered plain text.
    """

    claude_path = find_claude_cli()

    if not claude_path:
        yield StreamEvent(
            type="error",
            message="Claude CLI not found. Please install it with: npm install -g @anthropic-ai/claude-code"
        )
        return

    # Build command — use stream-json for structured real-time output
    cmd = [claude_path]

    model_id = get_model_id(request.model, for_cli=True)
    cmd.extend(["--model", model_id])

    mode_flag = get_mode_flag(request.mode)
    if mode_flag:
        cmd.append(mode_flag)

    prompt = build_prompt_with_context(request.message, request.paper, request.mode)
    cmd.extend(["-p", prompt, "--output-format", "stream-json", "--verbose"])

    yield StreamEvent(
        type="progress",
        data={
            "phase": "Starting",
            "message": f"[{request.mode}] Connecting to Claude CLI...",
            "model": request.model,
        }
    )

    try:
        try:
            cwd = str(_resolve_cli_project_dir(request.project_dir))
        except ValueError as exc:
            yield StreamEvent(type="error", message=str(exc))
            return

        # Write context pack to working directory so Claude CLI can read it
        if request.context_pack_id:
            pack_path = await asyncio.to_thread(
                _ensure_context_pack_on_disk, request.context_pack_id, cwd,
            )
            if pack_path:
                log.info("Wrote context pack to %s", pack_path)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, "FORCE_COLOR": "0"},
        )
        stderr_chunks: list[str] = []

        async def _drain_stderr() -> None:
            if process.stderr is None:
                return
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return
                stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

        stderr_task = asyncio.create_task(_drain_stderr())

        _KEEPALIVE_SECONDS = 15
        line_buffer = ""

        # Read stdout line-by-line (NDJSON) with keepalive heartbeats.
        while True:
            try:
                chunk = await asyncio.wait_for(
                    process.stdout.read(4096),
                    timeout=_KEEPALIVE_SECONDS,
                )
            except asyncio.TimeoutError:
                yield StreamEvent(
                    type="progress",
                    data={"keepalive": True, "mode": request.mode},
                )
                continue

            if not chunk:
                break

            line_buffer += chunk.decode("utf-8", errors="replace")

            # Process complete lines (each is a JSON object)
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON CLI line: %s", line[:120])
                    continue

                for event in _parse_cli_event(data):
                    yield event

        # Process any trailing data in buffer
        if line_buffer.strip():
            try:
                data = json.loads(line_buffer.strip())
                for event in _parse_cli_event(data):
                    yield event
            except json.JSONDecodeError:
                pass

        await process.wait()
        await stderr_task

        if process.returncode != 0:
            error_msg = "".join(stderr_chunks).strip()
            if error_msg:
                yield StreamEvent(type="error", message=error_msg)
                return

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
