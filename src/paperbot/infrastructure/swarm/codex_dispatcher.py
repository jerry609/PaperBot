"""
Codex Dispatcher -- sends coding tasks to OpenAI Codex API (cloud).

Uses principle-driven prompts (concise, goal-oriented) as Codex responds
better to this style than mechanics-driven prompts.
"""

import asyncio
import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from ...repro.base_executor import BaseExecutor
from .worker_tools import CODING_WORKER_TOOLS, TASK_COMPLETE_SENTINEL, ToolExecutor

# Lazy import to avoid circular dependency
_SANDBOX_WORKER_TOOLS = None


def _get_sandbox_worker_tools():
    global _SANDBOX_WORKER_TOOLS
    if _SANDBOX_WORKER_TOOLS is None:
        from .sandbox_tool_executor import SANDBOX_WORKER_TOOLS

        _SANDBOX_WORKER_TOOLS = SANDBOX_WORKER_TOOLS
    return _SANDBOX_WORKER_TOOLS

if TYPE_CHECKING:
    from ...api.routes.agent_board import AgentTask
    from .sandbox_tool_executor import SandboxToolExecutor

log = logging.getLogger(__name__)

# Preferred models in priority order
_CODEX_MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4o",
]

_CODE_BLOCK_RE = re.compile(
    r"```(?P<header>[^\n`]*)\n(?P<body>.*?)```",
    re.DOTALL,
)
_INLINE_FILE_HINT_RE = re.compile(r"(?:file(?:name)?|path)\s*[:=]\s*([^\s]+)", re.IGNORECASE)
_PRELUDE_FILE_HINT_RE = re.compile(r"(?im)^(?:file(?:name)?|path)\s*:\s*([^\s]+)\s*$")
_BODY_FIRST_LINE_HINT_RE = re.compile(
    r"^\s*(?:#|//|--)?\s*(?:file(?:name)?|path)\s*:\s*([^\s]+)\s*$",
    re.IGNORECASE,
)
_LIKELY_FILE_RE = re.compile(
    r"^[A-Za-z0-9_.\-/]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|txt|ini|cfg|sh|sql|css|html)$"
)
_FUNC_DECL_RE = re.compile(r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
_ARROW_FUNC_RE = re.compile(
    r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"
)
_CLASS_DECL_RE = re.compile(r"(?m)^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b")

MAX_ITERATIONS_DEFAULT = 50
MAX_REPEAT_TOOL_CALLS = 3
MAX_TOOL_ERRORS = 5
MAX_MESSAGES = 60
MAX_SUMMARY_CHARS = 1200
DEFAULT_STAGNATION_STEPS = 6
DEFAULT_AUTO_EXTEND_STEPS = 5
DEFAULT_AUTO_EXTEND_MAX = 20
DEFAULT_NEAR_LIMIT_WINDOW = 3

REASON_MAX_ITERATIONS_EXHAUSTED = "max_iterations_exhausted"
REASON_STAGNATION_DETECTED = "stagnation_detected"
REASON_REPEATED_TOOL_CALLS = "repeated_tool_calls"
REASON_TOO_MANY_TOOL_ERRORS = "too_many_tool_errors"
REASON_MALFORMED_TOOL_ARGS = "malformed_tool_arguments"
REASON_EMPTY_CHOICES = "empty_choices"
REASON_MISSING_TOOL_CALLS = "missing_tool_calls"
REASON_TERMINATED_FINISH_REASON = "terminated_finish_reason"
REASON_UNSUPPORTED_FINISH_REASON = "unsupported_finish_reason"


@dataclass
class ToolLoopPolicy:
    hard_max_iterations: int
    auto_budget_enabled: bool = False
    stagnation_steps: int = DEFAULT_STAGNATION_STEPS
    auto_extend_steps: int = DEFAULT_AUTO_EXTEND_STEPS
    max_total_extension: int = DEFAULT_AUTO_EXTEND_MAX
    near_limit_window: int = DEFAULT_NEAR_LIMIT_WINDOW

    @classmethod
    def from_env(cls, requested_max_iterations: Optional[int]) -> "ToolLoopPolicy":
        hard_max = _parse_int_env(
            name="CODEX_MAX_ITERATIONS",
            default=MAX_ITERATIONS_DEFAULT,
            min_value=1,
            max_value=200,
        )
        if requested_max_iterations is not None:
            hard_max = max(1, int(requested_max_iterations))

        auto_budget_enabled = (
            os.getenv("CODEX_LOOP_AUTO_BUDGET", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        stagnation_steps = _parse_int_env(
            name="CODEX_LOOP_STAGNATION_STEPS",
            default=DEFAULT_STAGNATION_STEPS,
            min_value=1,
            max_value=50,
        )
        auto_extend_steps = _parse_int_env(
            name="CODEX_LOOP_AUTO_EXTEND_STEPS",
            default=DEFAULT_AUTO_EXTEND_STEPS,
            min_value=1,
            max_value=50,
        )
        max_total_extension = _parse_int_env(
            name="CODEX_LOOP_AUTO_EXTEND_MAX",
            default=DEFAULT_AUTO_EXTEND_MAX,
            min_value=0,
            max_value=200,
        )
        near_limit_window = _parse_int_env(
            name="CODEX_LOOP_NEAR_LIMIT_WINDOW",
            default=DEFAULT_NEAR_LIMIT_WINDOW,
            min_value=1,
            max_value=20,
        )
        return cls(
            hard_max_iterations=hard_max,
            auto_budget_enabled=auto_budget_enabled,
            stagnation_steps=stagnation_steps,
            auto_extend_steps=auto_extend_steps,
            max_total_extension=max_total_extension,
            near_limit_window=near_limit_window,
        )


@dataclass
class LoopProgressTracker:
    stagnation_steps: int
    consecutive_no_progress_steps: int = 0
    last_progress_step: int = -1
    iterations_with_progress: int = 0
    last_tool_name: str = ""
    last_observation_preview: str = ""
    seen_tool_signatures: Set[str] = field(default_factory=set)

    def mark_iteration(self, step: int, had_progress: bool) -> None:
        if had_progress:
            self.consecutive_no_progress_steps = 0
            self.last_progress_step = step
            self.iterations_with_progress += 1
        else:
            self.consecutive_no_progress_steps += 1

    def should_stop_for_stagnation(self) -> bool:
        return self.consecutive_no_progress_steps >= self.stagnation_steps


class CacheMetrics:
    """Lightweight tracker for OpenAI prompt-cache hit rate."""

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.cached_prompt_tokens = 0

    def record(self, usage: Any) -> None:
        if not usage:
            return
        self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details and hasattr(details, "cached_tokens"):
            self.cached_prompt_tokens += getattr(details, "cached_tokens", 0)

    @property
    def hit_rate(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.cached_prompt_tokens / self.total_prompt_tokens

    def report(self) -> str:
        return (
            f"KV-cache: {self.cached_prompt_tokens}/{self.total_prompt_tokens} "
            f"tokens cached ({self.hit_rate:.0%})"
        )


@dataclass
class CodexResult:
    task_id: str
    success: bool
    output: str = ""
    files_generated: List[str] = field(default_factory=list)
    file_snapshots: Dict[str, str] = field(default_factory=dict)  # path → content for replay
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class CodexDispatcher:
    """Dispatches coding tasks to OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dispatch_timeout_seconds: Optional[float] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("CODEX_MODEL", _CODEX_MODELS[0])
        env_timeout = os.getenv("CODEX_DISPATCH_TIMEOUT_SECONDS")
        if dispatch_timeout_seconds is not None:
            self.dispatch_timeout_seconds = max(1.0, float(dispatch_timeout_seconds))
        elif env_timeout:
            try:
                self.dispatch_timeout_seconds = max(1.0, float(env_timeout))
            except ValueError:
                self.dispatch_timeout_seconds = 180.0
        else:
            self.dispatch_timeout_seconds = 180.0
        self.cache_metrics = CacheMetrics()

    async def dispatch(self, task_id: str, prompt: str, workspace: Path) -> CodexResult:
        """Send a coding task to OpenAI and return the result."""
        if not self.api_key:
            return CodexResult(
                task_id=task_id,
                success=False,
                error="OPENAI_API_KEY not set",
            )

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert coding agent. Generate clean, "
                                "well-documented code. Focus on correctness and "
                                "clarity. Follow the project conventions. "
                                "Output complete file contents with filenames."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                ),
                timeout=self.dispatch_timeout_seconds,
            )

            output_text = response.choices[0].message.content or ""
            files_generated = self._persist_output(task_id, output_text, workspace)

            return CodexResult(
                task_id=task_id,
                success=True,
                output=output_text,
                files_generated=files_generated,
            )

        except asyncio.TimeoutError:
            return CodexResult(
                task_id=task_id,
                success=False,
                error=(
                    f"Codex dispatch timed out after "
                    f"{int(self.dispatch_timeout_seconds)}s (model={self.model})"
                ),
            )
        except Exception as exc:
            log.exception("Codex dispatch failed for task %s", task_id)
            return CodexResult(
                task_id=task_id,
                success=False,
                error=str(exc),
            )

    async def dispatch_auto(
        self,
        task_id: str,
        prompt: str,
        workspace: Path,
        *,
        sandbox: Optional[BaseExecutor] = None,
        task: Optional["AgentTask"] = None,
        on_step: Optional[Callable[[int, str, Dict[str, Any], str], Awaitable[None]]] = None,
        max_iterations: Optional[int] = None,
    ) -> CodexResult:
        """Route to tool loop by default, with env-based fallback to legacy mode."""
        use_tools = os.getenv("CODEX_TOOL_USE", "true").lower() != "false"
        if use_tools:
            return await self.dispatch_with_tools(
                task_id=task_id,
                prompt=prompt,
                workspace=workspace,
                sandbox=sandbox,
                task=task,
                on_step=on_step,
                max_iterations=max_iterations,
            )
        return await self.dispatch(task_id=task_id, prompt=prompt, workspace=workspace)

    async def dispatch_with_tools(
        self,
        task_id: str,
        prompt: str,
        workspace: Path,
        *,
        sandbox: Optional[BaseExecutor] = None,
        task: Optional["AgentTask"] = None,
        on_step: Optional[Callable[[int, str, Dict[str, Any], str], Awaitable[None]]] = None,
        max_iterations: Optional[int] = None,
    ) -> CodexResult:
        """Run iterative tool-calling loop (CodeAct style) for a coding task."""
        if not self.api_key:
            return CodexResult(
                task_id=task_id,
                success=False,
                error="OPENAI_API_KEY not set",
            )

        policy = self._resolve_loop_policy(max_iterations)
        max_iter = policy.hard_max_iterations
        effective_max_iterations = max_iter
        extensions_used = 0
        steps_executed = 0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._tool_system_prompt(sandbox=sandbox)},
            {"role": "user", "content": prompt},
        ]
        tool_exec = ToolExecutor(workspace=workspace, sandbox=sandbox, task=task)
        tracker = LoopProgressTracker(stagnation_steps=policy.stagnation_steps)
        repeat_signature: Optional[str] = None
        repeat_count = 0
        tool_error_count = 0

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            step = 0
            while step < effective_max_iterations:
                steps_executed = step + 1
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=CODING_WORKER_TOOLS,
                        max_tokens=4096,
                    ),
                    timeout=self.dispatch_timeout_seconds,
                )

                if not getattr(response, "choices", None):
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_exec.files_written),
                        error="Codex tool loop returned empty choices.",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_EMPTY_CHOICES,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )

                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)

                if finish_reason == "tool_calls":
                    dumped = self._message_to_dict(message)
                    messages.append(dumped)
                    tool_calls = self._extract_tool_calls(message)
                    if not tool_calls:
                        return CodexResult(
                            task_id=task_id,
                            success=False,
                            files_generated=list(tool_exec.files_written),
                            error="finish_reason=tool_calls but no tool calls were returned.",
                            diagnostics=self._build_loop_diagnostics(
                                reason_code=REASON_MISSING_TOOL_CALLS,
                                policy=policy,
                                effective_max_iterations=effective_max_iterations,
                                steps_executed=steps_executed,
                                extensions_used=extensions_used,
                                tracker=tracker,
                                tool_error_count=tool_error_count,
                                repeat_count=repeat_count,
                            ),
                        )

                    had_progress_this_step = False
                    for tool_call in tool_calls:
                        call_id, tool_name, args_raw = tool_call
                        tracker.last_tool_name = tool_name
                        args, parse_error = self._parse_tool_args(args_raw)
                        signature: Optional[str] = None
                        if parse_error is not None:
                            observation = (
                                f"Error: malformed tool arguments for '{tool_name}': {parse_error}"
                            )
                            tool_error_count += 1
                        else:
                            signature = self._tool_signature(tool_name, args)
                            if repeat_signature == signature:
                                repeat_count += 1
                            else:
                                repeat_signature = signature
                                repeat_count = 1

                            if repeat_count > MAX_REPEAT_TOOL_CALLS:
                                return CodexResult(
                                    task_id=task_id,
                                    success=False,
                                    files_generated=list(tool_exec.files_written),
                                    error=(
                                        "Codex tool loop aborted due to repeated identical tool calls "
                                        f"(>{MAX_REPEAT_TOOL_CALLS})."
                                    ),
                                    diagnostics=self._build_loop_diagnostics(
                                        reason_code=REASON_REPEATED_TOOL_CALLS,
                                        policy=policy,
                                        effective_max_iterations=effective_max_iterations,
                                        steps_executed=steps_executed,
                                        extensions_used=extensions_used,
                                        tracker=tracker,
                                        tool_error_count=tool_error_count,
                                        repeat_count=repeat_count,
                                    ),
                                )

                            files_before = len(tool_exec.files_written)
                            observation = await tool_exec.execute(tool_name, args)
                            files_after = len(tool_exec.files_written)
                            had_progress_this_step = had_progress_this_step or self._tool_call_had_progress(
                                tool_name=tool_name,
                                args=args,
                                observation=observation,
                                signature=signature,
                                tracker=tracker,
                                files_before=files_before,
                                files_after=files_after,
                            )
                            if observation.lower().startswith("error:"):
                                tool_error_count += 1

                        tracker.last_observation_preview = (
                            observation if len(observation) <= 220 else f"{observation[:220]}..."
                        )

                        if on_step is not None:
                            await on_step(step, tool_name, args, observation)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": observation,
                            }
                        )
                        messages = self._compress_messages(messages)

                        if observation == TASK_COMPLETE_SENTINEL:
                            summary = args.get("summary", "") if isinstance(args, dict) else ""
                            return CodexResult(
                                task_id=task_id,
                                success=True,
                                output=summary,
                                files_generated=list(tool_exec.files_written),
                                diagnostics=self._build_loop_diagnostics(
                                    reason_code="completed",
                                    policy=policy,
                                    effective_max_iterations=effective_max_iterations,
                                    steps_executed=steps_executed,
                                    extensions_used=extensions_used,
                                    tracker=tracker,
                                    tool_error_count=tool_error_count,
                                    repeat_count=repeat_count,
                                ),
                            )

                        if tool_error_count >= MAX_TOOL_ERRORS:
                            return CodexResult(
                                task_id=task_id,
                                success=False,
                                files_generated=list(tool_exec.files_written),
                                error=(
                                    "Codex tool loop aborted due to too many tool errors "
                                    f"({tool_error_count})."
                                ),
                                diagnostics=self._build_loop_diagnostics(
                                    reason_code=REASON_TOO_MANY_TOOL_ERRORS,
                                    policy=policy,
                                    effective_max_iterations=effective_max_iterations,
                                    steps_executed=steps_executed,
                                    extensions_used=extensions_used,
                                    tracker=tracker,
                                    tool_error_count=tool_error_count,
                                    repeat_count=repeat_count,
                                ),
                            )

                    tracker.mark_iteration(step, had_progress_this_step)
                    if tracker.should_stop_for_stagnation():
                        return CodexResult(
                            task_id=task_id,
                            success=False,
                            files_generated=list(tool_exec.files_written),
                            error=(
                                "Codex tool loop stopped early due to stagnation: "
                                f"no meaningful progress for {tracker.consecutive_no_progress_steps} steps."
                            ),
                            diagnostics=self._build_loop_diagnostics(
                                reason_code=REASON_STAGNATION_DETECTED,
                                policy=policy,
                                effective_max_iterations=effective_max_iterations,
                                steps_executed=steps_executed,
                                extensions_used=extensions_used,
                                tracker=tracker,
                                tool_error_count=tool_error_count,
                                repeat_count=repeat_count,
                            ),
                        )
                    if self._should_auto_extend_budget(
                        policy=policy,
                        effective_max_iterations=effective_max_iterations,
                        current_step=step,
                        extensions_used=extensions_used,
                        tracker=tracker,
                    ):
                        grant = min(
                            policy.auto_extend_steps,
                            max(0, policy.max_total_extension - extensions_used),
                        )
                        if grant > 0:
                            effective_max_iterations += grant
                            extensions_used += grant

                elif finish_reason == "stop":
                    output_text = (getattr(message, "content", None) or "").strip()
                    messages.append({"role": "assistant", "content": output_text})

                    if tool_exec.files_written:
                        return CodexResult(
                            task_id=task_id,
                            success=True,
                            output=output_text,
                            files_generated=list(tool_exec.files_written),
                            diagnostics=self._build_loop_diagnostics(
                                reason_code="completed",
                                policy=policy,
                                effective_max_iterations=effective_max_iterations,
                                steps_executed=steps_executed,
                                extensions_used=extensions_used,
                                tracker=tracker,
                                tool_error_count=tool_error_count,
                                repeat_count=repeat_count,
                            ),
                        )

                    files_generated = self._persist_output(task_id, output_text, workspace)
                    return CodexResult(
                        task_id=task_id,
                        success=True,
                        output=output_text,
                        files_generated=files_generated,
                        diagnostics=self._build_loop_diagnostics(
                            reason_code="completed",
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )

                elif finish_reason in {"length", "content_filter"}:
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_exec.files_written),
                        error=f"Codex tool loop terminated with finish_reason={finish_reason}.",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_TERMINATED_FINISH_REASON,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )

                else:
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_exec.files_written),
                        error=f"Unsupported finish_reason in tool loop: {finish_reason}",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_UNSUPPORTED_FINISH_REASON,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )
                step += 1

            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_exec.files_written),
                error=f"Agent loop did not finish within {effective_max_iterations} iterations.",
                diagnostics=self._build_loop_diagnostics(
                    reason_code=REASON_MAX_ITERATIONS_EXHAUSTED,
                    policy=policy,
                    effective_max_iterations=effective_max_iterations,
                    steps_executed=steps_executed,
                    extensions_used=extensions_used,
                    tracker=tracker,
                    tool_error_count=tool_error_count,
                    repeat_count=repeat_count,
                ),
            )

        except asyncio.TimeoutError:
            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_exec.files_written),
                error=(
                    f"Codex dispatch timed out after "
                    f"{int(self.dispatch_timeout_seconds)}s (model={self.model})"
                ),
            )
        except Exception as exc:
            log.exception("Codex tool dispatch failed for task %s", task_id)
            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_exec.files_written),
                error=str(exc),
            )

    async def dispatch_parallel(
        self, tasks: List[Dict[str, Any]], workspace: Path
    ) -> List[CodexResult]:
        """Dispatch multiple tasks concurrently."""
        coros = [self.dispatch(t["task_id"], t["prompt"], workspace) for t in tasks]
        return await asyncio.gather(*coros, return_exceptions=False)

    async def dispatch_with_sandbox_tools(
        self,
        task_id: str,
        prompt: str,
        tool_executor: "SandboxToolExecutor",
        *,
        on_step: Optional[Callable[[int, str, Dict[str, Any], str], Awaitable[None]]] = None,
        on_think: Optional[Callable[[int, str], Awaitable[None]]] = None,
        max_iterations: Optional[int] = None,
    ) -> CodexResult:
        """Run iterative tool loop using SandboxToolExecutor (sandbox-as-workspace).

        All file operations go through the VM. No local file I/O.
        run_command is always available since the VM IS the workspace.
        """
        if not self.api_key:
            return CodexResult(task_id=task_id, success=False, error="OPENAI_API_KEY not set")

        policy = self._resolve_loop_policy(max_iterations)
        max_iter = policy.hard_max_iterations
        effective_max_iterations = max_iter
        extensions_used = 0
        steps_executed = 0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._sandbox_workspace_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        tools = _get_sandbox_worker_tools()
        tracker = LoopProgressTracker(stagnation_steps=policy.stagnation_steps)
        repeat_signature: Optional[str] = None
        repeat_count = 0
        tool_error_count = 0

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            step = 0
            while step < effective_max_iterations:
                steps_executed = step + 1
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        max_tokens=4096,
                    ),
                    timeout=self.dispatch_timeout_seconds,
                )

                # Record KV-cache metrics
                self.cache_metrics.record(getattr(response, "usage", None))

                if not getattr(response, "choices", None):
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_executor.files_written),
                        error="Codex tool loop returned empty choices.",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_EMPTY_CHOICES,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )

                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)

                if finish_reason == "tool_calls":
                    dumped = self._message_to_dict(message)
                    messages.append(dumped)

                    # Capture model reasoning text (if any) before tool calls.
                    # OpenAI models may place reasoning in `content` alongside tool_calls,
                    # or in a separate `reasoning` / `refusal` field.
                    thinking_text = (getattr(message, "content", None) or "").strip()
                    if not thinking_text:
                        thinking_text = (getattr(message, "reasoning", None) or "").strip()
                    if not thinking_text:
                        # Try extracting from dumped dict (covers edge cases)
                        thinking_text = (dumped.get("content") or "").strip() if isinstance(dumped, dict) else ""
                    if thinking_text and on_think is not None:
                        await on_think(step, thinking_text)

                    tool_calls = self._extract_tool_calls(message)
                    if not tool_calls:
                        return CodexResult(
                            task_id=task_id,
                            success=False,
                            files_generated=list(tool_executor.files_written),
                            error="finish_reason=tool_calls but no tool calls were returned.",
                            diagnostics=self._build_loop_diagnostics(
                                reason_code=REASON_MISSING_TOOL_CALLS,
                                policy=policy,
                                effective_max_iterations=effective_max_iterations,
                                steps_executed=steps_executed,
                                extensions_used=extensions_used,
                                tracker=tracker,
                                tool_error_count=tool_error_count,
                                repeat_count=repeat_count,
                            ),
                        )

                    had_progress_this_step = False
                    for tool_call in tool_calls:
                        call_id, tool_name, args_raw = tool_call
                        tracker.last_tool_name = tool_name
                        args, parse_error = self._parse_tool_args(args_raw)
                        signature: Optional[str] = None
                        if parse_error is not None:
                            observation = (
                                f"Error: malformed tool arguments for '{tool_name}': {parse_error}"
                            )
                            tool_error_count += 1
                        else:
                            signature = self._tool_signature(tool_name, args)
                            if repeat_signature == signature:
                                repeat_count += 1
                            else:
                                repeat_signature = signature
                                repeat_count = 1

                            if repeat_count > MAX_REPEAT_TOOL_CALLS:
                                return CodexResult(
                                    task_id=task_id,
                                    success=False,
                                    files_generated=list(tool_executor.files_written),
                                    error=(
                                        "Codex tool loop aborted due to repeated identical "
                                        f"tool calls (>{MAX_REPEAT_TOOL_CALLS})."
                                    ),
                                    diagnostics=self._build_loop_diagnostics(
                                        reason_code=REASON_REPEATED_TOOL_CALLS,
                                        policy=policy,
                                        effective_max_iterations=effective_max_iterations,
                                        steps_executed=steps_executed,
                                        extensions_used=extensions_used,
                                        tracker=tracker,
                                        tool_error_count=tool_error_count,
                                        repeat_count=repeat_count,
                                    ),
                                )

                            files_before = len(tool_executor.files_written)
                            observation = await tool_executor.execute(tool_name, args)
                            files_after = len(tool_executor.files_written)
                            had_progress_this_step = had_progress_this_step or self._tool_call_had_progress(
                                tool_name=tool_name,
                                args=args,
                                observation=observation,
                                signature=signature,
                                tracker=tracker,
                                files_before=files_before,
                                files_after=files_after,
                            )
                            if observation.lower().startswith("error:"):
                                tool_error_count += 1

                        tracker.last_observation_preview = (
                            observation if len(observation) <= 220 else f"{observation[:220]}..."
                        )

                        if on_step is not None:
                            await on_step(step, tool_name, args, observation)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": observation,
                        })
                        messages = self._compress_messages(messages)

                        if observation == TASK_COMPLETE_SENTINEL:
                            summary = args.get("summary", "") if isinstance(args, dict) else ""
                            return CodexResult(
                                task_id=task_id,
                                success=True,
                                output=summary,
                                files_generated=list(tool_executor.files_written),
                                diagnostics=self._build_loop_diagnostics(
                                    reason_code="completed",
                                    policy=policy,
                                    effective_max_iterations=effective_max_iterations,
                                    steps_executed=steps_executed,
                                    extensions_used=extensions_used,
                                    tracker=tracker,
                                    tool_error_count=tool_error_count,
                                    repeat_count=repeat_count,
                                ),
                            )

                        if tool_error_count >= MAX_TOOL_ERRORS:
                            return CodexResult(
                                task_id=task_id,
                                success=False,
                                files_generated=list(tool_executor.files_written),
                                error=(
                                    f"Codex tool loop aborted due to too many tool errors "
                                    f"({tool_error_count})."
                                ),
                                diagnostics=self._build_loop_diagnostics(
                                    reason_code=REASON_TOO_MANY_TOOL_ERRORS,
                                    policy=policy,
                                    effective_max_iterations=effective_max_iterations,
                                    steps_executed=steps_executed,
                                    extensions_used=extensions_used,
                                    tracker=tracker,
                                    tool_error_count=tool_error_count,
                                    repeat_count=repeat_count,
                                ),
                            )

                    tracker.mark_iteration(step, had_progress_this_step)
                    if tracker.should_stop_for_stagnation():
                        return CodexResult(
                            task_id=task_id,
                            success=False,
                            files_generated=list(tool_executor.files_written),
                            error=(
                                "Codex tool loop stopped early due to stagnation: "
                                f"no meaningful progress for {tracker.consecutive_no_progress_steps} steps."
                            ),
                            diagnostics=self._build_loop_diagnostics(
                                reason_code=REASON_STAGNATION_DETECTED,
                                policy=policy,
                                effective_max_iterations=effective_max_iterations,
                                steps_executed=steps_executed,
                                extensions_used=extensions_used,
                                tracker=tracker,
                                tool_error_count=tool_error_count,
                                repeat_count=repeat_count,
                            ),
                        )
                    if self._should_auto_extend_budget(
                        policy=policy,
                        effective_max_iterations=effective_max_iterations,
                        current_step=step,
                        extensions_used=extensions_used,
                        tracker=tracker,
                    ):
                        grant = min(
                            policy.auto_extend_steps,
                            max(0, policy.max_total_extension - extensions_used),
                        )
                        if grant > 0:
                            effective_max_iterations += grant
                            extensions_used += grant

                elif finish_reason == "stop":
                    output_text = (getattr(message, "content", None) or "").strip()
                    # Capture final text as thinking before returning
                    if output_text and on_think is not None:
                        await on_think(step, output_text)
                    return CodexResult(
                        task_id=task_id,
                        success=True,
                        output=output_text,
                        files_generated=list(tool_executor.files_written),
                        diagnostics=self._build_loop_diagnostics(
                            reason_code="completed",
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )

                elif finish_reason in {"length", "content_filter"}:
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_executor.files_written),
                        error=f"Codex tool loop terminated with finish_reason={finish_reason}.",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_TERMINATED_FINISH_REASON,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )
                else:
                    return CodexResult(
                        task_id=task_id,
                        success=False,
                        files_generated=list(tool_executor.files_written),
                        error=f"Unsupported finish_reason in tool loop: {finish_reason}",
                        diagnostics=self._build_loop_diagnostics(
                            reason_code=REASON_UNSUPPORTED_FINISH_REASON,
                            policy=policy,
                            effective_max_iterations=effective_max_iterations,
                            steps_executed=steps_executed,
                            extensions_used=extensions_used,
                            tracker=tracker,
                            tool_error_count=tool_error_count,
                            repeat_count=repeat_count,
                        ),
                    )
                step += 1

            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_executor.files_written),
                error=f"Agent loop did not finish within {effective_max_iterations} iterations.",
                diagnostics=self._build_loop_diagnostics(
                    reason_code=REASON_MAX_ITERATIONS_EXHAUSTED,
                    policy=policy,
                    effective_max_iterations=effective_max_iterations,
                    steps_executed=steps_executed,
                    extensions_used=extensions_used,
                    tracker=tracker,
                    tool_error_count=tool_error_count,
                    repeat_count=repeat_count,
                ),
            )

        except asyncio.TimeoutError:
            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_executor.files_written),
                error=(
                    f"Codex dispatch timed out after "
                    f"{int(self.dispatch_timeout_seconds)}s (model={self.model})"
                ),
            )
        except Exception as exc:
            log.exception("Sandbox tool dispatch failed for task %s", task_id)
            return CodexResult(
                task_id=task_id,
                success=False,
                files_generated=list(tool_executor.files_written),
                error=str(exc),
            )

    def _tool_system_prompt(self, *, sandbox: Optional[BaseExecutor] = None) -> str:
        base = (
            "You are an expert coding agent working in a workspace with tools.\n\n"
            "Workflow:\n"
            "1. Use list_files/read_file first to understand existing code.\n"
            "2. Use write_file to implement changes.\n"
            "3. Use run_command only when available to verify behavior.\n"
            "4. Update progress with update_subtask.\n"
            "5. Call task_done with a short summary when complete.\n\n"
            "CRITICAL: You MUST call task_done when your work is complete. "
            "Do not keep iterating after you have written all necessary files and verified them. "
            "If you have completed the task, call task_done immediately.\n\n"
            "Rules:\n"
            "- Make minimal, correct changes.\n"
            "- Inspect tool outputs before taking the next action.\n"
            "- Avoid repeated identical tool calls.\n"
            "- If a command or step fails, diagnose and fix before continuing."
        )
        if self._sandbox_available(sandbox):
            base += (
                "\n\nEnvironment:\n"
                "- You are running inside a sandboxed Ubuntu VM.\n"
                "- Pre-installed packages include: torch, transformers, datasets, numpy, scipy, "
                "pandas, matplotlib, scikit-learn, pillow, opencv-python-headless, pytest, black.\n"
                "- If you need an additional package, install it with run_command using pip install -q.\n"
                "- Do not reinstall packages that are already available.\n"
                "- Use apt-get install only when a system package is strictly required."
            )
        return base

    def _sandbox_workspace_system_prompt(self) -> str:
        """System prompt for sandbox-as-workspace mode."""
        return (
            "You are an expert coding agent working directly inside a VM sandbox.\n\n"
            "Your workspace IS the sandbox — every file you write is immediately executable.\n"
            "There is no separate upload step.\n\n"
            "## Thinking Out Loud\n"
            "Before EVERY tool call, write a brief 'Thought:' section explaining:\n"
            "- What you are about to do and why\n"
            "- What you expect the result to be\n"
            "This makes your reasoning visible and helps catch mistakes early.\n\n"
            "## File Organization\n"
            "Organize files into a proper directory structure:\n"
            "- src/ or <project_name>/ — main source code (modules, classes)\n"
            "- tests/ — test files\n"
            "- configs/ or config/ — configuration files (YAML, JSON)\n"
            "- scripts/ — utility/runner scripts\n"
            "- data/ — data files, datasets, checkpoints\n"
            "- docs/ — documentation\n"
            "NEVER dump all .py files flat in the root directory. "
            "Group related code into packages with __init__.py files.\n"
            "The entry point (e.g., main.py, train.py) may live in the root, "
            "but implementation modules must be in subdirectories.\n\n"
            "Workflow:\n"
            "1. Use list_files/read_file to understand existing code and plans.\n"
            "2. Use write_file to implement changes (files go directly to the VM).\n"
            "3. Use run_command to verify behavior — your code is already in the VM.\n"
            "4. Update progress with update_subtask.\n"
            "5. Call task_done with a short summary when complete.\n\n"
            "CRITICAL: You MUST call task_done when your work is complete. "
            "Do not keep iterating after you have written all necessary files and verified them. "
            "If you have completed the task, call task_done immediately.\n\n"
            "Rules:\n"
            "- Make minimal, correct changes.\n"
            "- Inspect tool outputs before taking the next action.\n"
            "- Avoid repeated identical tool calls.\n"
            "- If a command fails, diagnose and fix before continuing.\n\n"
            "Environment:\n"
            "- You are running inside a sandboxed Ubuntu VM.\n"
            "- Pre-installed packages include: torch, transformers, datasets, numpy, scipy, "
            "pandas, matplotlib, scikit-learn, pillow, opencv-python-headless, pytest, black.\n"
            "- If you need an additional package, install it with run_command using pip install -q.\n"
            "- Do not reinstall packages that are already available.\n"
            "- Use apt-get install only when a system package is strictly required."
        )

    @staticmethod
    def _sandbox_available(sandbox: Optional[BaseExecutor]) -> bool:
        return sandbox is not None and sandbox.available()

    def _resolve_loop_policy(self, requested: Optional[int]) -> ToolLoopPolicy:
        return ToolLoopPolicy.from_env(requested)

    def _resolve_max_iterations(self, requested: Optional[int]) -> int:
        return self._resolve_loop_policy(requested).hard_max_iterations

    def _build_loop_diagnostics(
        self,
        *,
        reason_code: str,
        policy: ToolLoopPolicy,
        effective_max_iterations: int,
        steps_executed: int,
        extensions_used: int,
        tracker: LoopProgressTracker,
        tool_error_count: int,
        repeat_count: int,
    ) -> Dict[str, Any]:
        return {
            "reason_code": reason_code,
            "hard_max_iterations": policy.hard_max_iterations,
            "effective_max_iterations": effective_max_iterations,
            "steps_executed": steps_executed,
            "extensions_used": extensions_used,
            "stagnation_steps": policy.stagnation_steps,
            "consecutive_no_progress_steps": tracker.consecutive_no_progress_steps,
            "last_progress_step": tracker.last_progress_step,
            "iterations_with_progress": tracker.iterations_with_progress,
            "tool_error_count": tool_error_count,
            "repeat_call_count": repeat_count,
            "last_tool_name": tracker.last_tool_name,
            "last_observation_preview": tracker.last_observation_preview,
        }

    def _should_auto_extend_budget(
        self,
        *,
        policy: ToolLoopPolicy,
        effective_max_iterations: int,
        current_step: int,
        extensions_used: int,
        tracker: LoopProgressTracker,
    ) -> bool:
        if not policy.auto_budget_enabled:
            return False
        if extensions_used >= policy.max_total_extension:
            return False
        remaining = effective_max_iterations - (current_step + 1)
        near_limit = remaining <= policy.near_limit_window
        recent_progress = tracker.last_progress_step >= 0 and (
            current_step - tracker.last_progress_step
        ) <= policy.near_limit_window
        return near_limit and recent_progress and tracker.consecutive_no_progress_steps == 0

    def _tool_call_had_progress(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        observation: str,
        signature: Optional[str],
        tracker: LoopProgressTracker,
        files_before: int,
        files_after: int,
    ) -> bool:
        observation_lower = (observation or "").lower()
        if signature and signature not in tracker.seen_tool_signatures:
            tracker.seen_tool_signatures.add(signature)
            signature_progress = True
        else:
            signature_progress = False

        if observation == TASK_COMPLETE_SENTINEL:
            return True
        if files_after > files_before:
            return True
        if tool_name == "write_file" and not observation_lower.startswith("error:"):
            return True
        if (
            tool_name == "update_subtask"
            and bool(args.get("done"))
            and not observation_lower.startswith("error:")
        ):
            return True
        if tool_name == "run_command" and "exit_code: 0" in observation_lower:
            return True
        return signature_progress

    def _parse_tool_args(self, raw_arguments: str) -> Tuple[Dict[str, Any], Optional[str]]:
        raw = (raw_arguments or "").strip()
        if not raw:
            return {}, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, str(exc)
        if not isinstance(parsed, dict):
            return {}, "tool arguments must decode to a JSON object"
        return parsed, None

    def _extract_tool_calls(self, message: Any) -> List[Tuple[str, str, str]]:
        tool_calls = getattr(message, "tool_calls", None) or []
        items: List[Tuple[str, str, str]] = []
        for tc in tool_calls:
            call_id = str(getattr(tc, "id", "") or "")
            function = getattr(tc, "function", None)
            if function is None and isinstance(tc, dict):
                function = tc.get("function", {})
                call_id = call_id or str(tc.get("id", ""))

            if isinstance(function, dict):
                tool_name = str(function.get("name", "") or "")
                arguments = str(function.get("arguments", "") or "")
            else:
                tool_name = str(getattr(function, "name", "") or "")
                arguments = str(getattr(function, "arguments", "") or "")

            items.append((call_id, tool_name, arguments))
        return items

    def _tool_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        encoded_args = json.dumps(args, sort_keys=True, ensure_ascii=True)
        return f"{tool_name}:{encoded_args}"

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        if message is None:
            return {"role": "assistant", "content": ""}
        dump_method = getattr(message, "model_dump", None)
        if callable(dump_method):
            dumped = dump_method()
            if isinstance(dumped, dict):
                return dumped
        content = getattr(message, "content", "") or ""
        payload: Dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = self._extract_tool_calls(message)
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
                for call_id, name, args in tool_calls
            ]
        return payload

    def _compress_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= MAX_MESSAGES:
            return messages

        head = messages[:2]
        tail_len = max(1, MAX_MESSAGES - len(head) - 1)
        # Ensure the tail doesn't start with a 'tool' message — extend it
        # backwards to include the preceding assistant message with tool_calls
        # so the OpenAI API sees a valid tool_calls→tool sequence.
        cut = len(messages) - tail_len
        while cut > 2 and messages[cut].get("role") == "tool":
            cut -= 1
        tail = messages[cut:]
        middle = messages[2:cut]
        summary = self._summarize_middle(middle)
        return head + [{"role": "user", "content": summary}] + tail

    def _summarize_middle(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return "[Previous tool interactions summary]\n(no intermediate steps)"

        lines = ["[Previous tool interactions summary]"]
        for msg in messages:
            role = str(msg.get("role", ""))
            if role == "assistant" and msg.get("tool_calls"):
                for call in msg.get("tool_calls", []):
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name = str(function.get("name", "unknown"))
                    args_raw = str(function.get("arguments", ""))
                    args, _ = self._parse_tool_args(args_raw)
                    arg_keys = ",".join(sorted(args.keys())) if args else "-"
                    lines.append(f"- call {name} keys=[{arg_keys}]")
            elif role == "tool":
                content = str(msg.get("content", "")).replace("\n", " ").strip()
                lines.append(f"- result {content[:140]}")

        text = "\n".join(lines)
        if len(text) > MAX_SUMMARY_CHARS:
            return text[:MAX_SUMMARY_CHARS].rstrip() + " [truncated]"
        return text

    def _persist_output(self, task_id: str, output_text: str, workspace: Path) -> List[str]:
        workspace.mkdir(parents=True, exist_ok=True)
        files = self._extract_files(output_text)
        written: List[str] = []
        written_contents: Dict[str, str] = {}

        if files:
            for rel_path, content in files.items():
                safe_rel = self._safe_relative_path(rel_path)
                if safe_rel is None:
                    log.warning("Skipping unsafe generated path for task %s: %s", task_id, rel_path)
                    continue
                dest = workspace / safe_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                safe_key = str(safe_rel.as_posix())
                written.append(safe_key)
                written_contents[safe_key] = content

        if not written:
            fallback_name = f"{task_id}-output.md"
            fallback = workspace / fallback_name
            fallback.write_text(output_text or "", encoding="utf-8")
            written.append(fallback_name)
            written_contents[fallback_name] = output_text or ""

        review_rel = Path("reviews") / f"{task_id}-user-review.md"
        review_dest = workspace / review_rel
        review_dest.parent.mkdir(parents=True, exist_ok=True)
        review_dest.write_text(
            self._build_user_review_doc(
                task_id=task_id,
                output_text=output_text,
                files_written=written_contents,
            ),
            encoding="utf-8",
        )
        review_path = review_rel.as_posix()
        if review_path not in written:
            written.append(review_path)

        return written

    def _build_user_review_doc(
        self,
        *,
        task_id: str,
        output_text: str,
        files_written: Dict[str, str],
    ) -> str:
        rationale = self._extract_rationale(output_text)
        files = sorted(files_written.keys())

        lines = [
            f"# User Review Brief: {task_id}",
            "",
            f"- Generated at (UTC): {datetime.utcnow().isoformat()}",
            f"- Task ID: `{task_id}`",
            "",
            "## What Was Added",
        ]
        if files:
            lines.extend([f"- `{path}`" for path in files])
        else:
            lines.append("- No concrete files were detected from the agent output.")

        lines.extend(
            [
                "",
                "## Why This Approach",
                rationale,
                "",
                "## File & Function Overview",
            ]
        )

        if files:
            for path in files:
                content = files_written.get(path, "")
                purpose = self._infer_file_purpose(path, content)
                lines.extend(
                    [
                        "",
                        f"### `{path}`",
                        f"- Purpose: {purpose}",
                    ]
                )
                functions = self._extract_functions(path, content)
                if functions:
                    lines.append("- Functions:")
                    for fn in functions:
                        lines.append(f"  - `{fn['name']}`: {fn['purpose']}")
                else:
                    lines.append("- Functions: No explicit functions detected.")
        else:
            lines.append("")
            lines.append("No file-level function inventory is available for this task.")

        lines.extend(
            [
                "",
                "## Human Review Checklist",
                "1. Confirm file paths and contents match the task goal.",
                "2. Verify each listed function has correct behavior and naming.",
                "3. Validate rationale aligns with project constraints and acceptance criteria.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def _extract_rationale(self, output_text: str) -> str:
        if not output_text:
            return "No explicit rationale provided by the agent output."

        # Keep only explanatory prose outside fenced code blocks.
        text = _CODE_BLOCK_RE.sub("", output_text)
        text = "\n".join(
            line for line in text.splitlines() if not _PRELUDE_FILE_HINT_RE.match(line.strip())
        )
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return "No explicit rationale provided by the agent output."

        word_count = len(normalized.split())
        if word_count <= 8:
            return "No explicit rationale provided by the agent output."

        if len(normalized) > 1000:
            return normalized[:997].rstrip() + "..."
        return normalized

    def _extract_functions(self, path: str, content: str) -> List[Dict[str, str]]:
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            return self._extract_python_functions(content)
        if suffix in {".ts", ".tsx", ".js", ".jsx"}:
            return self._extract_js_functions(content)
        return []

    def _extract_python_functions(self, content: str) -> List[Dict[str, str]]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        entries: List[Dict[str, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entries.append(
                    {
                        "name": node.name,
                        "purpose": self._purpose_from_docstring_or_name(
                            ast.get_docstring(node), node.name
                        ),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                class_doc = self._purpose_from_docstring_or_name(ast.get_docstring(node), node.name)
                entries.append({"name": node.name, "purpose": f"Class: {class_doc}"})
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = f"{node.name}.{child.name}"
                        entries.append(
                            {
                                "name": method_name,
                                "purpose": self._purpose_from_docstring_or_name(
                                    ast.get_docstring(child),
                                    child.name,
                                ),
                            }
                        )
        return entries

    def _extract_js_functions(self, content: str) -> List[Dict[str, str]]:
        names: List[str] = []
        names.extend(_FUNC_DECL_RE.findall(content))
        names.extend(_ARROW_FUNC_RE.findall(content))
        names.extend(_CLASS_DECL_RE.findall(content))
        unique_names = list(dict.fromkeys(names))
        return [
            {
                "name": name,
                "purpose": self._purpose_from_docstring_or_name(None, name),
            }
            for name in unique_names
        ]

    def _purpose_from_docstring_or_name(self, docstring: Optional[str], name: str) -> str:
        if docstring:
            first_line = docstring.strip().splitlines()[0].strip()
            if first_line:
                return first_line
        words = self._humanize_identifier(name)
        return f"Handles {words}."

    def _infer_file_purpose(self, path: str, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return (
                    stripped.lstrip("#").strip()
                    or f"Implements {self._humanize_identifier(Path(path).stem)}."
                )
            if stripped.startswith("//"):
                return (
                    stripped.lstrip("/").strip()
                    or f"Implements {self._humanize_identifier(Path(path).stem)}."
                )
            if stripped.startswith("/*") or stripped.startswith("*"):
                cleaned = stripped.replace("/*", "").replace("*/", "").lstrip("*").strip()
                if cleaned:
                    return cleaned
                return f"Implements {self._humanize_identifier(Path(path).stem)}."
            break
        return f"Implements {self._humanize_identifier(Path(path).stem)}."

    def _humanize_identifier(self, value: str) -> str:
        if not value:
            return "core behavior"
        split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        normalized = split_camel.replace("_", " ").replace("-", " ")
        words = [w.lower() for w in normalized.split() if w]
        return " ".join(words) if words else "core behavior"

    def _extract_files(self, output_text: str) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if not output_text:
            return results

        for match in _CODE_BLOCK_RE.finditer(output_text):
            header = (match.group("header") or "").strip()
            body = match.group("body") or ""

            file_path = self._path_from_header(header)
            if not file_path:
                prelude = output_text[max(0, match.start() - 220) : match.start()]
                file_path = self._path_from_prelude(prelude)

            if not file_path:
                maybe_from_body, trimmed = self._path_from_body_first_line(body)
                file_path = maybe_from_body
                body = trimmed

            if not file_path:
                continue

            normalized = file_path.replace("\\", "/").strip()
            if normalized:
                results[normalized] = body

        return results

    def _path_from_header(self, header: str) -> Optional[str]:
        if not header:
            return None

        hint_match = _INLINE_FILE_HINT_RE.search(header)
        if hint_match:
            return hint_match.group(1).strip().strip("`\"'")

        tokens = [token for token in re.split(r"\s+", header) if token]
        for token in reversed(tokens):
            clean = token.strip("`\"'")
            if _LIKELY_FILE_RE.match(clean):
                return clean
        return None

    def _path_from_prelude(self, prelude: str) -> Optional[str]:
        if not prelude:
            return None
        lines = prelude.splitlines()[-3:]
        for line in reversed(lines):
            match = _PRELUDE_FILE_HINT_RE.match(line.strip())
            if match:
                return match.group(1).strip().strip("`\"'")
        return None

    def _path_from_body_first_line(self, body: str) -> tuple[Optional[str], str]:
        lines = body.splitlines()
        if not lines:
            return None, body
        first = lines[0]
        match = _BODY_FIRST_LINE_HINT_RE.match(first.strip())
        if not match:
            return None, body
        path = match.group(1).strip().strip("`\"'")
        trimmed = "\n".join(lines[1:]).lstrip("\n")
        return path, trimmed

    def _safe_relative_path(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return None
        if path.drive:
            return None
        normalized_parts = []
        for part in path.parts:
            if part in ("", "."):
                continue
            if part == "..":
                return None
            normalized_parts.append(part)
        if not normalized_parts:
            return None
        return Path(*normalized_parts)


def _parse_int_env(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))
