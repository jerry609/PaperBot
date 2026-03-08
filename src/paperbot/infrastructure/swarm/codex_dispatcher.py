"""
Codex Dispatcher -- sends coding tasks to OpenAI Codex API (cloud).

Uses principle-driven prompts (concise, goal-oriented) as Codex responds
better to this style than mechanics-driven prompts.
"""

import asyncio
import ast
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class CodexResult:
    task_id: str
    success: bool
    output: str = ""
    files_generated: List[str] = field(default_factory=list)
    error: Optional[str] = None


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

    async def dispatch_parallel(
        self, tasks: List[Dict[str, Any]], workspace: Path
    ) -> List[CodexResult]:
        """Dispatch multiple tasks concurrently."""
        coros = [self.dispatch(t["task_id"], t["prompt"], workspace) for t in tasks]
        return await asyncio.gather(*coros, return_exceptions=False)

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
