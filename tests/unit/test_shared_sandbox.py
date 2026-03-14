"""Tests for SharedSandbox -- VM-native file operations."""

import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperbot.infrastructure.swarm.shared_sandbox import SharedSandbox


class _FakeFS:
    """In-memory filesystem mock for E2B sandbox."""

    def __init__(self):
        self._files = {}

    def read(self, path):
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    def write(self, path, content):
        self._files[path] = content

    def make_dir(self, path):
        pass


class _FakeSandbox:
    """Minimal E2B sandbox mock."""

    def __init__(self):
        self.files = _FakeFS()
        self._commands = []

    @property
    def commands(self):
        return self

    def run(self, command, timeout=120, cwd="/home/user"):
        self._commands.append((command, cwd))
        # Simulate find command
        if command.startswith("find"):
            lines = []
            prefix = cwd.rstrip("/")
            for k in sorted(self.files._files):
                if k.startswith(prefix + "/"):
                    lines.append(k)
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("ls -1d"):
            return _CmdResult(0, "", "")
        if command.startswith("grep"):
            return _CmdResult(0, "(no matches)", "")
        if command.startswith("mkdir"):
            return _CmdResult(0, "", "")
        if command.startswith("rm"):
            return _CmdResult(0, "", "")
        return _CmdResult(0, f"ran: {command}", "")


class _CmdResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True

    @property
    def executor_type(self):
        return "fake"


def _make_sandbox():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)
    return shared, raw


class TestSharedSandbox:
    def test_alive(self):
        shared, _ = _make_sandbox()
        assert shared.alive is True

    def test_alive_false_when_no_executor(self):
        shared = SharedSandbox(MagicMock(available=lambda: False))
        assert shared.alive is False

    def test_paper_root(self):
        shared, _ = _make_sandbox()
        assert shared.paper_root("attn-a9b1") == "/home/user/attn-a9b1"

    def test_write_and_read(self):
        shared, raw = _make_sandbox()
        ok = shared.write_file("slug", "src/main.py", "print('hello')")
        assert ok is True
        content = shared.read_file("slug", "src/main.py")
        assert content == "print('hello')"

    def test_read_nonexistent(self):
        shared, _ = _make_sandbox()
        assert shared.read_file("slug", "nope.py") is None

    def test_write_read_bytes_response(self):
        shared, raw = _make_sandbox()
        raw.files._files["/home/user/slug/data.txt"] = b"binary content"
        content = shared.read_file("slug", "data.txt")
        assert content == "binary content"

    def test_run_in_paper(self):
        shared, raw = _make_sandbox()
        result = shared.run_in_paper("slug", "echo hello")
        assert result.exit_code == 0
        assert ("echo hello", "/home/user/slug") in raw._commands

    def test_ensure_paper_dir(self):
        shared, raw = _make_sandbox()
        ok = shared.ensure_paper_dir("my-paper-abc1")
        assert ok is True
        assert any("mkdir" in cmd for cmd, _ in raw._commands)

    def test_list_files_recursive(self):
        shared, raw = _make_sandbox()
        raw.files._files["/home/user/slug/a.py"] = "a"
        raw.files._files["/home/user/slug/sub/b.py"] = "b"
        files = shared.list_files_recursive("slug")
        assert "a.py" in files
        assert "sub/b.py" in files

    def test_download_paper(self, tmp_path):
        shared, raw = _make_sandbox()
        raw.files._files["/home/user/slug/src/main.py"] = "code"
        raw.files._files["/home/user/slug/.plan/roadmap.md"] = "plan"
        raw.files._files["/home/user/slug/.status/task-1.md"] = "status"
        raw.files._files["/home/user/slug/.knowledge/notes.md"] = "knowledge"
        raw.files._files["/home/user/slug/__pycache__/mod.pyc"] = "cache"

        downloaded = shared.download_paper("slug", tmp_path / "local")
        assert "src/main.py" in downloaded
        assert ".plan/roadmap.md" in downloaded
        assert ".status/task-1.md" in downloaded
        assert ".knowledge/notes.md" in downloaded
        assert "__pycache__/mod.pyc" not in downloaded  # skipped
        assert (tmp_path / "local" / "src" / "main.py").read_text() == "code"
        assert (tmp_path / "local" / ".plan" / "roadmap.md").read_text() == "plan"

    def test_cross_paper_isolation(self):
        shared, raw = _make_sandbox()
        shared.write_file("paper-a", "secret.txt", "a-data")
        shared.write_file("paper-b", "other.txt", "b-data")
        # paper-a should not see paper-b's files
        assert shared.read_file("paper-a", "other.txt") is None
        assert shared.read_file("paper-b", "secret.txt") is None

    def test_search_files(self):
        shared, _ = _make_sandbox()
        result = shared.search_files("slug", "pattern")
        assert isinstance(result, str)

    def test_teardown(self):
        shared, _ = _make_sandbox()
        shared.teardown()  # Should not raise

    def test_list_files_with_special_chars_in_slug(self):
        shared, raw = _make_sandbox()
        slug = "paper with spaces'; rm -rf / #"
        shared.list_files(slug, ".")
        command, _ = raw._commands[-1]
        target = f"/home/user/{slug}"
        assert f"find {shlex.quote(target)} -maxdepth 1" in command
