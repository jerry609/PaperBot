"""Tests for E2E execution -- runs the paper's main entry point in sandbox."""

import pytest

from paperbot.infrastructure.swarm.e2e_execution import (
    E2EExecutionPolicy,
    E2EResult,
    build_run_command,
    detect_entry_point,
    run_e2e,
    run_e2e_with_repair,
)
from paperbot.infrastructure.swarm.shared_sandbox import SharedSandbox


class _FakeFS:
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


class _CmdResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeSandbox:
    def __init__(self, file_list=None, run_results=None):
        self.files = _FakeFS()
        self._file_list = file_list or []
        self._run_results = run_results or []
        self._run_call_count = 0
        self._commands = []

    @property
    def commands(self):
        return self

    def run(self, command, timeout=120, cwd="/home/user"):
        self._commands.append((command, cwd))
        if command.startswith("find") and "-maxdepth" in command:
            prefix = cwd.rstrip("/")
            lines = [f"{prefix}/{f}" for f in self._file_list]
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("find") and "-type f" in command:
            prefix = cwd.rstrip("/")
            lines = [f"{prefix}/{f}" for f in self._file_list]
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("mkdir"):
            return _CmdResult(0, "", "")
        # Return queued results for program execution
        if self._run_results:
            idx = min(self._run_call_count, len(self._run_results) - 1)
            self._run_call_count += 1
            r = self._run_results[idx]
            return _CmdResult(r[0], r[1], r[2])
        return _CmdResult(0, "", "")


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True


def _make_shared(file_list=None, run_results=None):
    raw = _FakeSandbox(file_list=file_list, run_results=run_results)
    executor = _FakeExecutor(raw)
    return SharedSandbox(executor), raw


# ---------------------------------------------------------------------------
# detect_entry_point
# ---------------------------------------------------------------------------


class TestDetectEntryPoint:
    def test_detects_main_py(self):
        shared, _ = _make_shared(["main.py", "utils.py", "requirements.txt"])
        result = detect_entry_point(shared, "slug")
        assert result == "main.py"

    def test_detects_train_py(self):
        shared, _ = _make_shared(["train.py", "model.py"])
        result = detect_entry_point(shared, "slug")
        assert result == "train.py"

    def test_detects_run_sh(self):
        shared, _ = _make_shared(["run.sh", "model.py"])
        result = detect_entry_point(shared, "slug")
        # main.py/train.py not present, so should find run.sh
        assert result == "run.sh"

    def test_returns_none_for_empty(self):
        shared, _ = _make_shared([])
        result = detect_entry_point(shared, "slug")
        assert result is None

    def test_prefers_main_over_train(self):
        shared, _ = _make_shared(["train.py", "main.py"])
        result = detect_entry_point(shared, "slug")
        assert result == "main.py"

    def test_detects_run_experiment(self):
        shared, _ = _make_shared(["run_experiment.py", "config.yaml"])
        result = detect_entry_point(shared, "slug")
        assert result == "run_experiment.py"

    def test_detect_entry_point_ignores_commented_main_guard(self):
        shared, raw = _make_shared(
            [
                "helpers.py",
                "pipeline.py",
                "launcher.py",
            ]
        )
        raw.files._files["/home/user/slug/helpers.py"] = (
            "# if __name__ == '__main__':\n"
            "#     print('comment only')\n"
        )
        raw.files._files["/home/user/slug/pipeline.py"] = "def run():\n    return 1\n"
        raw.files._files["/home/user/slug/launcher.py"] = (
            "def main():\n"
            "    print('real main')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        result = detect_entry_point(shared, "slug")
        assert result == "launcher.py"


# ---------------------------------------------------------------------------
# build_run_command
# ---------------------------------------------------------------------------


class TestBuildRunCommand:
    def test_python_file(self):
        policy = E2EExecutionPolicy(entry_point="main.py")
        assert build_run_command(policy) == "python main.py"

    def test_shell_script(self):
        policy = E2EExecutionPolicy(entry_point="run.sh")
        assert build_run_command(policy) == "bash run.sh"

    def test_explicit_command(self):
        policy = E2EExecutionPolicy(entry_command="python -m mypackage --config=exp1")
        assert build_run_command(policy) == "python -m mypackage --config=exp1"

    def test_makefile_target(self):
        policy = E2EExecutionPolicy(entry_point="Makefile:train")
        assert build_run_command(policy) == "make train"

    def test_command_overrides_entry_point(self):
        policy = E2EExecutionPolicy(
            entry_point="main.py",
            entry_command="bash custom_run.sh",
        )
        assert build_run_command(policy) == "bash custom_run.sh"


# ---------------------------------------------------------------------------
# E2EExecutionPolicy.from_context
# ---------------------------------------------------------------------------


class TestE2EPolicy:
    def test_auto_detect(self):
        shared, _ = _make_shared(["main.py", "requirements.txt"])
        policy = E2EExecutionPolicy.from_context(shared, "slug")
        assert policy.enabled is True
        assert policy.entry_point == "main.py"

    def test_disabled_when_no_entry_point(self):
        shared, _ = _make_shared(["README.md"])
        policy = E2EExecutionPolicy.from_context(shared, "slug")
        assert policy.enabled is False

    def test_context_pack_entry_command(self):
        shared, _ = _make_shared([])
        policy = E2EExecutionPolicy.from_context(
            shared, "slug",
            context_pack={"entry_command": "python train.py --epochs 100"},
        )
        assert policy.enabled is True
        assert policy.entry_command == "python train.py --epochs 100"

    def test_context_pack_entry_point(self):
        shared, _ = _make_shared([])
        policy = E2EExecutionPolicy.from_context(
            shared, "slug",
            context_pack={"entry_point": "experiment.py"},
        )
        assert policy.enabled is True
        assert policy.entry_point == "experiment.py"


# ---------------------------------------------------------------------------
# run_e2e
# ---------------------------------------------------------------------------


class TestRunE2E:
    def test_success(self):
        shared, _ = _make_shared(
            file_list=["main.py"],
            run_results=[(0, "Accuracy: 0.95\nLoss: 0.05", "")],
        )
        policy = E2EExecutionPolicy(entry_point="main.py", install_deps=False)
        result = run_e2e(shared, "slug", policy)
        assert result.success is True
        assert "Accuracy" in result.stdout

    def test_failure(self):
        shared, _ = _make_shared(
            file_list=["main.py"],
            run_results=[(1, "", "ModuleNotFoundError: No module named 'torch'")],
        )
        policy = E2EExecutionPolicy(entry_point="main.py", install_deps=False)
        result = run_e2e(shared, "slug", policy)
        assert result.success is False
        assert result.exit_code == 1

    def test_installs_deps_on_first_attempt(self):
        shared, raw = _make_shared(
            file_list=["main.py", "requirements.txt"],
            run_results=[
                (0, "", ""),  # pip install
                (0, "done", ""),  # main.py
            ],
        )
        policy = E2EExecutionPolicy(entry_point="main.py", install_deps=True)
        result = run_e2e(shared, "slug", policy, attempt=0)
        # pip install should have been called
        pip_calls = [c for c, _ in raw._commands if "pip install" in c]
        assert len(pip_calls) >= 1

    def test_deps_reinstalled_on_repair_attempt(self):
        shared, raw = _make_shared(
            file_list=["main.py", "requirements.txt"],
            run_results=[
                (0, "", ""),  # pip install on retry
                (0, "retry run success", ""),  # program run
            ],
        )
        policy = E2EExecutionPolicy(entry_point="main.py", install_deps=True)
        result = run_e2e(shared, "slug", policy, attempt=1)
        assert result.success is True
        pip_calls = [c for c, _ in raw._commands if "pip install -q -r requirements.txt" in c]
        assert len(pip_calls) >= 1

    def test_output_truncation_keeps_tail(self):
        tail_marker = "TRACEBACK_LAST_LINE: module import failed"
        long_output = ("progress...\n" * 3000) + f"\n{tail_marker}"
        shared, _ = _make_shared(
            file_list=["main.py"],
            run_results=[(1, long_output, "")],
        )
        policy = E2EExecutionPolicy(entry_point="main.py", install_deps=False)
        result = run_e2e(shared, "slug", policy)
        assert result.success is False
        assert result.stdout.startswith("...[truncated]\n")
        assert tail_marker in result.stdout


# ---------------------------------------------------------------------------
# run_e2e_with_repair
# ---------------------------------------------------------------------------


class _FakeDispatcher:
    def __init__(self):
        self.repair_prompts = []

    async def dispatch_with_sandbox_tools(
        self, task_id, prompt, tool_executor, on_step=None, max_iterations=25
    ):
        self.repair_prompts.append(prompt)


class _FakeSandboxToolExecutor:
    def __init__(self, sandbox, slug):
        self.sandbox = sandbox
        self.slug = slug
        self.files_written = []
        self.tool_log = []


@pytest.mark.asyncio
async def test_e2e_repair_loop_succeeds_on_retry():
    """First run fails, repair fixes it, second run succeeds."""
    # First run: fail. Second run (after repair): succeed.
    shared, raw = _make_shared(
        file_list=["main.py"],
        run_results=[
            (1, "ImportError: No module 'model'", ""),  # first e2e attempt
            (0, "Accuracy: 0.92", ""),  # second e2e attempt (after repair)
        ],
    )
    policy = E2EExecutionPolicy(
        entry_point="main.py",
        max_repair_attempts=2,
        install_deps=False,
    )
    dispatcher = _FakeDispatcher()
    attempts = []

    async def on_attempt(result):
        attempts.append(result)

    result = await run_e2e_with_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=dispatcher,
        tool_executor_factory=lambda: _FakeSandboxToolExecutor(shared, "slug"),
        on_attempt=on_attempt,
    )

    assert result.success is True
    assert result.attempt == 1
    assert len(dispatcher.repair_prompts) == 1
    assert "ImportError" in dispatcher.repair_prompts[0]
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_e2e_repair_loop_exhausted():
    """All attempts fail -- returns last failed result."""
    shared, _ = _make_shared(
        file_list=["main.py"],
        run_results=[
            (1, "Error 1", ""),
            (1, "Error 2", ""),
            (1, "Error 3", ""),
            (1, "Error 4", ""),
        ],
    )
    policy = E2EExecutionPolicy(
        entry_point="main.py",
        max_repair_attempts=2,
        install_deps=False,
    )
    dispatcher = _FakeDispatcher()

    result = await run_e2e_with_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=dispatcher,
        tool_executor_factory=lambda: _FakeSandboxToolExecutor(shared, "slug"),
    )

    assert result.success is False
    assert len(dispatcher.repair_prompts) == 2  # 2 repair attempts


@pytest.mark.asyncio
async def test_e2e_succeeds_first_try():
    """No repair needed when first run succeeds."""
    shared, _ = _make_shared(
        file_list=["train.py"],
        run_results=[(0, "Training complete. Loss: 0.01", "")],
    )
    policy = E2EExecutionPolicy(
        entry_point="train.py",
        max_repair_attempts=3,
        install_deps=False,
    )
    dispatcher = _FakeDispatcher()

    result = await run_e2e_with_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=dispatcher,
        tool_executor_factory=lambda: _FakeSandboxToolExecutor(shared, "slug"),
    )

    assert result.success is True
    assert result.attempt == 0
    assert len(dispatcher.repair_prompts) == 0


@pytest.mark.asyncio
async def test_repair_dispatch_exception_returns_last_result():
    shared, _ = _make_shared(
        file_list=["main.py"],
        run_results=[(1, "RuntimeError: boom", "")],
    )
    policy = E2EExecutionPolicy(
        entry_point="main.py",
        max_repair_attempts=2,
        install_deps=False,
    )

    class _ExplodingDispatcher(_FakeDispatcher):
        async def dispatch_with_sandbox_tools(
            self, task_id, prompt, tool_executor, on_step=None, max_iterations=25
        ):
            raise RuntimeError("dispatcher unavailable")

    dispatcher = _ExplodingDispatcher()
    result = await run_e2e_with_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=dispatcher,
        tool_executor_factory=lambda: _FakeSandboxToolExecutor(shared, "slug"),
    )

    assert result.success is False
    assert result.attempt == 0
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_e2e_with_negative_repair_attempts_returns_defensive_result():
    shared, _ = _make_shared(
        file_list=["main.py"],
        run_results=[],
    )
    policy = E2EExecutionPolicy(
        entry_point="main.py",
        max_repair_attempts=-1,
        install_deps=False,
    )
    dispatcher = _FakeDispatcher()
    result = await run_e2e_with_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=dispatcher,
        tool_executor_factory=lambda: _FakeSandboxToolExecutor(shared, "slug"),
    )

    assert result.success is False
    assert result.exit_code == 1
    assert "No execution attempts were made" in result.stderr
