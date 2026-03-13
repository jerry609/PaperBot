from __future__ import annotations

from pathlib import Path

from paperbot.repro import e2b_executor as e2b_mod


def test_e2b_executor_reads_template_from_env(monkeypatch):
    monkeypatch.setenv("E2B_TEMPLATE", "paperbot-repro-v2")
    executor = e2b_mod.E2BExecutor(api_key="test-key")

    assert executor.template == "paperbot-repro-v2"


def test_e2b_executor_reads_timeout_from_env(monkeypatch):
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT_SECONDS", "1800")
    executor = e2b_mod.E2BExecutor(api_key="test-key")

    assert executor.timeout_sandbox == 1800


def test_e2b_run_uses_template_when_supported(monkeypatch, tmp_path):
    calls: list[dict] = []

    class _FakeCommandResult:
        stdout = "ok"
        stderr = ""
        exit_code = 0

    class _FakeCommands:
        def run(self, _cmd, timeout=300, cwd="/home/user"):
            return _FakeCommandResult()

    class _FakeSandboxInstance:
        def __init__(self):
            self.commands = _FakeCommands()
            self.files = None

        def kill(self):
            return None

    class _FakeSandboxFactory:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            return _FakeSandboxInstance()

    monkeypatch.setattr(e2b_mod, "HAS_E2B", True)
    monkeypatch.setattr(e2b_mod, "Sandbox", _FakeSandboxFactory)

    executor = e2b_mod.E2BExecutor(api_key="test-key", template="paperbot-repro")
    result = executor.run(
        workdir=Path(tmp_path),
        commands=["python -c \"print('ok')\""],
    )

    assert result.status == "success"
    assert calls
    assert calls[0].get("template") == "paperbot-repro"


def test_e2b_run_captures_command_exception_details(monkeypatch, tmp_path):
    class _FakeCommandError(Exception):
        def __init__(self):
            super().__init__("Command exited with code 2 and error:")
            self.exit_code = 2
            self.stdout = "collected 0 items"
            self.stderr = "ModuleNotFoundError: No module named 'pytest'"

    class _FakeCommands:
        def run(self, _cmd, timeout=300, cwd="/home/user"):
            raise _FakeCommandError()

    class _FakeSandboxInstance:
        def __init__(self):
            self.commands = _FakeCommands()
            self.files = None

        def kill(self):
            return None

    class _FakeSandboxFactory:
        @staticmethod
        def create(timeout=300):
            return _FakeSandboxInstance()

    monkeypatch.setattr(e2b_mod, "HAS_E2B", True)
    monkeypatch.setattr(e2b_mod, "Sandbox", _FakeSandboxFactory)

    executor = e2b_mod.E2BExecutor(api_key="test-key")
    result = executor.run(
        workdir=Path(tmp_path),
        commands=["pytest -q"],
    )

    assert result.status == "failed"
    assert result.exit_code == 2
    assert "[stdout] collected 0 items" in result.logs
    assert "[stderr] ModuleNotFoundError: No module named 'pytest'" in result.logs
    assert "[error] Command exited with code 2 and error:" in result.logs


def test_e2b_keep_alive_reuses_same_sandbox(monkeypatch, tmp_path):
    class _FakeCommandResult:
        def __init__(self):
            self.stdout = "ok"
            self.stderr = ""
            self.exit_code = 0

    class _FakeCommands:
        def run(self, _cmd, timeout=300, cwd="/home/user"):
            return _FakeCommandResult()

    class _FakeSandboxInstance:
        def __init__(self, sid: str):
            self.id = sid
            self.commands = _FakeCommands()
            self.files = None
            self.killed = 0

        def kill(self):
            self.killed += 1
            return None

    class _FakeSandboxFactory:
        created = 0
        latest = None

        @staticmethod
        def create(**_kwargs):
            _FakeSandboxFactory.created += 1
            sid = f"sbx-{_FakeSandboxFactory.created}"
            instance = _FakeSandboxInstance(sid)
            _FakeSandboxFactory.latest = instance
            return instance

    monkeypatch.setattr(e2b_mod, "HAS_E2B", True)
    monkeypatch.setattr(e2b_mod, "Sandbox", _FakeSandboxFactory)

    executor = e2b_mod.E2BExecutor(api_key="test-key", keep_alive=True)
    first = executor.run(workdir=Path(tmp_path), commands=["echo 1"])
    second = executor.run(workdir=Path(tmp_path), commands=["echo 2"])

    assert first.status == "success"
    assert second.status == "success"
    assert _FakeSandboxFactory.created == 1
    assert executor.sandbox_id == "sbx-1"
    assert first.runtime_meta.get("sandbox_id") == "sbx-1"
    assert second.runtime_meta.get("sandbox_id") == "sbx-1"
    assert _FakeSandboxFactory.latest.killed == 0

    executor.cleanup()
    assert _FakeSandboxFactory.latest.killed == 1


def test_e2b_attach_existing_sandbox(monkeypatch, tmp_path):
    class _FakeCommandResult:
        stdout = "connected"
        stderr = ""
        exit_code = 0

    class _FakeCommands:
        def run(self, _cmd, timeout=300, cwd="/home/user"):
            return _FakeCommandResult()

    class _FakeSandboxInstance:
        def __init__(self, sid: str):
            self.sandbox_id = sid
            self.commands = _FakeCommands()
            self.files = None

        def kill(self):
            return None

    class _FakeSandboxFactory:
        @staticmethod
        def create(**_kwargs):
            raise AssertionError("create should not be called when attach succeeds")

        @staticmethod
        def connect(sid):
            return _FakeSandboxInstance(sid)

    monkeypatch.setattr(e2b_mod, "HAS_E2B", True)
    monkeypatch.setattr(e2b_mod, "Sandbox", _FakeSandboxFactory)

    executor = e2b_mod.E2BExecutor(api_key="test-key", keep_alive=True)
    assert executor.attach_sandbox("sbx-existing") is True
    assert executor.sandbox_id == "sbx-existing"

    result = executor.run(workdir=Path(tmp_path), commands=["echo hello"])
    assert result.status == "success"
    assert result.runtime_meta.get("sandbox_id") == "sbx-existing"


def test_e2b_run_recovers_from_expired_sandbox(monkeypatch, tmp_path):
    class _SandboxExpiredError(Exception):
        pass

    class _FakeCommandResult:
        stdout = "ok"
        stderr = ""
        exit_code = 0

    class _FakeCommands:
        def __init__(self, *, fail_first: bool):
            self.fail_first = fail_first
            self.calls = 0

        def run(self, _cmd, timeout=300, cwd="/home/user"):
            self.calls += 1
            if self.fail_first and self.calls == 1:
                raise _SandboxExpiredError(
                    "The sandbox was not found: This error is likely due to sandbox timeout."
                )
            return _FakeCommandResult()

    class _FakeSandboxInstance:
        def __init__(self, sid: str, *, fail_first: bool = False):
            self.id = sid
            self.commands = _FakeCommands(fail_first=fail_first)
            self.files = None
            self.timeouts = []

        def set_timeout(self, value):
            self.timeouts.append(value)
            return None

        def kill(self):
            return None

    class _FakeSandboxFactory:
        created = 0

        @staticmethod
        def create(**_kwargs):
            _FakeSandboxFactory.created += 1
            sid = f"sbx-{_FakeSandboxFactory.created}"
            return _FakeSandboxInstance(
                sid,
                fail_first=(_FakeSandboxFactory.created == 1),
            )

    monkeypatch.setattr(e2b_mod, "HAS_E2B", True)
    monkeypatch.setattr(e2b_mod, "Sandbox", _FakeSandboxFactory)

    executor = e2b_mod.E2BExecutor(api_key="test-key", keep_alive=True)
    result = executor.run(workdir=Path(tmp_path), commands=["pytest -q tests"])

    assert result.status == "success"
    assert result.runtime_meta.get("sandbox_recovered") is True
    assert result.runtime_meta.get("sandbox_id") == "sbx-2"
    assert _FakeSandboxFactory.created == 2


def test_e2b_upload_files_creates_nested_directories(tmp_path):
    nested = tmp_path / "src" / "models"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "vae.py").write_text("class VAE:\n    pass\n", encoding="utf-8")

    class _FakeFS:
        def __init__(self):
            self.dirs = {"/home/user"}
            self.writes: dict[str, str] = {}

        def make_dir(self, path: str):
            parent = str(Path(path).parent)
            if parent not in self.dirs:
                raise RuntimeError(f"missing parent: {parent}")
            self.dirs.add(path)

        def write(self, path: str, content: str):
            parent = str(Path(path).parent)
            if parent not in self.dirs:
                raise RuntimeError(f"missing parent for write: {parent}")
            self.writes[path] = content

    sandbox = type("_Sandbox", (), {"files": _FakeFS()})()
    executor = e2b_mod.E2BExecutor(api_key="test-key")
    uploaded = executor._upload_files(sandbox, Path(tmp_path))

    assert "src/models/vae.py" in uploaded
    assert "/home/user/src/models/vae.py" in sandbox.files.writes
