import asyncio
from pathlib import Path
from repro.repro_agent import ReproAgent


class DummyExecutor:
    def __init__(self, status="success"):
        self._status = status

    def available(self):
        return True

    def run(self, workdir: Path, commands, timeout_sec: int = 300, cache_dir=None):
        return {
            "status": self._status,
            "exit_code": 0 if self._status == "success" else 1,
            "logs": "ok",
            "duration_sec": 1.0,
        }


def test_repro_agent_no_docker():
    agent = ReproAgent({})
    agent.executor.client = None
    res = asyncio.get_event_loop().run_until_complete(agent.run(Path(".")))
    assert res["status"] == "error"


def test_repro_agent_with_dummy_executor(monkeypatch, tmp_path):
    agent = ReproAgent({})
    agent.executor = DummyExecutor()
    res = asyncio.get_event_loop().run_until_complete(agent.run(tmp_path))
    assert res["status"] == "success"
    assert res["score"] == 100

