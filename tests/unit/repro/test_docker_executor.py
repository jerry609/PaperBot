"""Unit tests for DockerExecutor — workspace_read_only mount mode."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperbot.repro.docker_executor import DockerExecutor


def _make_executor(*, workspace_read_only: bool = True) -> DockerExecutor:
    """Create a DockerExecutor with a mocked Docker client."""
    with patch("paperbot.repro.docker_executor.HAS_DOCKER", True):
        executor = DockerExecutor.__new__(DockerExecutor)
        executor.image = "test:latest"
        executor.cpu_shares = 1
        executor.mem_limit = "1g"
        executor.network_disabled = True
        executor.workspace_read_only = workspace_read_only

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"ok"
        mock_client.containers.run.return_value = mock_container
        executor.client = mock_client
    return executor


class TestWorkspaceReadOnly:
    def test_default_is_read_only(self):
        with patch("paperbot.repro.docker_executor.HAS_DOCKER", True), \
             patch("paperbot.repro.docker_executor.docker") as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            executor = DockerExecutor(image="test:latest")
            assert executor.workspace_read_only is True

    def test_mount_mode_ro_when_read_only(self, tmp_path: Path):
        executor = _make_executor(workspace_read_only=True)
        executor.run(tmp_path, ["echo hello"])

        call_kwargs = executor.client.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        assert volumes[str(tmp_path)]["mode"] == "ro"

    def test_mount_mode_rw_when_writable(self, tmp_path: Path):
        executor = _make_executor(workspace_read_only=False)
        executor.run(tmp_path, ["echo hello"])

        call_kwargs = executor.client.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        assert volumes[str(tmp_path)]["mode"] == "rw"

    def test_runtime_meta_records_read_only_flag(self, tmp_path: Path):
        executor = _make_executor(workspace_read_only=True)
        result = executor.run(tmp_path, ["echo hello"], record_meta=True)
        assert result.runtime_meta["workspace_read_only"] is True

        executor_rw = _make_executor(workspace_read_only=False)
        result_rw = executor_rw.run(tmp_path, ["echo hello"], record_meta=True)
        assert result_rw.runtime_meta["workspace_read_only"] is False
