from __future__ import annotations

from paperbot.infrastructure.swarm.worker_tools import LocalToolExecutor


def test_compress_install_output_keeps_key_lines(tmp_path):
    tool = LocalToolExecutor(workspace=tmp_path, sandbox=None)
    raw = "\n".join(
        [
            "Downloading wheel metadata...",
            "Collecting numpy",
            "Installing build dependencies ... done",
            "Successfully installed numpy-2.0.0",
            "random non-signal line",
            "WARNING: running pip as root",
        ]
    )

    compressed = tool._compress_install_output(raw)

    assert "Collecting numpy" in compressed
    assert "Successfully installed numpy-2.0.0" in compressed
    assert "WARNING: running pip as root" in compressed
    assert "random non-signal line" not in compressed


def test_compress_install_output_fallback_when_no_signal(tmp_path):
    tool = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    compressed = tool._compress_install_output("line a\nline b\nline c")

    assert compressed == "(install completed, no notable output)"
