from __future__ import annotations

import pytest

from paperbot.infrastructure.swarm.worker_tools import _is_install_command


@pytest.mark.parametrize(
    "command",
    [
        "pip install numpy -q",
        "pip3 install pandas",
        "python -m pip install scipy",
        "python3 -m pip install torch",
        "apt-get install -y ffmpeg",
        "apt install git",
        "sudo apt-get install -y libgl1",
        "npm install typescript",
        "conda install pytorch",
    ],
)
def test_is_install_command_true(command: str):
    assert _is_install_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "npm run build",
        "python train.py",
        "pip list",
        "echo 'hello'",
    ],
)
def test_is_install_command_false(command: str):
    assert _is_install_command(command) is False
