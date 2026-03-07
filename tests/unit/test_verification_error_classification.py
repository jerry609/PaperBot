from __future__ import annotations

from paperbot.repro.agents.debugging_agent import DebuggingAgent
from paperbot.repro.models import ErrorType
from paperbot.repro.nodes.verification_node import ErrorClassifier


def test_debugging_agent_treats_cannot_import_name_as_logic():
    agent = DebuggingAgent()
    error_type, detail = agent._classify_error(
        "ImportError: cannot import name 'DataLoader' from 'data'"
    )

    assert error_type == ErrorType.LOGIC
    assert detail == "DataLoader"


def test_verification_error_classifier_treats_cannot_import_name_as_logic():
    error_type, detail = ErrorClassifier.classify(
        "ImportError: cannot import name 'DataLoader' from 'data'"
    )

    assert error_type == ErrorType.LOGIC
    assert detail == "DataLoader"
