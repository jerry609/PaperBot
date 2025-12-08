# repro/nodes/__init__.py
"""
Node-based processing pipeline for ReproAgent.
Inspired by BettaFish's node architecture.
"""

from .base_node import BaseNode, NodeResult

__all__ = [
    "BaseNode",
    "NodeResult",
]
