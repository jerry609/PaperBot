"""
Memory evaluation module for Scope and Acceptance criteria.

Provides metrics collection and evaluation for:
- extraction_precision
- false_positive_rate
- retrieval_hit_rate
- injection_pollution_rate
- deletion_compliance
"""

from paperbot.memory.eval.collector import MemoryMetricCollector
from paperbot.memory.eval.injection_guard import (
    InjectionDetectionResult,
    detect_injection_patterns,
    normalize_injection_text,
)

__all__ = [
    "InjectionDetectionResult",
    "MemoryMetricCollector",
    "detect_injection_patterns",
    "normalize_injection_text",
]
