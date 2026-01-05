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

__all__ = ["MemoryMetricCollector"]
