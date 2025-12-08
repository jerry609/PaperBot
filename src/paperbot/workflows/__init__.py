"""
工作流定义层 - 业务流程编排。
"""

from .scholar_tracking import ScholarTrackingWorkflow

# 兼容层：也导出原有的 ScholarWorkflowCoordinator
try:
    from core.workflow_coordinator import ScholarWorkflowCoordinator
except ImportError:
    ScholarWorkflowCoordinator = None

__all__ = [
    "ScholarTrackingWorkflow",
    "ScholarWorkflowCoordinator",
]
