# Reports Module
# 报告生成模块

from .writer import ReportWriter
from .notifier import Notifier
from .core import (
    IR_VERSION,
    SECTION_ORDER_STEP,
    TemplateSection,
    ChapterRecord,
    DocumentComposer,
    ChapterStorage,
    parse_template_sections,
    slugify,
)

__all__ = [
    "ReportWriter",
    "Notifier",
    # 核心组件
    "IR_VERSION",
    "SECTION_ORDER_STEP",
    "TemplateSection",
    "ChapterRecord",
    "DocumentComposer",
    "ChapterStorage",
    "parse_template_sections",
    "slugify",
]
