"""
IR (Intermediate Representation) 模块。

提供报告中间表示的 Schema 定义和验证功能。
"""

from .schema import (
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    default_chapter_template,
)
from .validator import IRValidator, IRValidationError

__all__ = [
    "ALLOWED_BLOCK_TYPES",
    "ALLOWED_INLINE_MARKS",
    "default_chapter_template",
    "IRValidator",
    "IRValidationError",
]

