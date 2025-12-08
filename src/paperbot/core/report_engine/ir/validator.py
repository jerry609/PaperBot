"""
简化 IR 校验器，检查必要字段与类型。
"""

from typing import Dict, Any, List

from .schema import ALLOWED_BLOCK_TYPES, ALLOWED_INLINE_MARKS


class IRValidationError(ValueError):
    """IR 验证错误。"""
    pass


class IRValidator:
    """IR 文档验证器。"""
    
    def validate_document(self, chapters: List[Dict[str, Any]]) -> None:
        """
        验证整个文档结构。
        
        Args:
            chapters: 章节列表
            
        Raises:
            IRValidationError: 验证失败时抛出
        """
        if not isinstance(chapters, list) or not chapters:
            raise IRValidationError("chapters 不能为空")
        for ch in chapters:
            self.validate_chapter(ch)

    def validate_chapter(self, chapter: Dict[str, Any]) -> None:
        """
        验证单个章节结构。
        
        Args:
            chapter: 章节字典
            
        Raises:
            IRValidationError: 验证失败时抛出
        """
        if "title" not in chapter or "blocks" not in chapter:
            raise IRValidationError("chapter 缺少 title/blocks")
        blocks = chapter.get("blocks", [])
        if not isinstance(blocks, list) or not blocks:
            raise IRValidationError("chapter.blocks 不能为空")
        for b in blocks:
            self.validate_block(b)

    def validate_block(self, block: Dict[str, Any]) -> None:
        """
        验证单个块结构。
        
        Args:
            block: 块字典
            
        Raises:
            IRValidationError: 验证失败时抛出
        """
        btype = block.get("type")
        if btype not in ALLOWED_BLOCK_TYPES:
            raise IRValidationError(f"不支持的块类型: {btype}")
        content = block.get("content", "")
        if not isinstance(content, (str, list)):
            raise IRValidationError("block.content 需为 str/list")
        marks = block.get("marks", [])
        if marks:
            if not isinstance(marks, list) or not all(m in ALLOWED_INLINE_MARKS for m in marks):
                raise IRValidationError("marks 非法")

