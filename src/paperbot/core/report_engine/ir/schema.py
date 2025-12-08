"""
简化 IR Schema 定义。

定义报告中间表示的块类型和内联标记。
"""

from typing import List, Set

# 允许的块类型与内联标记（可按需扩展）
ALLOWED_BLOCK_TYPES: Set[str] = {
    "heading",
    "paragraph",
    "bullet_list",
    "number_list",
    "quote",
    "code",
}

ALLOWED_INLINE_MARKS: Set[str] = {
    "bold",
    "italic",
    "code",
    "link",
    "underline",
    "strike",
}


def default_chapter_template(slug: str, title: str, body: str) -> dict:
    """
    创建默认章节模板。
    
    Args:
        slug: 章节标识符
        title: 章节标题
        body: 章节正文
        
    Returns:
        章节字典结构
    """
    return {
        "slug": slug,
        "title": title,
        "blocks": [
            {
                "type": "heading",
                "content": title,
                "level": 2,
            },
            {
                "type": "paragraph",
                "content": body,
            },
        ],
    }

