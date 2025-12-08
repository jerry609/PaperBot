"""
简化 IR Schema 定义。
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

