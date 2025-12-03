"""
报告核心模块

参考: BettaFish/ReportEngine/core/
适配: PaperBot 学者追踪报告生成

包含:
- DocumentComposer: 文档装订器
- TemplateParser: 模板解析器
- ChapterStorage: 章节存储管理
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


# ===== IR 版本 =====
IR_VERSION = "1.0.0"


# ===== 模板章节定义 =====

@dataclass
class TemplateSection:
    """
    模板章节实体
    
    记录标题、slug、序号、层级、原始标题、章节编号与提纲
    """
    title: str
    slug: str
    order: int
    depth: int
    raw_title: str
    number: str = ""
    chapter_id: str = ""
    outline: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "title": self.title,
            "slug": self.slug,
            "order": self.order,
            "depth": self.depth,
            "number": self.number,
            "chapterId": self.chapter_id,
            "outline": self.outline,
        }


# ===== 章节记录 =====

@dataclass
class ChapterRecord:
    """章节元数据记录"""
    chapter_id: str
    slug: str
    title: str
    order: int
    status: str
    files: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "chapterId": self.chapter_id,
            "slug": self.slug,
            "title": self.title,
            "order": self.order,
            "status": self.status,
            "files": self.files,
            "errors": self.errors,
            "updatedAt": self.updated_at,
        }


# ===== 文档装订器 =====

class DocumentComposer:
    """
    将章节拼接成 Document IR 的装订器
    
    功能:
    - 按 order 排序章节
    - 防止 anchor 重复，生成全局唯一锚点
    - 注入 IR 版本与生成时间戳
    """
    
    def __init__(self):
        """初始化装订器"""
        self._seen_anchors: Set[str] = set()
    
    def build_document(
        self,
        report_id: str,
        metadata: Dict[str, Any],
        chapters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        把所有章节按 order 排序并注入唯一锚点，形成整本 IR
        
        Args:
            report_id: 报告 ID
            metadata: 全局元信息（标题、主题等）
            chapters: 章节 payload 列表
            
        Returns:
            满足渲染器需求的 Document IR
        """
        toc_anchor_map = self._build_toc_anchor_map(metadata)
        
        ordered = sorted(chapters, key=lambda c: c.get("order", 0))
        for idx, chapter in enumerate(ordered, start=1):
            chapter.setdefault("chapterId", f"S{idx}")
            
            chapter_id = chapter.get("chapterId")
            anchor = (
                toc_anchor_map.get(chapter_id) or
                chapter.get("anchor") or
                f"section-{idx}"
            )
            chapter["anchor"] = self._ensure_unique_anchor(anchor)
            chapter.setdefault("order", idx * 10)
            
            if chapter.get("errorPlaceholder"):
                self._ensure_heading_block(chapter)
        
        document = {
            "version": IR_VERSION,
            "reportId": report_id,
            "metadata": {
                **metadata,
                "generatedAt": metadata.get("generatedAt")
                or datetime.utcnow().isoformat() + "Z",
            },
            "themeTokens": metadata.get("themeTokens", {}),
            "chapters": ordered,
            "assets": metadata.get("assets", {}),
        }
        return document
    
    def _ensure_unique_anchor(self, anchor: str) -> str:
        """确保锚点全局唯一"""
        base = anchor
        counter = 2
        while anchor in self._seen_anchors:
            anchor = f"{base}-{counter}"
            counter += 1
        self._seen_anchors.add(anchor)
        return anchor
    
    def _build_toc_anchor_map(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """构建 chapterId 到 anchor 的映射"""
        toc_config = metadata.get("toc") or {}
        custom_entries = toc_config.get("customEntries") or []
        anchor_map = {}
        
        for entry in custom_entries:
            if isinstance(entry, dict):
                chapter_id = entry.get("chapterId")
                anchor = entry.get("anchor")
                if chapter_id and anchor:
                    anchor_map[chapter_id] = anchor
        
        return anchor_map
    
    def _ensure_heading_block(self, chapter: Dict[str, Any]) -> None:
        """保证占位章节拥有可用于目录的 heading block"""
        blocks = chapter.get("blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "heading":
                    return
        
        heading = {
            "type": "heading",
            "level": 2,
            "text": chapter.get("title") or "占位章节",
            "anchor": chapter.get("anchor"),
        }
        
        if isinstance(blocks, list):
            blocks.insert(0, heading)
        else:
            chapter["blocks"] = [heading]


# ===== 模板解析器 =====

SECTION_ORDER_STEP = 10

# 正则表达式
heading_pattern = re.compile(
    r"""
    (?P<marker>\#{1,6})       # Markdown标题标记
    [ \t]+                    # 必需的空白字符
    (?P<title>[^\r\n]+)       # 不包含换行的标题文本
    """,
    re.VERBOSE,
)

bullet_pattern = re.compile(
    r"""
    (?P<marker>[-*+])         # 列表项目符号
    [ \t]+
    (?P<title>[^\r\n]+)
    """,
    re.VERBOSE,
)

number_pattern = re.compile(
    r"""
    (?P<num>
        (?:0|[1-9]\d*)
        (?:\.(?:0|[1-9]\d*))*
    )
    (?:
        (?:[ \t\u00A0\u3000、:：-]+|\.(?!\d))+
        (?P<label>[^\r\n]*)
    )?
    """,
    re.VERBOSE,
)


def slugify(text: str) -> str:
    """将文本转换为 URL 友好的 slug"""
    # 移除非字母数字字符
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text or 'section'


def parse_template_sections(template_md: str) -> List[TemplateSection]:
    """
    将 Markdown 模板切分成章节列表
    
    Args:
        template_md: 模板 Markdown 全文
        
    Returns:
        结构化的章节序列
    """
    sections: List[TemplateSection] = []
    current: Optional[TemplateSection] = None
    order = SECTION_ORDER_STEP
    used_slugs: Set[str] = set()
    
    for raw_line in template_md.splitlines():
        if not raw_line.strip():
            continue
        
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        
        meta = _classify_line(stripped, indent)
        if not meta:
            continue
        
        if meta["is_section"]:
            slug = _ensure_unique_slug(meta["slug"], used_slugs)
            section = TemplateSection(
                title=meta["title"],
                slug=slug,
                order=order,
                depth=meta["depth"],
                raw_title=meta["raw"],
                number=meta["number"],
            )
            sections.append(section)
            current = section
            order += SECTION_ORDER_STEP
            continue
        
        # 提纲条目
        if current:
            current.outline.append(meta["title"])
    
    for idx, section in enumerate(sections, start=1):
        section.chapter_id = f"S{idx}"
    
    return sections


def _classify_line(stripped: str, indent: int) -> Optional[dict]:
    """根据缩进与符号分类行"""
    # Markdown 标题
    m = heading_pattern.match(stripped)
    if m:
        depth = len(m.group("marker"))
        title = m.group("title").strip()
        return {
            "is_section": depth <= 2,
            "depth": depth,
            "title": title,
            "slug": slugify(title),
            "number": "",
            "raw": stripped,
        }
    
    # 列表项
    m = bullet_pattern.match(stripped)
    if m:
        title = m.group("title").strip()
        # 检查是否有编号
        num_match = number_pattern.match(title)
        if num_match:
            number = num_match.group("num")
            label = (num_match.group("label") or "").strip()
            is_section = indent == 0 and "." not in number
            return {
                "is_section": is_section,
                "depth": 1 if is_section else 2,
                "title": label or title,
                "slug": slugify(label or title),
                "number": number,
                "raw": stripped,
            }
        return {
            "is_section": indent == 0,
            "depth": 1 if indent == 0 else 2,
            "title": title,
            "slug": slugify(title),
            "number": "",
            "raw": stripped,
        }
    
    return None


def _ensure_unique_slug(slug: str, used: Set[str]) -> str:
    """确保 slug 唯一"""
    base = slug
    counter = 2
    while slug in used:
        slug = f"{base}-{counter}"
        counter += 1
    used.add(slug)
    return slug


# ===== 章节存储管理 =====

class ChapterStorage:
    """
    章节 JSON 写入与 manifest 管理器
    
    功能:
    - 为每次报告创建独立 run 目录与 manifest 快照
    - 在章节流式生成时即时写入
    - 校验通过后持久化并更新 manifest 状态
    """
    
    def __init__(self, base_dir: str):
        """
        创建章节存储器
        
        Args:
            base_dir: 输出目录根路径
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, report_id: str, metadata: Dict[str, Any]) -> Path:
        """
        为本次报告创建独立的章节输出目录与 manifest
        
        Args:
            report_id: 任务 ID
            metadata: 报告元数据
            
        Returns:
            新建的 run 目录
        """
        run_dir = self.base_dir / report_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "reportId": report_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata,
            "chapters": [],
        }
        
        self._manifests[self._key(run_dir)] = manifest
        self._write_manifest(run_dir, manifest)
        return run_dir
    
    def begin_chapter(self, run_dir: Path, chapter_meta: Dict[str, Any]) -> Path:
        """
        创建章节子目录并在 manifest 中标记为 streaming 状态
        
        Args:
            run_dir: 会话根目录
            chapter_meta: 包含 chapterId/title/slug/order 的元数据
            
        Returns:
            章节目录
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        
        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status="streaming",
            files={"raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir))},
        )
        
        self._upsert_record(run_dir, record)
        return chapter_dir
    
    def persist_chapter(
        self,
        run_dir: Path,
        chapter_meta: Dict[str, Any],
        payload: Dict[str, Any],
        errors: Optional[List[str]] = None,
    ) -> Path:
        """
        章节流式生成完毕后写入最终 JSON 并更新 manifest 状态
        
        Args:
            run_dir: 会话根目录
            chapter_meta: 章节元信息
            payload: 校验通过的章节 JSON
            errors: 可选的错误列表
            
        Returns:
            最终的 chapter.json 文件路径
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        
        chapter_file = chapter_dir / "chapter.json"
        chapter_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        status = "invalid" if errors else "complete"
        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status=status,
            files={
                "raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir)),
                "chapter": str(chapter_file.relative_to(run_dir)),
            },
            errors=errors or [],
        )
        
        self._upsert_record(run_dir, record)
        return chapter_file
    
    def get_manifest(self, run_dir: Path) -> Dict[str, Any]:
        """获取 manifest"""
        return self._manifests.get(self._key(run_dir), {})
    
    def _key(self, run_dir: Path) -> str:
        """生成 manifest 缓存键"""
        return str(run_dir.resolve())
    
    def _write_manifest(self, run_dir: Path, manifest: Dict[str, Any]) -> None:
        """写入 manifest.json"""
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    
    def _chapter_dir(self, run_dir: Path, slug: str, order: int) -> Path:
        """获取或创建章节目录"""
        dir_name = f"{order:03d}-{slug}"
        chapter_dir = run_dir / dir_name
        chapter_dir.mkdir(parents=True, exist_ok=True)
        return chapter_dir
    
    def _raw_stream_path(self, chapter_dir: Path) -> Path:
        """获取原始流文件路径"""
        return chapter_dir / "stream.raw"
    
    def _upsert_record(self, run_dir: Path, record: ChapterRecord) -> None:
        """更新或插入章节记录"""
        manifest = self._manifests.get(self._key(run_dir))
        if not manifest:
            return
        
        chapters = manifest.get("chapters", [])
        # 查找现有记录
        for i, ch in enumerate(chapters):
            if ch.get("chapterId") == record.chapter_id:
                chapters[i] = record.to_dict()
                break
        else:
            chapters.append(record.to_dict())
        
        manifest["chapters"] = chapters
        self._write_manifest(run_dir, manifest)


__all__ = [
    # 常量
    "IR_VERSION",
    "SECTION_ORDER_STEP",
    # 数据类
    "TemplateSection",
    "ChapterRecord",
    # 类
    "DocumentComposer",
    "ChapterStorage",
    # 函数
    "parse_template_sections",
    "slugify",
]
