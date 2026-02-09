# 新增 Topic Search Source 模板（开发者指南）

本文档用于指导你把新的数据源接入 Topic Workflow（例如替换或补充 `papers_cool`）。

## 1. 接口契约

实现 `TopicSearchSource` 协议（见 `src/paperbot/application/workflows/topic_search_sources.py`）：

```python
class TopicSearchSource(Protocol):
    name: str

    def search(
        self,
        *,
        query: str,
        branches: Sequence[str],
        show_per_branch: int,
    ) -> List[TopicSearchRecord]:
        ...
```

返回值必须是 `TopicSearchRecord` 列表。

## 2. 数据字段映射

`TopicSearchRecord` 最小建议字段映射：

- `source`: 数据源名（例如 `my_source`）
- `source_record_id`: 源内唯一 ID
- `title`: 论文标题
- `url`: 系统内可跳转链接（优先你的聚合页）
- `external_url`: 原始来源链接（论文详情页）
- `source_branch`: 数据分支（如 `arxiv`/`venue`/`preprint`）
- `snippet`: 摘要或简述
- `keywords`: 关键词数组
- `pdf_stars` / `kimi_stars`: 若无可填 `0`

其余字段可空字符串或空数组，但请保持结构稳定。

## 3. 最小实现模板

```python
from __future__ import annotations

from typing import List, Sequence

from paperbot.application.workflows.topic_search_sources import TopicSearchRecord


class MySource:
    name = "my_source"

    def search(
        self,
        *,
        query: str,
        branches: Sequence[str],
        show_per_branch: int,
    ) -> List[TopicSearchRecord]:
        rows: List[TopicSearchRecord] = []

        # 1) 构造请求
        # 2) 拉取并解析
        # 3) 映射成 TopicSearchRecord

        rows.append(
            TopicSearchRecord(
                source=self.name,
                source_record_id="id-123",
                title="A Sample Paper",
                url="https://example.com/paper/123",
                source_branch="venue",
                external_url="https://publisher.com/123",
                pdf_url="https://publisher.com/123.pdf",
                authors=["Alice", "Bob"],
                subject_or_venue="ACL 2026",
                published_at="2026-01-10",
                snippet="short summary",
                keywords=["icl", "compression"],
                pdf_stars=0,
                kimi_stars=0,
            )
        )
        return rows
```

## 4. 注册方式

### 方式 A：默认注册表中加入

在 `build_default_topic_source_registry(...)` 里注册：

```python
registry.register("my_source", MySource)
```

### 方式 B：运行时注入（推荐给实验）

```python
registry = TopicSearchSourceRegistry()
registry.register("my_source", MySource)
workflow = PapersCoolTopicSearchWorkflow(source_registry=registry)
```

## 5. 联调与调用

### CLI

```bash
python -m paperbot.presentation.cli.main topic-search \
  -q "ICL压缩" --source my_source --json
```

### API

`POST /api/research/paperscool/search`

```json
{
  "queries": ["ICL压缩"],
  "sources": ["my_source"],
  "branches": ["venue"]
}
```

## 6. 测试清单（建议）

1. `test_topic_search_sources.py`
   - register/create/unknown source
2. 你的 source parser fixture test
   - 固定 HTML/JSON fixture，避免线上结构变动导致不稳定
3. workflow 聚合测试
   - 验证跨分支/跨 query 去重与打分

## 7. 常见坑

- `name` 与请求里的 `sources` 不一致（大小写/下划线）
- `url` 缺失导致去重效果差
- `source_record_id` 不稳定导致事件追踪困难
- parser 直接依赖线上结构且无 fixture 回归

---

建议：先按最小字段跑通，再逐步补充 `authors/venue/published_at/keywords` 的质量。
