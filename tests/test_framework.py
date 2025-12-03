"""
PaperBot 测试框架

参考: BettaFish/ReportEngine/tests/
适配: PaperBot 单元测试与集成测试

包含:
- 测试基类与 fixtures
- Mock 数据生成
- 测试辅助函数
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from datetime import datetime


# ===== 测试数据目录 =====

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"


# ===== Mock 数据生成器 =====

class MockDataGenerator:
    """Mock 数据生成器"""
    
    @staticmethod
    def create_scholar(
        scholar_id: str = "12345",
        name: str = "Test Scholar",
        h_index: int = 50,
        citation_count: int = 5000,
        paper_count: int = 100,
    ) -> Dict[str, Any]:
        """生成 Mock 学者数据"""
        return {
            "authorId": scholar_id,
            "name": name,
            "affiliations": [{"name": "Test University"}],
            "hIndex": h_index,
            "citationCount": citation_count,
            "paperCount": paper_count,
            "url": f"https://www.semanticscholar.org/author/{scholar_id}",
        }
    
    @staticmethod
    def create_paper(
        paper_id: str = "paper123",
        title: str = "Test Paper Title",
        year: int = 2024,
        venue: str = "IEEE S&P",
        citation_count: int = 100,
        authors: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """生成 Mock 论文数据"""
        if authors is None:
            authors = [{"authorId": "12345", "name": "Test Author"}]
        
        return {
            "paperId": paper_id,
            "title": title,
            "abstract": f"This is the abstract of {title}.",
            "year": year,
            "venue": venue,
            "citationCount": citation_count,
            "authors": authors,
            "url": f"https://www.semanticscholar.org/paper/{paper_id}",
            "openAccessPdf": {"url": f"https://example.com/{paper_id}.pdf"},
            "fieldsOfStudy": ["Computer Science", "Security"],
            "publicationDate": f"{year}-01-15",
        }
    
    @staticmethod
    def create_search_response(
        query: str = "test query",
        papers: Optional[List[Dict[str, Any]]] = None,
        total: int = 100,
    ) -> Dict[str, Any]:
        """生成 Mock 搜索响应"""
        if papers is None:
            papers = [MockDataGenerator.create_paper(paper_id=f"paper{i}") for i in range(5)]
        
        return {
            "total": total,
            "offset": 0,
            "data": papers,
        }
    
    @staticmethod
    def create_influence_result(
        scholar_id: str = "12345",
        academic_score: float = 75.5,
        engineering_score: float = 45.2,
    ) -> Dict[str, Any]:
        """生成 Mock 影响力评估结果"""
        return {
            "scholar_id": scholar_id,
            "academic_score": academic_score,
            "engineering_score": engineering_score,
            "total_score": 0.6 * academic_score + 0.4 * engineering_score,
            "tier1_papers": 5,
            "tier2_papers": 10,
            "high_citation_papers": 8,
            "github_stars": 1500,
            "github_forks": 300,
        }


# ===== 测试 Fixtures =====

@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock()
    client.invoke = MagicMock(return_value='{"result": "test"}')
    client.invoke_async = AsyncMock(return_value='{"result": "test"}')
    return client


@pytest.fixture
def mock_semantic_scholar():
    """Mock Semantic Scholar API"""
    with patch("tools.search.requests") as mock_requests:
        mock_response = MagicMock()
        mock_response.json.return_value = MockDataGenerator.create_search_response()
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response
        yield mock_requests


@pytest.fixture
def sample_scholar():
    """示例学者数据"""
    return MockDataGenerator.create_scholar()


@pytest.fixture
def sample_papers():
    """示例论文列表"""
    return [
        MockDataGenerator.create_paper(paper_id=f"paper{i}", title=f"Paper {i}")
        for i in range(10)
    ]


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_cache_db(tmp_path):
    """临时缓存数据库"""
    return str(tmp_path / "test_cache.db")


# ===== 测试辅助函数 =====

def assert_valid_scholar(data: Dict[str, Any]) -> None:
    """断言学者数据有效"""
    assert "authorId" in data or "scholar_id" in data
    assert "name" in data
    assert isinstance(data.get("hIndex") or data.get("h_index", 0), int)


def assert_valid_paper(data: Dict[str, Any]) -> None:
    """断言论文数据有效"""
    assert "paperId" in data or "paper_id" in data
    assert "title" in data
    assert isinstance(data.get("year", 2024), int)


def assert_valid_influence(data: Dict[str, Any]) -> None:
    """断言影响力数据有效"""
    assert "scholar_id" in data or "scholarId" in data
    assert "total_score" in data or "totalScore" in data


def load_test_data(filename: str) -> Dict[str, Any]:
    """加载测试数据文件"""
    filepath = TEST_DATA_DIR / filename
    if filepath.exists():
        return json.loads(filepath.read_text(encoding="utf-8"))
    return {}


def save_test_output(data: Any, filename: str) -> Path:
    """保存测试输出"""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TEST_OUTPUT_DIR / filename
    
    if isinstance(data, (dict, list)):
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        filepath.write_text(str(data), encoding="utf-8")
    
    return filepath


# ===== 测试基类 =====

class BaseTestCase:
    """测试基类"""
    
    @classmethod
    def setup_class(cls):
        """类级别设置"""
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def setup_method(self):
        """方法级别设置"""
        pass
    
    def teardown_method(self):
        """方法级别清理"""
        pass


class AsyncTestCase(BaseTestCase):
    """异步测试基类"""
    
    @pytest.fixture(autouse=True)
    def setup_event_loop(self, event_loop):
        """设置事件循环"""
        self.loop = event_loop


# ===== 示例测试 =====

class TestMockDataGenerator:
    """Mock 数据生成器测试"""
    
    def test_create_scholar(self):
        """测试创建学者数据"""
        scholar = MockDataGenerator.create_scholar(
            scholar_id="test123",
            name="John Doe",
            h_index=30,
        )
        
        assert scholar["authorId"] == "test123"
        assert scholar["name"] == "John Doe"
        assert scholar["hIndex"] == 30
        assert_valid_scholar(scholar)
    
    def test_create_paper(self):
        """测试创建论文数据"""
        paper = MockDataGenerator.create_paper(
            paper_id="paper456",
            title="Security Analysis",
            year=2023,
        )
        
        assert paper["paperId"] == "paper456"
        assert paper["title"] == "Security Analysis"
        assert paper["year"] == 2023
        assert_valid_paper(paper)
    
    def test_create_search_response(self):
        """测试创建搜索响应"""
        response = MockDataGenerator.create_search_response(
            query="fuzzing",
            total=50,
        )
        
        assert response["total"] == 50
        assert "data" in response
        assert len(response["data"]) > 0


class TestSearchModule:
    """搜索模块测试"""
    
    def test_semantic_scholar_search_import(self):
        """测试搜索模块导入"""
        from tools.search import SemanticScholarSearch, SearchResultRanker
        
        assert SemanticScholarSearch is not None
        assert SearchResultRanker is not None
    
    def test_ranker_by_citation(self, sample_papers):
        """测试按引用排序"""
        from tools.search import SearchResultRanker, PaperResult
        
        papers = [
            PaperResult(paper_id=f"p{i}", title=f"Paper {i}", citation_count=i * 10)
            for i in range(5)
        ]
        
        ranked = SearchResultRanker.rank_papers_by_citation(papers)
        
        assert ranked[0].citation_count >= ranked[-1].citation_count


class TestKeywordOptimizer:
    """关键词优化器测试"""
    
    def test_optimizer_import(self):
        """测试优化器导入"""
        from tools.keyword_optimizer import KeywordOptimizer, SecurityPaperQueryBuilder
        
        optimizer = KeywordOptimizer()
        assert optimizer is not None
    
    def test_expand_abbreviations(self):
        """测试缩写展开"""
        from tools.keyword_optimizer import KeywordOptimizer
        
        optimizer = KeywordOptimizer()
        result = optimizer.expand_abbreviations("ML security in IoT")
        
        assert "machine learning" in result.lower() or "ML" in result
    
    def test_get_synonyms(self):
        """测试同义词获取"""
        from tools.keyword_optimizer import KeywordOptimizer
        
        optimizer = KeywordOptimizer()
        synonyms = optimizer.get_synonyms("vulnerability")
        
        assert len(synonyms) > 0


class TestCacheModule:
    """缓存模块测试"""
    
    def test_local_cache_import(self):
        """测试缓存模块导入"""
        from utils.db import LocalCache, JSONFileStorage
        
        assert LocalCache is not None
        assert JSONFileStorage is not None
    
    def test_local_cache_operations(self, temp_cache_db):
        """测试缓存操作"""
        from utils.db import LocalCache
        
        cache = LocalCache(db_path=temp_cache_db)
        
        # 测试学者缓存
        cache.set_scholar("scholar1", "Test Scholar", {"h_index": 50})
        result = cache.get_scholar("scholar1")
        
        assert result is not None
        assert result["h_index"] == 50
    
    def test_kv_cache(self, temp_cache_db):
        """测试键值缓存"""
        from utils.db import LocalCache
        
        cache = LocalCache(db_path=temp_cache_db)
        
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        cache.delete("test_key")
        assert cache.get("test_key") is None


class TestReportCore:
    """报告核心模块测试"""
    
    def test_document_composer_import(self):
        """测试文档装订器导入"""
        from reports.core import DocumentComposer, ChapterStorage
        
        assert DocumentComposer is not None
        assert ChapterStorage is not None
    
    def test_build_document(self):
        """测试构建文档"""
        from reports.core import DocumentComposer
        
        composer = DocumentComposer()
        
        chapters = [
            {"chapterId": "S1", "title": "Chapter 1", "order": 10, "blocks": []},
            {"chapterId": "S2", "title": "Chapter 2", "order": 20, "blocks": []},
        ]
        
        document = composer.build_document(
            report_id="test-report",
            metadata={"title": "Test Report"},
            chapters=chapters,
        )
        
        assert document["reportId"] == "test-report"
        assert len(document["chapters"]) == 2
        assert "version" in document
    
    def test_template_parser(self):
        """测试模板解析"""
        from reports.core import parse_template_sections
        
        template = """
# 第一章
## 1.1 小节
- 要点1
- 要点2

# 第二章
## 2.1 小节
"""
        
        sections = parse_template_sections(template)
        
        assert len(sections) >= 2


# ===== 运行配置 =====

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
