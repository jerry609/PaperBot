"""
文本处理工具函数

来源: BettaFish/InsightEngine/utils/text_processing.py
适配: PaperBot 学者追踪系统

用于清理LLM输出、解析JSON、格式化搜索结果等
"""

import re
import json
from typing import Dict, Any, List, Optional
from json.decoder import JSONDecodeError
from loguru import logger


def clean_json_tags(text: str) -> str:
    """
    清理文本中的JSON标签
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除```json 和 ```标签
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def clean_markdown_tags(text: str) -> str:
    """
    清理文本中的Markdown标签
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除```markdown 和 ```标签
    text = re.sub(r'```markdown\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def remove_reasoning_from_output(text: str) -> str:
    """
    移除输出中的推理过程文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 查找JSON开始位置
    json_start = -1
    
    # 尝试找到第一个 { 或 [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # 从JSON开始位置截取
        return text[json_start:].strip()
    
    # 如果没有找到JSON标记，尝试其他方法
    # 移除常见的推理标识
    patterns = [
        r'(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)',  # 移除推理部分
        r'(?:explanation|解释|说明)[:：]\s*.*?(?=\{|\[)',   # 移除解释部分
        r'^.*?(?=\{|\[)',  # 移除JSON前的所有文本
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    提取并清理响应中的JSON内容
    
    Args:
        text: 原始响应文本
        
    Returns:
        解析后的JSON字典
    """
    # 清理文本
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # 尝试直接解析
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # 尝试修复不完整的JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON对象
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON数组
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 如果所有方法都失败，返回错误信息
    logger.warning(f"无法解析JSON响应: {cleaned_text[:200]}...")
    return {"error": "JSON解析失败", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    修复不完整的JSON响应
    
    Args:
        text: 原始文本
        
    Returns:
        修复后的JSON文本，如果无法修复则返回空字符串
    """
    # 移除多余的逗号和空白
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # 检查是否已经是有效的JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # 检查是否缺少开头的数组符号
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # 如果以对象开始，尝试包装成数组
        if text.count('{') > 1:
            # 多个对象，包装成数组
            text = '[' + text + ']'
        else:
            # 单个对象，包装成数组
            text = '[' + text + ']'
    
    # 检查是否缺少结尾的数组符号
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # 如果以对象结束，尝试包装成数组
        if text.count('}') > 1:
            # 多个对象，包装成数组
            text = '[' + text + ']'
        else:
            # 单个对象，包装成数组
            text = '[' + text + ']'
    
    # 检查括号是否匹配
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # 修复不匹配的括号
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # 验证修复后的JSON是否有效
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # 如果仍然无效，尝试更激进的修复
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    更激进的JSON修复方法
    
    Args:
        text: 原始文本
        
    Returns:
        修复后的JSON文本
    """
    # 查找所有可能的JSON对象
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # 如果有多个对象，包装成数组
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # 如果只有一个对象，包装成数组
        return '[' + objects[0] + ']'
    else:
        # 如果没有找到对象，返回空数组
        return '[]'


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    验证JSON数据是否包含必需字段
    
    Args:
        data: 要验证的数据
        required_fields: 必需字段列表
        
    Returns:
        验证是否通过
    """
    return all(field in data for field in required_fields)


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    截断内容到指定长度
    
    Args:
        content: 原始内容
        max_length: 最大长度
        
    Returns:
        截断后的内容
    """
    if len(content) <= max_length:
        return content
    
    # 尝试在单词边界截断
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # 如果最后一个空格位置合理
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(
    search_results: List[Dict[str, Any]], 
    max_length: int = 20000
) -> List[str]:
    """
    格式化搜索结果用于提示词
    
    Args:
        search_results: 搜索结果列表
        max_length: 每个结果的最大长度
        
    Returns:
        格式化后的内容列表
    """
    formatted_results = []
    
    for result in search_results:
        content = result.get('content', '') or result.get('abstract', '') or result.get('title', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)
    
    return formatted_results


# ============ 学者追踪专用函数 ============

def format_paper_for_prompt(paper: Dict[str, Any], include_abstract: bool = True) -> str:
    """
    格式化论文信息用于提示词
    
    Args:
        paper: 论文信息字典
        include_abstract: 是否包含摘要
        
    Returns:
        格式化后的论文信息
    """
    parts = []
    
    # 标题
    if paper.get('title'):
        parts.append(f"标题: {paper['title']}")
    
    # 作者
    authors = paper.get('authors', [])
    if authors:
        if isinstance(authors, list):
            author_str = ', '.join(authors[:5])  # 最多显示5个作者
            if len(authors) > 5:
                author_str += f" 等共{len(authors)}人"
        else:
            author_str = str(authors)
        parts.append(f"作者: {author_str}")
    
    # 会议/期刊
    venue = paper.get('venue') or paper.get('journal')
    if venue:
        parts.append(f"发表于: {venue}")
    
    # 年份
    year = paper.get('year') or paper.get('publication_year')
    if year:
        parts.append(f"年份: {year}")
    
    # 引用数
    citations = paper.get('citation_count') or paper.get('citations')
    if citations:
        parts.append(f"引用数: {citations}")
    
    # 摘要
    if include_abstract:
        abstract = paper.get('abstract')
        if abstract:
            truncated = truncate_content(abstract, 500)
            parts.append(f"摘要: {truncated}")
    
    return '\n'.join(parts)


def format_scholar_for_prompt(scholar: Dict[str, Any]) -> str:
    """
    格式化学者信息用于提示词
    
    Args:
        scholar: 学者信息字典
        
    Returns:
        格式化后的学者信息
    """
    parts = []
    
    # 姓名
    if scholar.get('name'):
        parts.append(f"姓名: {scholar['name']}")
    
    # 机构
    affiliations = scholar.get('affiliations', [])
    if affiliations:
        if isinstance(affiliations, list):
            parts.append(f"机构: {', '.join(affiliations[:3])}")
        else:
            parts.append(f"机构: {affiliations}")
    
    # H-index
    if scholar.get('h_index'):
        parts.append(f"H-index: {scholar['h_index']}")
    
    # 引用数
    if scholar.get('citation_count'):
        parts.append(f"总引用数: {scholar['citation_count']}")
    
    # 论文数
    if scholar.get('paper_count'):
        parts.append(f"论文数: {scholar['paper_count']}")
    
    # 研究领域
    fields = scholar.get('research_fields') or scholar.get('fields_of_study', [])
    if fields:
        if isinstance(fields, list):
            parts.append(f"研究领域: {', '.join(fields[:5])}")
        else:
            parts.append(f"研究领域: {fields}")
    
    return '\n'.join(parts)


def deduplicate_papers(papers: List[Dict[str, Any]], key: str = 'title') -> List[Dict[str, Any]]:
    """
    去重论文列表
    
    Args:
        papers: 论文列表
        key: 用于去重的键名
        
    Returns:
        去重后的论文列表
    """
    seen = set()
    unique_papers = []
    
    for paper in papers:
        identifier = paper.get(key, '')
        if identifier:
            # 标准化标题（小写、去除空格）
            normalized = identifier.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_papers.append(paper)
        else:
            # 如果没有标识符，保留论文
            unique_papers.append(paper)
    
    return unique_papers


def extract_github_url(text: str) -> Optional[str]:
    """
    从文本中提取 GitHub URL
    
    Args:
        text: 可能包含 GitHub URL 的文本
        
    Returns:
        GitHub URL 或 None
    """
    if not text:
        return None
    
    # GitHub URL 模式
    patterns = [
        r'https?://github\.com/[\w-]+/[\w.-]+',
        r'github\.com/[\w-]+/[\w.-]+',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group()
            if not url.startswith('http'):
                url = 'https://' + url
            return url
    
    return None


def extract_arxiv_id(text: str) -> Optional[str]:
    """
    从文本中提取 arXiv ID
    
    Args:
        text: 可能包含 arXiv ID 的文本
        
    Returns:
        arXiv ID 或 None
    """
    if not text:
        return None
    
    # arXiv ID 模式
    patterns = [
        r'arxiv[:\s]*(\d{4}\.\d{4,5})',  # 新格式: 2301.12345
        r'arxiv[:\s]*([\w-]+/\d+)',      # 旧格式: cs.CV/0501001
        r'(\d{4}\.\d{4,5})',              # 纯数字格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


__all__ = [
    # JSON处理
    'clean_json_tags',
    'clean_markdown_tags',
    'remove_reasoning_from_output',
    'extract_clean_response',
    'fix_incomplete_json',
    'fix_aggressive_json',
    'validate_json_schema',
    
    # 文本处理
    'truncate_content',
    'format_search_results_for_prompt',
    
    # 学者追踪专用
    'format_paper_for_prompt',
    'format_scholar_for_prompt',
    'deduplicate_papers',
    'extract_github_url',
    'extract_arxiv_id',
]
