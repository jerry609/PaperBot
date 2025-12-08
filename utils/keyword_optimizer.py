"""
关键词优化器

参考: BettaFish/MediaEngine/tools/
适配: PaperBot 学者追踪 - 提升 Semantic Scholar 查询命中率

功能:
- 查询扩展 (同义词、相关术语)
- 查询重写 (LLM 辅助)
- 安全领域术语优化
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from loguru import logger


# ===== 安全领域术语库 =====

SECURITY_SYNONYMS: Dict[str, List[str]] = {
    # 攻击类型
    "vulnerability": ["security flaw", "weakness", "exploit", "bug", "CVE"],
    "malware": ["virus", "trojan", "ransomware", "worm", "spyware", "rootkit"],
    "phishing": ["social engineering", "spear phishing", "whaling"],
    "ddos": ["denial of service", "distributed denial of service", "DoS attack"],
    "injection": ["SQL injection", "code injection", "command injection", "XSS"],
    "overflow": ["buffer overflow", "stack overflow", "heap overflow", "memory corruption"],
    
    # 防御技术
    "detection": ["intrusion detection", "anomaly detection", "threat detection"],
    "encryption": ["cryptography", "cipher", "AES", "RSA", "TLS", "SSL"],
    "authentication": ["identity verification", "MFA", "2FA", "biometrics"],
    "firewall": ["network security", "packet filtering", "WAF"],
    "antivirus": ["anti-malware", "endpoint protection", "EDR"],
    
    # 研究领域
    "fuzzing": ["fuzz testing", "mutation testing", "coverage-guided fuzzing", "AFL"],
    "binary analysis": ["reverse engineering", "disassembly", "decompilation"],
    "program analysis": ["static analysis", "dynamic analysis", "taint analysis"],
    "formal verification": ["model checking", "theorem proving", "symbolic execution"],
    "machine learning security": ["adversarial ML", "ML robustness", "AI security"],
    
    # 会议/标准
    "top venue": ["S&P", "CCS", "USENIX Security", "NDSS"],
    "IEEE S&P": ["Oakland", "IEEE Symposium on Security and Privacy"],
    "CCS": ["ACM CCS", "ACM Conference on Computer and Communications Security"],
    "USENIX Security": ["USENIX Security Symposium"],
    "NDSS": ["Network and Distributed System Security"],
}

# 常见缩写展开
ABBREVIATION_EXPANSIONS: Dict[str, str] = {
    "ML": "machine learning",
    "DL": "deep learning",
    "AI": "artificial intelligence",
    "NLP": "natural language processing",
    "IoT": "Internet of Things",
    "ICS": "industrial control systems",
    "SCADA": "supervisory control and data acquisition",
    "APT": "advanced persistent threat",
    "CVE": "Common Vulnerabilities and Exposures",
    "CTF": "capture the flag",
    "PWN": "binary exploitation",
    "RE": "reverse engineering",
    "ROP": "return-oriented programming",
    "ASLR": "address space layout randomization",
    "DEP": "data execution prevention",
    "CFI": "control flow integrity",
    "SGX": "Software Guard Extensions",
    "TEE": "trusted execution environment",
    "TLS": "Transport Layer Security",
    "PKI": "public key infrastructure",
}


# ===== 数据结构 =====

@dataclass
class OptimizedQuery:
    """优化后的查询"""
    original: str
    optimized: str
    expansions: List[str] = field(default_factory=list)
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== 关键词优化器 =====

class KeywordOptimizer:
    """
    关键词优化器
    
    提供查询扩展和优化功能，提升搜索命中率
    """
    
    def __init__(
        self,
        synonyms: Optional[Dict[str, List[str]]] = None,
        abbreviations: Optional[Dict[str, str]] = None,
        llm_client: Any = None,
    ):
        """
        初始化优化器
        
        Args:
            synonyms: 自定义同义词词典
            abbreviations: 自定义缩写词典
            llm_client: LLM 客户端 (用于高级重写)
        """
        self.synonyms = {**SECURITY_SYNONYMS, **(synonyms or {})}
        self.abbreviations = {**ABBREVIATION_EXPANSIONS, **(abbreviations or {})}
        self.llm_client = llm_client
    
    def expand_abbreviations(self, query: str) -> str:
        """
        展开查询中的缩写
        
        Args:
            query: 原始查询
            
        Returns:
            展开后的查询
        """
        words = query.split()
        expanded = []
        
        for word in words:
            upper_word = word.upper()
            if upper_word in self.abbreviations:
                # 保留原词并添加展开形式
                expanded.append(f"{word} ({self.abbreviations[upper_word]})")
            else:
                expanded.append(word)
        
        return " ".join(expanded)
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        获取术语的同义词
        
        Args:
            term: 术语
            
        Returns:
            同义词列表
        """
        term_lower = term.lower()
        
        # 直接匹配
        if term_lower in self.synonyms:
            return self.synonyms[term_lower]
        
        # 部分匹配
        results = []
        for key, values in self.synonyms.items():
            if term_lower in key or key in term_lower:
                results.extend(values)
            for value in values:
                if term_lower in value.lower():
                    results.append(key)
                    break
        
        return list(set(results))
    
    def expand_query(self, query: str, max_expansions: int = 3) -> OptimizedQuery:
        """
        扩展查询
        
        Args:
            query: 原始查询
            max_expansions: 最大扩展数量
            
        Returns:
            优化后的查询
        """
        # 1. 展开缩写
        expanded = self.expand_abbreviations(query)
        
        # 2. 收集同义词
        words = query.lower().split()
        all_synonyms: Set[str] = set()
        
        for word in words:
            syns = self.get_synonyms(word)
            all_synonyms.update(syns[:max_expansions])
        
        # 3. 构建扩展查询
        expansions = list(all_synonyms)[:max_expansions]
        
        if expansions:
            expansion_str = " OR ".join(f'"{e}"' for e in expansions)
            optimized = f"({expanded}) OR ({expansion_str})"
        else:
            optimized = expanded
        
        return OptimizedQuery(
            original=query,
            optimized=optimized,
            expansions=expansions,
        )
    
    def optimize_for_semantic_scholar(self, query: str) -> OptimizedQuery:
        """
        针对 Semantic Scholar API 优化查询
        
        Args:
            query: 原始查询
            
        Returns:
            优化后的查询
        """
        # Semantic Scholar 搜索技巧:
        # - 使用引号进行精确匹配
        # - 简洁明了的关键词效果更好
        # - 避免过长的查询
        
        # 1. 基本清理
        cleaned = query.strip()
        
        # 2. 检测并保留已有引号
        if '"' in cleaned:
            # 用户已经使用了精确匹配，保持原样
            optimized = cleaned
            expansions = []
        else:
            # 3. 识别核心术语
            core_terms = self._extract_core_terms(cleaned)
            
            # 4. 展开缩写但不过度扩展
            expanded_terms = []
            expansions = []
            
            for term in core_terms:
                upper_term = term.upper()
                if upper_term in self.abbreviations:
                    expanded_terms.append(self.abbreviations[upper_term])
                    expansions.append(self.abbreviations[upper_term])
                else:
                    expanded_terms.append(term)
            
            optimized = " ".join(expanded_terms)
        
        return OptimizedQuery(
            original=query,
            optimized=optimized,
            expansions=expansions,
            metadata={"target": "semantic_scholar"},
        )
    
    def _extract_core_terms(self, query: str) -> List[str]:
        """提取核心术语"""
        # 简单的停用词过滤
        stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "is", "are", "was", "were", "be",
            "with", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under",
        }
        
        words = query.lower().split()
        core = [w for w in words if w not in stop_words and len(w) > 2]
        
        return core if core else words
    
    def generate_search_variants(self, query: str, num_variants: int = 3) -> List[OptimizedQuery]:
        """
        生成多个搜索变体
        
        Args:
            query: 原始查询
            num_variants: 变体数量
            
        Returns:
            查询变体列表
        """
        variants = []
        
        # 变体1: 原始查询
        variants.append(OptimizedQuery(
            original=query,
            optimized=query,
            score=1.0,
            metadata={"type": "original"},
        ))
        
        # 变体2: 缩写展开
        expanded = self.expand_abbreviations(query)
        if expanded != query:
            variants.append(OptimizedQuery(
                original=query,
                optimized=expanded,
                score=0.9,
                metadata={"type": "abbreviation_expanded"},
            ))
        
        # 变体3: 同义词扩展
        syn_query = self.expand_query(query)
        if syn_query.optimized != query:
            syn_query.score = 0.8
            syn_query.metadata["type"] = "synonym_expanded"
            variants.append(syn_query)
        
        # 变体4: 精确匹配 (如果查询足够短)
        if len(query.split()) <= 4:
            variants.append(OptimizedQuery(
                original=query,
                optimized=f'"{query}"',
                score=0.7,
                metadata={"type": "exact_match"},
            ))
        
        return variants[:num_variants]
    
    async def rewrite_with_llm(self, query: str, context: Optional[str] = None) -> OptimizedQuery:
        """
        使用 LLM 重写查询
        
        Args:
            query: 原始查询
            context: 额外上下文
            
        Returns:
            重写后的查询
        """
        if not self.llm_client:
            logger.warning("LLM 客户端未配置，返回原始查询")
            return OptimizedQuery(original=query, optimized=query)
        
        prompt = f"""你是一个学术搜索专家。请将以下查询优化为更适合在学术数据库（如 Semantic Scholar）中搜索的形式。

原始查询: {query}
{f"上下文: {context}" if context else ""}

要求:
1. 使用标准学术术语
2. 保持简洁（不超过6个关键词）
3. 去除无关词汇
4. 如有必要，展开缩写

只返回优化后的查询，不要解释。"""

        try:
            response = await self.llm_client.invoke_async(
                messages=[{"role": "user", "content": prompt}]
            )
            optimized = response.strip().strip('"')
            
            return OptimizedQuery(
                original=query,
                optimized=optimized,
                metadata={"type": "llm_rewritten"},
            )
        except Exception as e:
            logger.error(f"LLM 重写失败: {e}")
            return OptimizedQuery(original=query, optimized=query)


# ===== 安全论文查询构建器 =====

class SecurityPaperQueryBuilder:
    """
    安全论文查询构建器
    
    专门用于构建安全领域的学术搜索查询
    """
    
    TOP_VENUES = [
        "IEEE S&P",
        "CCS",
        "USENIX Security",
        "NDSS",
    ]
    
    ATTACK_CATEGORIES = {
        "web": ["XSS", "CSRF", "SQL injection", "SSRF", "web vulnerability"],
        "network": ["DDoS", "man-in-the-middle", "network attack", "traffic analysis"],
        "system": ["buffer overflow", "kernel exploit", "privilege escalation", "rootkit"],
        "mobile": ["Android security", "iOS security", "mobile malware", "app vulnerability"],
        "iot": ["IoT security", "smart device", "embedded security", "firmware"],
        "ml": ["adversarial", "model poisoning", "backdoor attack", "evasion attack"],
    }
    
    DEFENSE_CATEGORIES = {
        "detection": ["intrusion detection", "anomaly detection", "malware detection"],
        "prevention": ["access control", "sandboxing", "isolation", "mitigation"],
        "analysis": ["static analysis", "dynamic analysis", "fuzzing", "symbolic execution"],
        "crypto": ["encryption", "authentication", "secure protocol", "key management"],
    }
    
    def __init__(self, optimizer: Optional[KeywordOptimizer] = None):
        """
        初始化构建器
        
        Args:
            optimizer: 关键词优化器
        """
        self.optimizer = optimizer or KeywordOptimizer()
    
    def build_attack_query(self, attack_type: str, specific_terms: Optional[List[str]] = None) -> str:
        """
        构建攻击类型查询
        
        Args:
            attack_type: 攻击类别 (web, network, system, mobile, iot, ml)
            specific_terms: 特定术语
            
        Returns:
            查询字符串
        """
        base_terms = self.ATTACK_CATEGORIES.get(attack_type.lower(), [attack_type])
        all_terms = base_terms + (specific_terms or [])
        
        return " OR ".join(f'"{term}"' for term in all_terms)
    
    def build_defense_query(self, defense_type: str, specific_terms: Optional[List[str]] = None) -> str:
        """
        构建防御类型查询
        
        Args:
            defense_type: 防御类别 (detection, prevention, analysis, crypto)
            specific_terms: 特定术语
            
        Returns:
            查询字符串
        """
        base_terms = self.DEFENSE_CATEGORIES.get(defense_type.lower(), [defense_type])
        all_terms = base_terms + (specific_terms or [])
        
        return " OR ".join(f'"{term}"' for term in all_terms)
    
    def build_venue_filter(self, venues: Optional[List[str]] = None) -> str:
        """
        构建会议/期刊过滤
        
        Args:
            venues: 会议/期刊列表，默认为四大安全顶会
            
        Returns:
            venue 过滤字符串
        """
        target_venues = venues or self.TOP_VENUES
        return " OR ".join(f'venue:"{v}"' for v in target_venues)
    
    def build_comprehensive_query(
        self,
        topic: str,
        attack_type: Optional[str] = None,
        defense_type: Optional[str] = None,
        top_venues_only: bool = False,
        year_range: Optional[str] = None,
    ) -> str:
        """
        构建综合查询
        
        Args:
            topic: 主题
            attack_type: 攻击类别
            defense_type: 防御类别
            top_venues_only: 是否只搜索顶会
            year_range: 年份范围
            
        Returns:
            综合查询字符串
        """
        parts = [topic]
        
        if attack_type:
            parts.append(f"({self.build_attack_query(attack_type)})")
        
        if defense_type:
            parts.append(f"({self.build_defense_query(defense_type)})")
        
        query = " AND ".join(parts)
        
        if top_venues_only:
            query = f"({query}) AND ({self.build_venue_filter()})"
        
        return query


__all__ = [
    # 数据结构
    "OptimizedQuery",
    # 优化器
    "KeywordOptimizer",
    "SecurityPaperQueryBuilder",
    # 常量
    "SECURITY_SYNONYMS",
    "ABBREVIATION_EXPANSIONS",
]
