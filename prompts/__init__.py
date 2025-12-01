# Prompts Module
# Prompt 模板集中管理

from .research_prompts import RESEARCH_PROMPTS
from .code_analysis_prompts import CODE_ANALYSIS_PROMPTS
from .quality_prompts import QUALITY_PROMPTS
from .report_prompts import REPORT_PROMPTS

__all__ = [
    "RESEARCH_PROMPTS",
    "CODE_ANALYSIS_PROMPTS",
    "QUALITY_PROMPTS",
    "REPORT_PROMPTS",
]
