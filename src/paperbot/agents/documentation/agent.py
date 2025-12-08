# src/paperbot/agents/documentation/agent.py
"""
负责生成文档的代理
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import jinja2
import markdown
from ..base import BaseAgent


class DocumentationAgent(BaseAgent):
    """负责生成文档的代理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.template_env = self._setup_template_env()
        self.output_formats = {
            'markdown': self._generate_markdown,
            'html': self._generate_html,
            'pdf': self._generate_pdf
        }

    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行文档生成"""
        analysis_results = kwargs.get("analysis_results", {})
        quality_report = kwargs.get("quality_report", {})
        return await self.process(analysis_results, quality_report)

    async def process(self,
                      analysis_results: Dict[str, Any],
                      quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档生成任务"""
        try:
            # 获取输出格式
            output_format = self.config.get('output_format', 'markdown')
            if output_format not in self.output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            # 生成各类文档
            documentation = {
                'api_docs': await self._generate_api_documentation(analysis_results),
                'security_guide': await self._generate_security_guide(analysis_results),
                'best_practices': await self._generate_best_practices(quality_report),
                'implementation_guide': await self._generate_implementation_guide(
                    analysis_results
                )
            }

            # 转换为指定格式
            generator = self.output_formats[output_format]
            output = await generator(documentation)

            return {
                'format': output_format,
                'content': output,
                'sections': list(documentation.keys())
            }

        except Exception as e:
            self.log_error(e)
            raise

    async def _generate_api_documentation(self,
                                          analysis_results: Dict[str, Any]) -> str:
        """生成API文档"""
        try:
            api_docs = {
                'overview': self._generate_api_overview(analysis_results),
                'endpoints': self._generate_endpoint_docs(analysis_results),
                'models': self._generate_model_docs(analysis_results),
                'examples': self._generate_usage_examples(analysis_results)
            }

            template = self.template_env.get_template('api_docs.md.j2')
            return template.render(api_docs=api_docs)

        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_security_guide(self,
                                       analysis_results: Dict[str, Any]) -> str:
        """生成安全指南"""
        try:
            security_info = {
                'vulnerabilities': self._extract_vulnerabilities(analysis_results),
                'security_measures': self._extract_security_measures(analysis_results),
                'best_practices': self._extract_security_practices(analysis_results)
            }

            template = self.template_env.get_template('security_guide.md.j2')
            return template.render(security_info=security_info)

        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_best_practices(self,
                                       quality_report: Dict[str, Any]) -> str:
        """生成最佳实践文档"""
        try:
            practices = {
                'code_style': self._extract_code_style_practices(quality_report),
                'architecture': self._extract_architecture_practices(quality_report),
                'testing': self._extract_testing_practices(quality_report),
                'performance': self._extract_performance_practices(quality_report)
            }

            template = self.template_env.get_template('best_practices.md.j2')
            return template.render(practices=practices)

        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_implementation_guide(self,
                                             analysis_results: Dict[str, Any]) -> str:
        """生成实现指南"""
        try:
            guide = {
                'setup': self._generate_setup_guide(analysis_results),
                'architecture': self._generate_architecture_guide(analysis_results),
                'workflows': self._generate_workflow_guide(analysis_results),
                'deployment': self._generate_deployment_guide(analysis_results)
            }

            template = self.template_env.get_template('implementation_guide.md.j2')
            return template.render(guide=guide)

        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_markdown(self, documentation: Dict[str, Any]) -> str:
        """生成Markdown格式文档"""
        try:
            template = self.template_env.get_template('main.md.j2')
            return template.render(documentation=documentation)
        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_html(self, documentation: Dict[str, Any]) -> str:
        """生成HTML格式文档"""
        try:
            md_content = await self._generate_markdown(documentation)
            html_content = markdown.markdown(
                md_content,
                extensions=['extra', 'codehilite', 'toc']
            )
            template = self.template_env.get_template('main.html.j2')
            return template.render(content=html_content)
        except Exception as e:
            self.log_error(e)
            return ""

    async def _generate_pdf(self, documentation: Dict[str, Any]) -> bytes:
        """生成PDF格式文档"""
        try:
            html_content = await self._generate_html(documentation)
            # 使用weasyprint或其他工具将HTML转换为PDF
            # 这里需要实现具体的PDF生成逻辑
            return b""
        except Exception as e:
            self.log_error(e)
            return b""

    def _setup_template_env(self) -> jinja2.Environment:
        """设置Jinja2模板环境"""
        try:
            template_path = Path(__file__).parent.parent.parent / 'templates'
            return jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(template_path)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
        except Exception as e:
            self.log_error(e)
            return jinja2.Environment()

    def validate_config(self) -> bool:
        """验证配置"""
        required_keys = ['output_format', 'output_path', 'template_path']
        return all(key in self.config for key in required_keys)

    # ==================== Helper methods (stubs) ====================
    def _generate_api_overview(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_endpoint_docs(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_model_docs(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_usage_examples(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _extract_vulnerabilities(self, analysis_results: Dict[str, Any]) -> List[str]:
        return []

    def _extract_security_measures(self, analysis_results: Dict[str, Any]) -> List[str]:
        return []

    def _extract_security_practices(self, analysis_results: Dict[str, Any]) -> List[str]:
        return []

    def _extract_code_style_practices(self, quality_report: Dict[str, Any]) -> List[str]:
        return []

    def _extract_architecture_practices(self, quality_report: Dict[str, Any]) -> List[str]:
        return []

    def _extract_testing_practices(self, quality_report: Dict[str, Any]) -> List[str]:
        return []

    def _extract_performance_practices(self, quality_report: Dict[str, Any]) -> List[str]:
        return []

    def _generate_setup_guide(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_architecture_guide(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_workflow_guide(self, analysis_results: Dict[str, Any]) -> str:
        return ""

    def _generate_deployment_guide(self, analysis_results: Dict[str, Any]) -> str:
        return ""

