项目详细文档（DETAILED）

概览
- 语言：Python 3.8+
- 目的：批量抓取安全/会议论文 PDF（CCS/NDSS/USENIX/SP 等），为自动化论文收集提供工具。

主要模块与职责
- main.py：CLI 和任务协调器入口。解析命令行、选择会议/年份、调度下载。
- utils/downloader.py：通用下载器，包含多个会议的通用与特定逻辑（NDSS/USENIX/IEEE-SP 等）。核心：解析文章列表、解析 PDF URL、重试与内容验证（最小体积、%PDF 开头检查）。
  - _resolve_ieee_pdf_url：将 DOI/ieeexplore 链接解析为可直接下载的 stampPDF/getPDF.jsp 链接（提取 arnumber，并构造 base64 ref）。
  - _download_ieee_pdf_with_httpx：使用 httpx HTTP/2 流程获取 ERIGHTS cookie（初始 ERIGHTS=0000），处理 302 重定向并抓取最终 PDF。
  - _download_with_retry：根据 URL 路径选择合适的下载器（包含对 ieee stampPDF 的分流）。
- utils/downloader_ccs.py：针对 ACM CCS 的定制逻辑。负责抓取会议页面、提取 DOI 列表并通过 ACM API 批量请求论文元数据与 PDF 链接。
  - 已实现对 DOI 的预校验：仅将最后路径段包含“.”的 DOI 视为有效论文 DOI（例如 10.1145/3372297.3423365）。
- utils/CCS-DOWN.py：另一个 CCS 相关解析脚本，处理 JSON/HTML 并在解析阶段过滤无效 DOI。
- utils/*（其它工具）：包含下载器后备实现（curl）、解析器、日志模块等。
- agents/ 与 agent_module/：面向更高层的研究/分析 agent（当前主要是项目架构与扩展点）。

重要实现细节
1. IEEE ERIGHTS 流程
- 一些 IEEE PDF 获取需要特定 Cookie（ERIGHTS）来授权访问。实现步骤：
  1) 构造 stampPDF 请求：/stampPDF/getPDF.jsp?tp=&arnumber={文档号}&ref={base64(ieeexplore 文档 URL)}
  2) 使用 httpx 启用 HTTP/2，先以 Cookie: ERIGHTS=0000 发起请求。
  3) 服务器会返回 302 并在 Set-Cookie 中包含新的 ERIGHTS 值。捕获该值并随后的请求中使用它以获取 PDF 内容。
- 备注：必须安装 httpx[http2] 才能启用 HTTP/2；若环境不支持 HTTP/2，将无法获得 ERIGHTS 更新。

2. ACM CCS DOI 过滤策略
- 问题：页面中包含会话标题、索引等条目，它们在 dl.acm.org 的 DOI 最后段没有“.”，不是有效论文 DOI。
- 解决：在解析阶段和向 ACM 批量 POST 请求前均过滤 DOI。仅保留最后路径段包含“.”的 DOI。

3. 通用防护与回退
- 下载文件验证：检查大小阈值（>1KB）和二进制流以 "%PDF-" 起始。
- 回退机制：在部分场景使用 curl（子进程）作为回退，代码里保留了 _download_with_curl 或 curl_cffi、cloudscraper 可选策略以应对反爬。

运行与依赖
- requirements.txt：包含已有依赖，已加入 httpx[http2]。
- 建议 Python 环境：3.8 或更高。

错误与已知问题
- 403 on ACM pages：通常是访问受限或反爬导致。可通过机构网络、VPN 或更复杂的 anti-bot 工具（cloudscraper / curl_cffi）缓解。
- Markdown lint：README.md 的初版生成曾触发若干 markdown-lint 警告，已简化为快速开始版并将详细文档迁移到 docs/ 以便逐步修复格式问题。

扩展与待办
- 添加自动化测试：针对解析函数（文档 ID 提取、DOI 过滤）和下载器（模拟响应）编写单元测试。
- 抽象会议适配器接口：为不同会议实现插件式适配器，统一输入（会议页面/DOI 列表）与输出（元数据 + pdf_url）。
- 更强的反爬策略：可选集成 cloudscraper、或浏览器自动化（Playwright/selenium）用于难以绕过的站点。

文件映射（简要）
- main.py：入口与 CLI
- utils/downloader.py：通用下载器（包含 IEEE HTTP/2 ERIGHTS 流程）
- utils/downloader_ccs.py：CCS 专用处理
- utils/CCS-DOWN.py：备用/历史 CCS 解析
- utils/*.py：解析、下载回退、日志等工具
- agents/, agent_module/：更高层 agent 逻辑

联系点
- 若需要我将 docs 中的部分内容拆分为更细的文件（例如 API、设计、运维指南、开发者文档），请指明优先级与格式要求。
