快速使用指南（USAGE）

先决条件
- Python 3.8+
- 安装依赖：pip install -r requirements.txt（注意包含 httpx[http2]）

快速开始
1. 准备 papers 存放目录（默认 `papers/`）
2. 运行下载器示例：
   - 下载 CCS（示例）: python main.py --conf ccs --year 23 --out papers/ccs_23
   - 下载 SP（示例）: python main.py --conf sp --year 24 --out papers/sp_24

常见问题
- 403/被阻止：尝试使用机构 VPN 或在 headers 中模仿浏览器，或启用 curl_cffi/cloudscraper 回退。
- ERIGHTS 获取失败：确保 httpx[http2] 已安装并且网络环境支持 HTTP/2。

高级
- 要仅抓取 DOI 列表并人工检查：在 `utils/downloader_ccs.py` 中调用 `_parse_ccs_papers` 并打印 DOI 列表。
- 若需通过代理或 SOCKS：为 aiohttp/httpx 会话设置 proxy 参数。

反馈
- 如果你想把某个会议的解析器做成插件，请告诉我目标会议和页面样例，我会实现一个适配器模板。
