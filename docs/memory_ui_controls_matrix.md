# 各产品 Memory / 个性化 / 数据控制项对照表（UI 侧）

更新时间：2025-12-23（以各产品公开文档为准；不同地区/版本/套餐可能不同）

说明：
- 这里的 “Memory” 以**用户可控的跨会话个性化**为主；IDE 类产品往往更偏 “上下文/代码库索引”，不一定提供“用户记忆条目列表”。
- 如果某产品的官方页面需要交互验证（如 Cloudflare/登录），表内会标注为“文档不可直连/需登录”。

| 产品 | Memory/个性化形态 | 开关（作用域） | 查看/编辑 | 删除（粒度） | 训练使用（默认与控制） | 公开来源 |
|---|---|---|---|---|---|---|
| ChatGPT（OpenAI） | `Saved memories`（高层偏好/事实）+ `Reference chat history`（从历史对话提取/引用） | 两个开关：`Reference saved memories`、`Reference chat history`；支持 `Temporary Chat`（不引用/不写入记忆）；Enterprise 管理员可在 Admin Settings 统一开关 | 可询问 “What do you remember about me?”；可在 `Settings > Personalization > Manage memories` 管理 | 支持删单条/清空；并明确“关 memory 不会自动删除已保存的 memory”；“删聊天不等于删 memory”；“彻底删除需同时删 memory + 相关聊天” | 若打开 `Improve the model for everyone`，可能使用 past chats/saved memories 等用于改进模型；可在 Data Controls 关闭；Business/Enterprise/Edu 默认不训练 | https://help.openai.com/en/articles/8590148-memory-faq |
| Claude.ai（Anthropic） | **Projects**：把“文档/代码/洞察 + 聊天活动”放入项目（200K context），并可设定每个 Project 的自定义指令；更像“项目级长期上下文”而非全局用户记忆 | Pro/Team 可用 Projects（UI 维度是 Project）；是否有“全局记忆开关”未在公开文档中明确 | Projects 支持“上传/添加项目知识、项目级自定义指令、分享 project 内对话快照”等（具体 UI 操作在 Claude.ai 内） | 删除/导出等 UI 细节未在该公告中细化；Consumer Terms 提到服务终止时可能删除 Materials（账户数据） | Projects 公告：项目内共享数据/聊天**不会在未获得用户明确同意时用于训练**；Consumer Terms：可在 account settings `opt out of training`，但反馈/安全审查等例外可能仍用于训练 | https://www.anthropic.com/news/projects ; https://www.anthropic.com/legal/consumer-terms |
| Gemini（Google） | 主要围绕 Activity/聊天历史与个性化：`Keep Activity`（是否保存活动到账号）；可“基于 past chats 个性化”；隐私中心提到 `saved info`/instructions 等 | `Keep Activity` 可开关；个性化可在设置中开关（且依赖 Keep Activity）；部分功能地区限制（如 EEA/UK/CH） | 可在 `Gemini Apps Activity / My Activity` 查看；在 Gemini UI 中也可管理最近聊天（pin/rename/delete） | 支持删除聊天/活动；可设置 auto-delete；Keep Activity 关闭后仍可能短期保留部分数据用于业务/安全等（详见隐私中心） | 隐私中心：数据可能用于改进服务/训练模型，并存在“是否用于改进 Google AI”的控制项；部分数据可能会被 human reviewers 审阅 | https://support.google.com/gemini/answer/13594961 ; https://support.google.com/gemini/answer/13278892 ; https://support.google.com/gemini/answer/15637730 |
| Grok（xAI，Grok.com/iOS/Android） | 更偏“对话历史/数据控制”而非“可编辑记忆条目”；提供 `Private Chat`（不保存可见历史/不用于训练） | 提供 `Private Chat`（ghost icon）；并提到 Grok apps 的 Settings 有 data controls（用于选择功能/限制） | 未公开“记忆条目列表”；提供数据控制项（Settings） | `Private Chat`：历史不对用户可见，并在 30 天内从 xAI 系统删除 | Consumer FAQs：可 `opt out of model training`；或用 `Private Chat` 使内容不用于训练；但用户主动反馈等可能用于训练 | https://x.ai/legal/faq |
| GitHub Copilot Chat（IDE） | 主要是“上下文增强”，不是用户长期记忆：仓库名/打开文件等上下文 + 可选指令文件 | 支持可选 `.github/copilot-instructions.md` 自动附加指令；用户可在 Copilot extension settings 禁用该特性 | 指令文件可版本化审计；聊天响应可能列出引用文件 | 非“记忆条目”范式；删除通常依赖 IDE/服务侧的聊天记录管理 | 文档未把它描述为“用户记忆训练开关”；重点在上下文收集与安全/责任使用 | https://docs.github.com/api/article/body?pathname=/en/copilot/responsible-use/chat-in-your-ide |
| Augment Code（augmentcode.com） | IDE 编程助手：强调“context engine”（对代码库的上下文理解），未公开描述为“用户可编辑的长期记忆条目” | 公开页面未给出“Memory UI 开关/查看/删除”细节（大概率在 IDE 插件或 app 内） | 同上 | 同上 | Security & Privacy 页面明确：**不使用客户专有代码训练模型** | https://www.augmentcode.com/security |
| Perplexity（perplexity.ai） | 公开站点的 Help/Legal 页面在本环境中需要交互验证（Cloudflare），因此本表不对其 UI 控制项做断言 | 文档不可直连/需交互验证 | 文档不可直连/需交互验证 | 文档不可直连/需交互验证 | 文档不可直连/需交互验证 | https://www.perplexity.ai/sitemap.xml（可见其 legal/help 路径，但正文需交互验证） |

