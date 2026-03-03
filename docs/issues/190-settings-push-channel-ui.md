# Issue #190: Settings — Push Channel Configuration UI

## Priority: P1

## Summary

Settings 页面需要新增"推送通道"配置面板，让用户通过 UI 管理 Telegram/Discord/Lark/WeCom/DingTalk/Slack/Email 等推送渠道，而非手动编辑 `config/push_channels.yaml`。

## Background

Epic #179 已完成后端推送基础设施（Apprise 集成 + 4 个 formatter），但当前只能通过编辑 YAML 配置。用户需要一个可视化配置界面。

## Requirements

### 1. Push Channels Panel (`components/settings/PushChannelsPanel.tsx`)

- 展示已配置的推送通道列表（从后端 API 读取 `push_channels.yaml` 内容）
- 每个通道卡片显示：
  - 渠道类型图标（Telegram/Discord/Lark/WeCom/Slack/Email/DingTalk）
  - 连接状态（URL 是否有效）
  - 标签（daily / alert 等）
  - 编辑 / 删除按钮
- "Add Channel" 按钮 + Quick Presets（类似 Model Providers 的预设）
  - Telegram Bot → `tgram://bot_token/chat_id`
  - Discord Webhook → `discord://webhook_id/webhook_token`
  - Lark/Feishu Webhook → `lark://app_id/app_secret`
  - WeCom Webhook → `wecom://key`
  - Slack Webhook → `slack://token_a/token_b/token_c/#channel`
  - Email SMTP → `mailto://user:pass@smtp/to`
  - DingTalk Webhook → `dingtalk://token`
- "Test Push" 按钮：向选定通道发送测试消息
- 标签管理：选择该通道接收哪些类型的推送（daily / alert / all）

### 2. Backend API

- `GET /api/push-channels` — 读取当前通道配置
- `POST /api/push-channels` — 添加通道
- `PATCH /api/push-channels/{id}` — 更新通道
- `DELETE /api/push-channels/{id}` — 删除通道
- `POST /api/push-channels/{id}/test` — 发送测试推送

### 3. RSS Feed 地址展示

- 在 Push Channels Panel 下方显示 RSS/Atom 订阅地址：
  - `GET /api/feed/daily.xml` — RSS 2.0
  - `GET /api/feed/daily.atom` — Atom
  - 一键复制按钮

### 4. Settings 页面集成

- 在 `ScholarSubscriptionsPanel` 下方新增 `PushChannelsPanel`

## Acceptance Criteria

- [ ] 用户可以通过 UI 添加/编辑/删除推送通道
- [ ] 测试推送功能正常工作
- [ ] RSS 订阅地址可一键复制
- [ ] 与现有 YAML 配置双向兼容

## Dependencies

- Epic #179 后端推送基础设施（已完成）
