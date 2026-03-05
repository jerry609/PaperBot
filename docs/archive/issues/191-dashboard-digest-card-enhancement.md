# Issue #191: Dashboard — Digest Card & Feed Enhancement

## Priority: P2

## Summary

Dashboard 的 Feed/Activity 区域需要升级，展示新增的 digest card 字段（highlight / method / finding / tags）和 Judge 推荐等级，提升信息密度和决策效率。

## Background

Epic #179 新增了 `digest_card`（highlight/method/finding/tags）和分层推荐（must_read / worth_reading / skim），但 Dashboard 和 Papers Feed 页面尚未使用这些新字段。

## Requirements

### 1. PaperCard 升级

- 在 `PaperCard` 组件中渲染 digest card 字段：
  - 💎 highlight 行（加粗，最醒目）
  - 🔬 method + 📌 finding（灰色小字）
  - tag pills（彩色标签）
- 推荐等级 badge：🔥 Must Read / 👍 Worth Reading / 📖 Skim
- 如果有 `main_figure`，显示缩略图

### 2. FeedTab 升级

- 支持按推荐等级筛选（All / Must Read / Worth Reading）
- 支持按 tag 筛选
- 显示 digest card 数据（从后端 track feed API 返回）

### 3. Dashboard ActivityFeed 升级

- 在 daily paper feed events 中展示 highlight 摘要而非原始 title
- 添加 tag pills

### 4. RSS 订阅入口

- 在 Dashboard 侧边栏或 Feed 区域添加 RSS 订阅按钮/图标
- 点击展示订阅 URL + 复制按钮

## Acceptance Criteria

- [ ] PaperCard 显示 highlight/tags/recommendation badge
- [ ] FeedTab 支持按推荐等级和 tag 筛选
- [ ] Dashboard 活动流展示 digest card 摘要
- [ ] RSS 订阅入口可用

## Dependencies

- Epic #179 后端推送基础设施（已完成）
- Issue #190（Push Channel UI，可独立实施）
