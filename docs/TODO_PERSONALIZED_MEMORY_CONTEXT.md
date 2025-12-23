# Personalized Research Memory + Context Engine — TODO

目标：把 PaperBot 定位为“个性化论文爬取与推荐平台”，并能记住用户科研进度/方向；当用户切换方向时，不会被旧方向记忆污染。

## P0（MVP）

- [x] Track + Progress（tracks/tasks）
- [x] Memory scoping：`global` / `track`（写入/检索按 scope 隔离）
- [x] Context Engine：输出 `ContextPack`（track + progress + memories + papers）
- [x] Paper feedback：`like/dislike/save/...`（用于过滤与加权）

## P1（可运营）

- [x] Memory Inbox：`/research/memory/suggest` 写入 `pending`，`/research/memory/inbox` 查看
- [x] 批量治理：`bulk_moderate / bulk_move / clear_track_memory`
- [x] Track Router（多特征）：keyword + memory hits + task overlap + (optional) embedding
- [ ] UI：Inbox 批量审核/迁移/清空确认
- [ ] 推荐多样性/探索：作者/主题多样性更强、探索比例可配置

## P2（护城河）

- [ ] Track/Profile embedding 预计算与后台更新
- [ ] 基于阅读进度阶段的推荐策略（survey / writing / rebuttal）
- [ ] 评测回放：离线 eval + 在线反馈闭环

