import Link from "next/link"
import type { ReactNode } from "react"
import {
  ArrowRight,
  BookOpen,
  type LucideIcon,
  RadioTower,
  Sparkles,
} from "lucide-react"

import type { DashboardDailyBrief } from "@/lib/dashboard-brief"
import type { DashboardIntelligenceCard } from "@/lib/dashboard-intelligence"

export type DashboardDecisionTone = "good" | "warn" | "bad" | "info"

export type DashboardActionItem = {
  title: string
  copy: string
  metaLeft: string
  metaRight: string
  tone: DashboardDecisionTone
  href?: string
}

export type DashboardQueueItem = {
  id: string
  title: string
  venue: string
  tags: string[]
  time: string
  priority: "high" | "medium" | "low"
  href: string
}

export type DashboardDestinationCard = {
  title: string
  description: string
  metric: string
  href: string
  icon: LucideIcon
}

export type DashboardHotPaper = {
  id: string
  title: string
  href: string
  sourceLabel: string
  queryLabel: string
  metricLabel: string
  summary: string
  metaLabel: string
  tags: string[]
  recommendation?: string | null
}

interface DashboardDailyBriefViewProps {
  greeting: string
  focusLabel: string
  alertCount: number
  signalCount: number
  libraryCount: number
  workflowCostLabel: string
  workflowHref: string
  researchHref: string
  digest: DashboardDailyBrief | null
  hotPapers: DashboardHotPaper[]
  trendCards: DashboardIntelligenceCard[]
  hasSignals: boolean
  trendTopics: string[]
  intelligenceRefreshedAt?: string | null
  intelligenceSourceSummary: Array<{ label: string; count: number }>
  watchedRepos: string[]
  watchedSubreddits: string[]
  nowItems: DashboardActionItem[]
  laterItems: DashboardActionItem[]
  queueItems: DashboardQueueItem[]
  highPriorityQueue: number
  destinations: DashboardDestinationCard[]
}

const TONE_CLASSES: Record<DashboardDecisionTone, string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-700",
  warn: "border-amber-200 bg-amber-50 text-amber-700",
  bad: "border-rose-200 bg-rose-50 text-rose-700",
  info: "border-sky-200 bg-sky-50 text-sky-700",
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return "刚刚"

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value

  const diffMs = Date.now() - parsed.getTime()
  const diffMinutes = Math.max(1, Math.floor(diffMs / 60_000))
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 60) return `${diffMinutes} 分钟前`
  if (diffHours < 24) return `${diffHours} 小时前`
  if (diffDays === 1) return "昨天"
  if (diffDays < 7) return `${diffDays} 天前`

  return parsed.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" })
}

function formatBriefDate(value?: string | null): string {
  if (!value) return "等待简报"

  const parsed = new Date(value)
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleDateString("zh-CN", {
      month: "numeric",
      day: "numeric",
    })
  }

  return value
}

function TonePill({ tone, children }: { tone: DashboardDecisionTone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  )
}

function StatTile({
  label,
  value,
  helper,
}: {
  label: string
  value: string | number
  helper: string
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/80 p-4 shadow-sm backdrop-blur">
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-bold tracking-tight text-slate-900">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{helper}</p>
    </div>
  )
}

function HotPaperCard({ item, index }: { item: DashboardHotPaper; index: number }) {
  const content = (
    <article className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-transform duration-200 hover:-translate-y-0.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-700">
          {item.sourceLabel}
        </span>
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
          {item.queryLabel}
        </span>
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
          {item.metricLabel}
        </span>
        {item.recommendation ? (
          <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[11px] font-medium text-white">
            {item.recommendation}
          </span>
        ) : null}
      </div>

      <div className="mt-4 flex items-start justify-between gap-3">
        <h3 className="text-lg font-semibold leading-7 text-slate-900">{item.title}</h3>
        <span className="text-xs font-semibold text-slate-400">#{index + 1}</span>
      </div>

      <p className="mt-2 text-sm leading-6 text-slate-600">{item.summary}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {item.tags.map((tag) => (
          <span
            key={`${item.id}-${tag}`}
            className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-medium text-sky-700"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="mt-auto flex items-center justify-between gap-3 border-t border-slate-100 pt-4">
        <span className="text-xs text-slate-500">{item.metaLabel}</span>
        <span className="inline-flex items-center gap-1 text-sm font-semibold text-slate-900">
          打开
          <ArrowRight size={15} />
        </span>
      </div>
    </article>
  )

  const isExternal = /^https?:\/\//.test(item.href)
  if (isExternal) {
    return (
      <a href={item.href} target="_blank" rel="noreferrer" className="block">
        {content}
      </a>
    )
  }

  return (
    <Link href={item.href} className="block">
      {content}
    </Link>
  )
}

function SignalCard({ item }: { item: DashboardIntelligenceCard }) {
  return (
    <article className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold text-slate-700">
          {item.sourceLabel}
        </span>
        <span className="text-xs text-slate-500">{formatRelativeTime(item.timestamp)}</span>
      </div>

      <h3 className="mt-3 text-base font-semibold leading-6 text-slate-900">{item.title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{item.summary}</p>

      {item.reasonChips.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {item.reasonChips.slice(0, 3).map((reason) => (
            <span
              key={`${item.id}-${reason}`}
              className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600"
            >
              {reason}
            </span>
          ))}
        </div>
      ) : null}

      <div className="mt-auto flex items-center justify-between gap-3 border-t border-slate-100 pt-4">
        <span className="text-xs text-slate-500">{item.metricLabel}</span>
        <div className="flex items-center gap-3">
          <Link
            href={item.researchHref}
            className="inline-flex items-center gap-1 text-sm font-semibold text-slate-900"
          >
            {item.researchLabel}
            <ArrowRight size={15} />
          </Link>
          {item.isExternal ? (
            <Link
              href={item.href}
              target="_blank"
              rel="noreferrer"
              className="text-xs font-medium text-slate-500"
            >
              来源
            </Link>
          ) : null}
        </div>
      </div>
    </article>
  )
}

function DecisionRail({
  nowItems,
  laterItems,
  queueItems,
  highPriorityQueue,
  destinations,
}: {
  nowItems: DashboardActionItem[]
  laterItems: DashboardActionItem[]
  queueItems: DashboardQueueItem[]
  highPriorityQueue: number
  destinations: DashboardDestinationCard[]
}) {
  function renderActionCard(item: DashboardActionItem, key: string) {
    const content = (
      <div className="rounded-2xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100">
        <div className="flex items-center justify-between gap-3">
          <TonePill tone={item.tone}>{item.metaRight}</TonePill>
          <span className="text-xs text-slate-500">{item.metaLeft}</span>
        </div>
        <h4 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h4>
        <p className="mt-1 text-sm leading-6 text-slate-600">{item.copy}</p>
      </div>
    )

    return item.href ? (
      <Link key={key} href={item.href} className="block">
        {content}
      </Link>
    ) : (
      <div key={key}>{content}</div>
    )
  }

  return (
    <aside className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Decision Rail</p>
        <h2 className="mt-2 text-xl font-bold tracking-tight text-slate-900">今天先处理什么</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          提醒、队列和工作台入口压进一条侧栏，首页主区只保留热点与趋势。
        </p>
      </div>

      <div className="mt-5 space-y-5">
        <section>
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-900">现在要处理</h3>
            <TonePill tone="warn">{nowItems.length} 项</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {nowItems.map((item, index) => renderActionCard(item, `now-${index}`))}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-900">稍后处理</h3>
            <TonePill tone="info">{laterItems.length} 项</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {laterItems.map((item, index) => renderActionCard(item, `later-${index}`))}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-900">阅读队列</h3>
            <TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} 篇高优</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {queueItems.length > 0 ? (
              queueItems.map((item) => (
                <Link
                  key={item.id}
                  href={item.href}
                  className="block rounded-2xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100"
                >
                  <div className="flex items-center justify-between gap-3">
                    <TonePill tone={item.priority === "high" ? "warn" : item.priority === "medium" ? "info" : "good"}>
                      {item.priority === "high" ? "高优" : item.priority === "medium" ? "中优" : "低优"}
                    </TonePill>
                    <span className="text-xs text-slate-500">{item.time}</span>
                  </div>
                  <h4 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h4>
                  <p className="mt-1 text-sm leading-6 text-slate-600">{item.venue}</p>
                </Link>
              ))
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-4 text-sm leading-6 text-slate-600">
                队列里还没有候选。先在 Workflows 跑一轮 Search 或 DailyPaper，把今天需要决策的论文拉起来。
              </div>
            )}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-900">工作台入口</h3>
            <span className="text-xs text-slate-500">{destinations.length} 个空间</span>
          </div>
          <div className="mt-3 space-y-2">
            {destinations.map((destination) => {
              const Icon = destination.icon
              return (
                <Link
                  key={destination.title}
                  href={destination.href}
                  className="flex items-start justify-between gap-3 rounded-2xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100"
                >
                  <div className="flex items-start gap-3">
                    <span className="mt-0.5 flex size-8 items-center justify-center rounded-2xl bg-white text-slate-700 shadow-sm">
                      <Icon size={16} />
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-slate-900">{destination.title}</p>
                      <p className="mt-1 text-sm leading-6 text-slate-600">{destination.description}</p>
                    </div>
                  </div>
                  <span className="text-xs font-semibold text-slate-500">{destination.metric}</span>
                </Link>
              )
            })}
          </div>
        </section>
      </div>
    </aside>
  )
}

export default function DashboardDailyBriefView({
  greeting,
  focusLabel,
  alertCount,
  signalCount,
  libraryCount,
  workflowCostLabel,
  workflowHref,
  researchHref,
  digest,
  hotPapers,
  trendCards,
  hasSignals,
  trendTopics,
  intelligenceRefreshedAt,
  intelligenceSourceSummary,
  watchedRepos,
  watchedSubreddits,
  nowItems,
  laterItems,
  queueItems,
  highPriorityQueue,
  destinations,
}: DashboardDailyBriefViewProps) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(245,158,11,0.08),_transparent_30%),linear-gradient(180deg,_#fcfcfb_0%,_#f6f4ef_100%)] pb-12 text-slate-900">
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="space-y-4">
          <section className="rounded-[32px] border border-stone-200 bg-[linear-gradient(135deg,rgba(255,255,255,0.96),rgba(251,247,236,0.96))] p-6 shadow-[0_24px_60px_rgba(15,23,42,0.06)]">
            <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
              <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-700">
                  <Sparkles className="size-3.5" />
                  Daily Research Brief
                </div>
                <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                  {greeting}，今天先看热点，再做判断。
                </h1>
                <p className="mt-3 text-sm leading-6 text-slate-600">
                  Dashboard 现在按日报逻辑组织信息：DailyPaper 负责今日热点，Signals 负责趋势雷达，行动队列统一收进右侧决策栏。完整编排和运行控制回到
                  <span className="font-semibold text-slate-900"> Workflows</span>。
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <Link
                  href={workflowHref}
                  className="inline-flex min-h-11 items-center gap-2 rounded-full bg-slate-900 px-5 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
                >
                  打开完整工作台
                  <ArrowRight size={15} />
                </Link>
                <Link
                  href={researchHref}
                  className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
                >
                  进入 Research
                  <ArrowRight size={15} />
                </Link>
              </div>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <StatTile
                label="今日简报"
                value={formatBriefDate(digest?.date || digest?.generatedAt || null)}
                helper={digest ? `${digest.sourceLabel} · ${digest.stats.queryCount} 个主题` : "尚未生成今日简报"}
              />
              <StatTile
                label="热点候选"
                value={hotPapers.length}
                helper={hotPapers.length > 0 ? "首页已压缩成高优摘要" : "等待 DailyPaper 或 Track feed"}
              />
              <StatTile
                label="趋势信号"
                value={signalCount}
                helper={hasSignals ? `最近刷新 ${formatRelativeTime(intelligenceRefreshedAt)}` : "当前没有需要上浮的外部信号"}
              />
              <StatTile
                label="优先事项"
                value={alertCount}
                helper="失败任务、临近 deadline 与高优队列合并计算"
              />
            </div>

            <div className="mt-5 flex flex-wrap gap-2">
              <span className="rounded-full border border-white/70 bg-white/80 px-3 py-1 text-xs font-medium text-slate-600">
                Focus: {focusLabel}
              </span>
              <span className="rounded-full border border-white/70 bg-white/80 px-3 py-1 text-xs font-medium text-slate-600">
                Library {libraryCount} 篇
              </span>
              <span className="rounded-full border border-white/70 bg-white/80 px-3 py-1 text-xs font-medium text-slate-600">
                Usage {workflowCostLabel}
              </span>
            </div>
          </section>

          <section className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.35fr)_360px]">
            <div className="space-y-4">
              <article id="daily-brief" className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-3xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-700">Daily Push</p>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">今日热点推送</h2>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      优先展示 DailyPaper 最近一次产出的热点候选；如果还没有完成日报，就退回到 Track feed，让首页始终有一组可判断的高优论文。
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {(digest?.sourceBadges.length ? digest.sourceBadges : ["DailyPaper"]).slice(0, 2).map((badge) => (
                      <span
                        key={badge}
                        className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600"
                      >
                        {badge}
                      </span>
                    ))}
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                      {digest ? `${digest.stats.totalQueryHits} hits` : "等待候选"}
                    </span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                      {digest ? `${digest.stats.uniqueItems} unique` : "No brief"}
                    </span>
                  </div>
                </div>

                {digest?.queryPulse.length ? (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {digest.queryPulse.map((item) => (
                      <span
                        key={item.query}
                        className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-medium text-sky-700"
                      >
                        {item.query} · {item.hits}
                      </span>
                    ))}
                  </div>
                ) : null}

                <div className="mt-5 grid gap-3 lg:grid-cols-2">
                  {hotPapers.length > 0 ? (
                    hotPapers.map((item, index) => <HotPaperCard key={item.id} item={item} index={index} />)
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-6 text-sm leading-6 text-slate-600 lg:col-span-2">
                      还没有新的热点候选。去 Workflows 跑一轮 Search / DailyPaper，或者在 Research 里先建立焦点 Track，首页会自动回填新的热点推送。
                    </div>
                  )}
                </div>
              </article>

              <article id="trend-radar" className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-3xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-700">Trend Radar</p>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">趋势雷达</h2>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      社区信号、HF Daily 热度和关注关键词聚合在一处，不再把趋势信息拆散到多个独立模块。
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {intelligenceSourceSummary.slice(0, 3).map((source) => (
                      <span
                        key={source.label}
                        className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600"
                      >
                        {source.label} {source.count}
                      </span>
                    ))}
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                      刷新 {formatRelativeTime(intelligenceRefreshedAt)}
                    </span>
                  </div>
                </div>

                {trendTopics.length > 0 ? (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {trendTopics.map((topic) => (
                      <span
                        key={topic}
                        className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                ) : null}

                {digest?.trendRows.length ? (
                  <div className="mt-5 grid gap-3 lg:grid-cols-3">
                    {digest.trendRows.map((trend) => (
                      <div key={trend.query} className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                        <p className="text-sm font-semibold text-slate-900">{trend.query}</p>
                        <p className="mt-2 text-sm leading-6 text-slate-600">{trend.analysis}</p>
                      </div>
                    ))}
                  </div>
                ) : null}

                <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {hasSignals ? (
                    trendCards.map((card) => <SignalCard key={card.id} item={card} />)
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-6 text-sm leading-6 text-slate-600 md:col-span-2 xl:col-span-3">
                      当前没有需要上浮到首页的趋势信号。可以直接在 Workflows 继续推进候选生成，或者去 Research 维护新的 Track 关键词。
                    </div>
                  )}
                </div>

                {(watchedRepos.length > 0 || watchedSubreddits.length > 0) ? (
                  <div className="mt-5 grid gap-3 lg:grid-cols-2">
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <BookOpen className="size-4 text-slate-500" />
                        Watch Repos
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {watchedRepos.slice(0, 6).map((repo) => (
                          <span
                            key={repo}
                            className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600"
                          >
                            {repo}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <RadioTower className="size-4 text-slate-500" />
                        Watched Communities
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {watchedSubreddits.slice(0, 6).map((subreddit) => (
                          <span
                            key={subreddit}
                            className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600"
                          >
                            {subreddit}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}
              </article>
            </div>

            <DecisionRail
              nowItems={nowItems}
              laterItems={laterItems}
              queueItems={queueItems}
              highPriorityQueue={highPriorityQueue}
              destinations={destinations}
            />
          </section>
        </div>
      </main>
    </div>
  )
}
