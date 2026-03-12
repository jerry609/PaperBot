import Link from "next/link"
import {
  ArrowRight,
  FileText,
  FlaskConical,
  Settings,
} from "lucide-react"

import DashboardActionBands, {
  type DashboardDestinationCard,
  type DashboardLaneItem,
  type DashboardQueuePreview,
} from "@/components/dashboard/DashboardActionBands"
import { fetchDeadlineRadar, fetchLLMUsage, fetchPapers, fetchPipelineTasks } from "@/lib/api"
import {
  buildDashboardIntelligenceCards,
  type DashboardIntelligenceCard,
} from "@/lib/dashboard-intelligence"
import {
  fetchDashboardReadingQueue,
  fetchDashboardTrackFeed,
  fetchDashboardTracks,
  fetchIntelligenceFeed,
} from "@/lib/dashboard-api"
import type {
  DeadlineRadarItem,
  LLMUsageSummary,
  Paper,
  PipelineTask,
  ReadingQueueItem,
  ResearchTrackSummary,
  TrackFeedItem,
} from "@/lib/types"

type DashboardRecommendationCardData = {
  id: string
  title: string
  href: string
  meta: string
  summary: string
  tags: string[]
  metric: string
  recommendation?: string | null
}

type PriorityLevel = "high" | "medium" | "low"

function getGreeting(): string {
  const hour = new Date().getHours()
  if (hour < 12) return "早上好"
  if (hour < 18) return "下午好"
  return "晚上好"
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

function getQueuePriority(item: ReadingQueueItem, index: number): PriorityLevel {
  if (typeof item.priority === "number") {
    if (item.priority <= 2) return "high"
    if (item.priority <= 4) return "medium"
    return "low"
  }

  if (index < 2) return "high"
  if (index < 4) return "medium"
  return "low"
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatRecommendationLabel(value?: string | null): string | null {
  const normalized = String(value || "").trim()
  if (!normalized) return null

  switch (normalized) {
    case "must_read":
      return "Must Read"
    case "worth_reading":
      return "Worth Reading"
    case "skim":
      return "Skim"
    case "skip":
      return "Skip"
    default:
      return normalized
  }
}

function buildQueuePreviewItems(
  readingQueue: ReadingQueueItem[],
  paperMap: Map<string, Paper>,
  activeTrack: ResearchTrackSummary | null,
): DashboardQueuePreview[] {
  const fallbackTags = (activeTrack?.keywords || []).slice(0, 2)

  return readingQueue.slice(0, 3).map((item, index) => {
    const paper = paperMap.get(String(item.paper_id || item.id))
    const tags = (
      paper?.tags?.length
        ? paper.tags
        : fallbackTags.length
          ? fallbackTags
          : ["Reading Queue"]
    ).slice(0, 2)

    return {
      id: item.id,
      title: item.title,
      venue:
        paper?.venue ||
        (item.authors
          ? `${item.authors.split(",").slice(0, 2).join(", ")}`
          : "Paper Library"),
      tags,
      time: formatRelativeTime(item.saved_at),
      priority: getQueuePriority(item, index),
      href: item.paper_id ? `/papers/${item.paper_id}` : "/papers",
    }
  })
}

function buildActionLanes(args: {
  tasks: PipelineTask[]
  deadlines: DeadlineRadarItem[]
  queueItems: DashboardQueuePreview[]
  usageSummary: LLMUsageSummary
  signalCount: number
  tracks: ResearchTrackSummary[]
}): { now: DashboardLaneItem[]; later: DashboardLaneItem[] } {
  const { tasks, deadlines, queueItems, usageSummary, signalCount, tracks } = args
  const failedTask = tasks.find((task) => task.status === "failed")
  const urgentDeadline = deadlines.find((deadline) => deadline.days_left <= 14)
  const highPriorityItem = queueItems.find((item) => item.priority === "high")

  const now: DashboardLaneItem[] = []
  const later: DashboardLaneItem[] = []

  if (failedTask) {
    now.push({
      title: `修复 ${failedTask.paper_title}`,
      copy: "后台抓取或预处理失败会直接影响下午的候选与 digest 质量。",
      metaLeft: failedTask.started_at,
      metaRight: "阻塞主流程",
      tone: "bad",
      href: "/workflows",
    })
  }

  if (urgentDeadline) {
    now.push({
      title: `${urgentDeadline.name} 还有 ${urgentDeadline.days_left} 天`,
      copy: "补齐材料和对照实验，避免 deadline 信息继续漂浮在首页边缘而没有真正进入行动队列。",
      metaLeft: urgentDeadline.field,
      metaRight: "高优先级",
      tone: "warn",
      href: urgentDeadline.matched_tracks[0]
        ? `/research?track_id=${urgentDeadline.matched_tracks[0].track_id}`
        : "/research",
    })
  }

  if (highPriorityItem) {
    now.push({
      title: `处理高优候选《${highPriorityItem.title}》`,
      copy: "把最接近当前焦点的问题优先决策，避免候选堆进长列表里继续增加认知负担。",
      metaLeft: highPriorityItem.venue,
      metaRight: highPriorityItem.time,
      tone: "warn",
      href: highPriorityItem.href,
    })
  }

  if (now.length === 0) {
    now.push({
      title: "今天没有立即阻塞项",
      copy: "可以直接回到主工作台跑 Search 或 DailyPaper，把注意力留给当前的 Focus Track。",
      metaLeft: "Calm mode",
      metaRight: "继续工作",
      tone: "good",
      href: "/workflows",
    })
  }

  if (usageSummary.totals.calls > 0 || usageSummary.totals.total_cost_usd > 0) {
    later.push({
      title: `近 ${usageSummary.window_days} 天完成 ${usageSummary.totals.calls} 次模型调用`,
      copy: `累计成本 ${formatCurrency(usageSummary.totals.total_cost_usd)}。模型使用概览被收进次级页，只在它影响决策时再回到首页。`,
      metaLeft: "Usage",
      metaRight: "Settings",
      tone: "info",
      href: "/settings",
    })
  }

  if (signalCount > 0) {
    later.push({
      title: `${signalCount} 条社区信号已压缩进证据快照`,
      copy: "首页只保留会影响当前判断的摘要，完整动态继续留在 Research 页面。",
      metaLeft: "Signals",
      metaRight: "Evidence",
      tone: "info",
      href: "#signals",
    })
  }

  if (tracks.length > 1) {
    later.push({
      title: `${tracks.length - 1} 个非焦点 Track 等待回顾`,
      copy: "这些 Track 继续存在，但不会再在首页和当前焦点争夺同一层级的注意力。",
      metaLeft: "Research",
      metaRight: "稍后处理",
      tone: "info",
      href: "/research",
    })
  }

  if (later.length === 0) {
    later.push({
      title: "当前没有额外维护项",
      copy: "今天可以把注意力完全集中在主工作台，不需要切换到别的页面清理配置或状态。",
      metaLeft: "Workspace",
      metaRight: "清爽",
      tone: "info",
      href: "/research",
    })
  }

  return {
    now: now.slice(0, 3),
    later: later.slice(0, 3),
  }
}

function buildRecommendationCards(args: {
  trackFeedItems: TrackFeedItem[]
  activeTrack: ResearchTrackSummary | null
}): DashboardRecommendationCardData[] {
  const { trackFeedItems, activeTrack } = args

  return trackFeedItems.slice(0, 4).map((item) => {
    const recommendation = formatRecommendationLabel(item.latest_judge?.recommendation)
    const judgeScore = item.latest_judge?.overall
    const metric = judgeScore != null
      ? `Judge ${Number(judgeScore).toFixed(1)}`
      : `Feed ${item.feed_score.toFixed(1)}`

    return {
      id: String(item.paper.id),
      title: item.paper.title,
      href: `/papers/${item.paper.id}`,
      meta: item.paper.venue || activeTrack?.name || "Recommendation",
      summary: recommendation
        ? `当前推荐等级为 ${recommendation}，已经进入今日优先判断列表。`
        : "这篇论文已进入今日候选池，建议先快速判断是否值得继续深入。",
      tags: item.matched_terms.slice(0, 3),
      metric,
      recommendation,
    }
  })
}

function getEvidenceStyles(source: string): {
  badgeClassName: string
  chipClassName: string
} {
  switch (source) {
    case "github":
      return {
        badgeClassName: "border-slate-200 bg-slate-50 text-slate-700",
        chipClassName: "border-slate-200 bg-slate-50 text-slate-600",
      }
    case "reddit":
      return {
        badgeClassName: "border-amber-200 bg-amber-50 text-amber-700",
        chipClassName: "border-amber-200 bg-amber-50 text-amber-700",
      }
    case "huggingface":
      return {
        badgeClassName: "border-sky-200 bg-sky-50 text-sky-700",
        chipClassName: "border-sky-200 bg-sky-50 text-sky-700",
      }
    default:
      return {
        badgeClassName: "border-emerald-200 bg-emerald-50 text-emerald-700",
        chipClassName: "border-slate-200 bg-slate-50 text-slate-600",
      }
  }
}

function SectionIntro({
  eyebrow,
  title,
  copy,
  actionHref,
  actionLabel,
}: {
  eyebrow: string
  title: string
  copy: string
  actionHref?: string
  actionLabel?: string
}) {
  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
      <div className="max-w-3xl">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">{eyebrow}</p>
        <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
          {title}
        </h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">{copy}</p>
      </div>

      {actionHref && actionLabel ? (
        <Link
          href={actionHref}
          className="inline-flex items-center gap-1.5 text-sm font-semibold text-indigo-600 transition-colors hover:text-indigo-700"
        >
          {actionLabel}
          <ArrowRight size={15} />
        </Link>
      ) : null}
    </div>
  )
}

function HeroStat({
  label,
  value,
  helper,
}: {
  label: string
  value: string | number
  helper: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-bold tracking-tight text-slate-900">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{helper}</p>
    </div>
  )
}

function RecommendationCard({ item }: { item: DashboardRecommendationCardData }) {
  return (
    <Link
      href={item.href}
      className="block rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors hover:bg-slate-50"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-700">
          {item.meta}
        </span>
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
          {item.metric}
        </span>
        {item.recommendation ? (
          <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[11px] font-medium text-white">
            {item.recommendation}
          </span>
        ) : null}
      </div>

      <h3 className="mt-4 text-lg font-semibold leading-7 text-slate-900">{item.title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{item.summary}</p>

      {item.tags.length > 0 ? (
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
      ) : null}

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-slate-100 pt-4">
        <span className="text-xs text-slate-500">打开论文</span>
        <span className="inline-flex items-center gap-1 text-sm font-semibold text-slate-900">
          查看
          <ArrowRight size={15} />
        </span>
      </div>
    </Link>
  )
}

function EvidencePreviewCard({
  item,
  compact = false,
}: {
  item: DashboardIntelligenceCard
  compact?: boolean
}) {
  const styles = getEvidenceStyles(item.source)

  return (
    <article className={`flex h-full flex-col rounded-2xl border border-slate-200 bg-white shadow-sm ${compact ? "p-3.5" : "p-4"}`}>
      <div className="flex items-start justify-between gap-3">
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${styles.badgeClassName}`}
        >
          {item.sourceLabel}
        </span>
        <span className="text-xs text-slate-500">{formatRelativeTime(item.timestamp)}</span>
      </div>

      <h3 className={`font-semibold text-slate-900 ${compact ? "mt-2 text-base leading-6" : "mt-3 text-lg leading-7"}`}>
        {item.title}
      </h3>
      <p className={`text-sm text-slate-600 ${compact ? "mt-1.5 leading-5" : "mt-2 leading-6"}`}>{item.summary}</p>

      {item.reasonChips.length > 0 ? (
        <div className={`flex flex-wrap gap-2 ${compact ? "mt-3" : "mt-4"}`}>
          {item.reasonChips.slice(0, compact ? 2 : 3).map((reason) => (
            <span
              key={`${item.id}-${reason}`}
              className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${styles.chipClassName}`}
            >
              {reason}
            </span>
          ))}
        </div>
      ) : null}

      <div className={`mt-auto flex items-center justify-between gap-3 border-t border-slate-100 ${compact ? "pt-3" : "pt-4"}`}>
        <span className="text-xs text-slate-500">{item.metricLabel}</span>
        <div className="flex items-center gap-3">
          <Link
            href={item.researchHref}
            className="inline-flex items-center gap-1 text-sm font-semibold text-indigo-600 transition-colors hover:text-indigo-700"
          >
            {item.researchLabel}
            <ArrowRight size={15} />
          </Link>
          {item.isExternal ? (
            <Link
              href={item.href}
              target="_blank"
              rel="noreferrer"
              className="text-xs font-medium text-slate-500 transition-colors hover:text-slate-900"
            >
              原始来源
            </Link>
          ) : null}
        </div>
      </div>
    </article>
  )
}

export default async function DashboardPage() {
  const [
    tracksResult,
    tasksResult,
    readingQueueResult,
    llmUsageResult,
    deadlineResult,
    intelligenceResult,
    papersResult,
  ] = await Promise.allSettled([
    fetchDashboardTracks("default"),
    fetchPipelineTasks(),
    fetchDashboardReadingQueue("default", 6),
    fetchLLMUsage(14),
    fetchDeadlineRadar("default"),
    fetchIntelligenceFeed("default", 6, { sortBy: "delta", sortOrder: "desc" }),
    fetchPapers(),
  ])

  const tracks = tracksResult.status === "fulfilled" ? tracksResult.value : []
  const tasks = tasksResult.status === "fulfilled" ? tasksResult.value : []
  const readingQueue = readingQueueResult.status === "fulfilled" ? readingQueueResult.value : []
  const usageSummary = llmUsageResult.status === "fulfilled"
    ? llmUsageResult.value
    : {
        window_days: 14,
        daily: [],
        provider_models: [],
        totals: { calls: 0, total_tokens: 0, total_cost_usd: 0 },
      }
  const deadlinesRaw = deadlineResult.status === "fulfilled" ? deadlineResult.value : []
  const intelligenceFeed = intelligenceResult.status === "fulfilled"
    ? intelligenceResult.value
    : { items: [], refreshed_at: null, refresh_scheduled: false, keywords: [], watch_repos: [], subreddits: [] }
  const papers = papersResult.status === "fulfilled" ? papersResult.value : []

  const deadlines = [...deadlinesRaw].sort((left, right) => left.days_left - right.days_left)
  const activeTrack = tracks.find((track) => track.is_active) || tracks[0] || null
  const activeTrackFeedResult = activeTrack
    ? await fetchDashboardTrackFeed(activeTrack.id, "default", 4).catch(() => ({ items: [], total: 0 }))
    : { items: [], total: 0 }
  const activeTrackFeedTotal = activeTrackFeedResult.total || 0
  const recommendationCards = buildRecommendationCards({
    trackFeedItems: activeTrackFeedResult.items || [],
    activeTrack,
  })
  const intelligenceCards = buildDashboardIntelligenceCards(intelligenceFeed.items)
  const signalCards = intelligenceCards.slice(0, 4)
  const queueItems = buildQueuePreviewItems(
    readingQueue,
    new Map(papers.map((paper) => [String(paper.id), paper])),
    activeTrack,
  )

  const highPriorityQueue = queueItems.filter((item) => item.priority === "high").length
  const failedTasks = tasks.filter((task) => task.status === "failed")
  const urgentDeadlines = deadlines.filter((deadline) => deadline.days_left <= 14)
  const signalCount = intelligenceCards.length
  const alertCount = failedTasks.length + urgentDeadlines.length + (highPriorityQueue > 0 ? 1 : 0)
  const greeting = getGreeting()
  const libraryCount = papers.length

  const lanes = buildActionLanes({
    tasks,
    deadlines,
    queueItems,
    usageSummary,
    signalCount,
    tracks,
  })
  const destinationCards: DashboardDestinationCard[] = [
    {
      title: "Research Workspace",
      description: activeTrack
        ? `继续在 ${activeTrack.name} 里查看完整 Track feed、anchors 和深度研究上下文。`
        : "在 Research 里建立 Track、整理上下文并继续深度探索。",
      metric: activeTrack ? `${activeTrackFeedTotal} 条更新` : "创建焦点",
      href: activeTrack ? `/research?track_id=${activeTrack.id}` : "/research",
      icon: FlaskConical,
    },
    {
      title: "Papers Library",
      description: "保存库、导出和 BibTeX 继续留在 Papers，不再和首页主路径抢同一层级。",
      metric: `${libraryCount} 篇文献`,
      href: "/papers",
      icon: FileText,
    },
    {
      title: "Settings & Delivery",
      description: "Provider、投递渠道和模型使用概览放回配置页，首页只保留会影响决策的摘要。",
      metric: `${usageSummary.totals.calls} calls · ${formatCurrency(usageSummary.totals.total_cost_usd)}`,
      href: "/settings",
      icon: Settings,
    },
  ]

  return (
    <div className="min-h-screen bg-stone-50/50 pb-12 text-slate-900">
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="space-y-4">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">Dashboard</p>
              <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                {greeting}，今天先看推荐，再看趋势。
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                借鉴 PaperMind 的首页思路，Dashboard 只回答三个问题：今天该看什么、最近在变什么、接下来去哪继续做。复杂控制全部留给 `/workflows` 和其他专业页面。
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <Link
                href="/workflows"
                className="inline-flex min-h-11 items-center gap-2 rounded-full bg-slate-900 px-5 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
              >
                打开完整工作台
                <ArrowRight size={15} />
              </Link>
              <Link
                href={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
                className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
              >
                进入 Research
                <ArrowRight size={15} />
              </Link>
            </div>
          </header>

          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <HeroStat
              label="Focus"
              value={activeTrack?.name || "未设置"}
              helper={activeTrack ? `${activeTrackFeedTotal} 条相关更新` : "先建立一个 Research Track"}
            />
            <HeroStat
              label="Recommendations"
              value={recommendationCards.length}
              helper={recommendationCards.length > 0 ? "今日推荐已经准备好" : "等待新的候选进入推荐池"}
            />
            <HeroStat
              label="Signals"
              value={signalCount}
              helper={signalCount > 0 ? `最近刷新 ${formatRelativeTime(intelligenceFeed.refreshed_at)}` : "当前没有上浮信号"}
            />
            <HeroStat
              label="Alerts"
              value={alertCount}
              helper={`${highPriorityQueue} 篇高优候选 · ${urgentDeadlines.length} 个临近 deadline`}
            />
          </section>

          <section className="mt-4" id="recommendations">
            <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
              <SectionIntro
                eyebrow="Daily Recommendations"
                title="今日推荐"
                copy="优先展示当前焦点 Track 里最值得看的几篇论文，让首页先给出答案，而不是先给控制台。"
                actionHref={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
                actionLabel="查看完整候选"
              />

              <div className="mt-5 grid gap-3 lg:grid-cols-2">
                {recommendationCards.length > 0 ? (
                  recommendationCards.map((item) => (
                    <RecommendationCard key={item.id} item={item} />
                  ))
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/50 p-5 text-sm leading-6 text-slate-600 lg:col-span-2">
                    当前还没有可展示的推荐。去 `/workflows` 跑一轮 Search / DailyPaper，或者在 Research 里先建立焦点 Track，首页会自动回填新的推荐。
                  </div>
                )}
              </div>
            </article>
          </section>

          <section className="mt-4" id="signals">
            <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
              <SectionIntro
                eyebrow="Daily Trends"
                title="每日趋势"
                copy="趋势模块只保留最近真正会影响判断的变化。复杂分析继续放在 Research 和 Workflows，首页只展示结果。"
                actionHref="/research"
                actionLabel="查看完整动态"
              />

              <div className="mt-4 flex flex-wrap gap-2">
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                  {signalCount} 条信号
                </span>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                  {intelligenceCards.filter((item) => item.source === "github").length} 条 GitHub
                </span>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                  最近刷新 {formatRelativeTime(intelligenceFeed.refreshed_at)}
                </span>
              </div>

              <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {signalCards.length > 0 ? (
                  signalCards.map((item) => (
                    <EvidencePreviewCard key={item.id} item={item} />
                  ))
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/50 p-5 text-sm leading-6 text-slate-600 md:col-span-2 xl:col-span-3">
                    当前没有需要上浮到首页的社区信号。可以直接在主工作台继续推进当前问题。
                  </div>
                )}
              </div>
            </article>
          </section>

          <DashboardActionBands
            nowItems={lanes.now}
            laterItems={lanes.later}
            destinations={destinationCards}
            queueItems={queueItems}
            highPriorityQueue={highPriorityQueue}
          />
        </div>
      </main>
    </div>
  )
}
