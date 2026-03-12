import { FileText, FlaskConical, Settings } from "lucide-react"

import DashboardDailyBriefView, {
  type DashboardActionItem,
  type DashboardDestinationCard,
  type DashboardHotPaper,
  type DashboardQueueItem,
} from "@/components/dashboard/DashboardDailyBriefView"
import { fetchDeadlineRadar, fetchLLMUsage, fetchPapers, fetchPipelineTasks } from "@/lib/api"
import { fetchLatestDashboardDailyBrief } from "@/lib/dashboard-brief"
import { buildDashboardIntelligenceCards } from "@/lib/dashboard-intelligence"
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

type PriorityLevel = "high" | "medium" | "low"

export const dynamic = "force-dynamic"

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

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
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

function buildQueuePreviewItems(
  readingQueue: ReadingQueueItem[],
  paperMap: Map<string, Paper>,
  activeTrack: ResearchTrackSummary | null,
): DashboardQueueItem[] {
  const fallbackTags = (activeTrack?.keywords || []).slice(0, 2)

  return readingQueue.slice(0, 4).map((item, index) => {
    const paper = paperMap.get(String(item.paper_id || item.id))
    const tags = (
      paper?.tags?.length
        ? paper.tags
        : fallbackTags.length
          ? fallbackTags
          : ["Reading Queue"]
    ).slice(0, 3)

    return {
      id: item.id,
      title: item.title,
      venue:
        paper?.venue ||
        (item.authors
          ? item.authors.split(",").slice(0, 2).join(", ")
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
  queueItems: DashboardQueueItem[]
  usageSummary: LLMUsageSummary
  signalCount: number
  tracks: ResearchTrackSummary[]
}): { now: DashboardActionItem[]; later: DashboardActionItem[] } {
  const { tasks, deadlines, queueItems, usageSummary, signalCount, tracks } = args
  const failedTask = tasks.find((task) => task.status === "failed")
  const urgentDeadline = deadlines.find((deadline) => deadline.days_left <= 14)
  const highPriorityItem = queueItems.find((item) => item.priority === "high")

  const now: DashboardActionItem[] = []
  const later: DashboardActionItem[] = []

  if (failedTask) {
    now.push({
      title: `修复 ${failedTask.paper_title}`,
      copy: "后台抓取或预处理失败会直接影响今天的候选质量，这个问题不应该继续留在首页之外。",
      metaLeft: failedTask.started_at,
      metaRight: "阻塞项",
      tone: "bad",
      href: "/workflows",
    })
  }

  if (urgentDeadline) {
    now.push({
      title: `${urgentDeadline.name} 还有 ${urgentDeadline.days_left} 天`,
      copy: "把 deadline 拉进行动区，而不是让它继续漂浮在边缘模块里。",
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
      copy: "先判断当前最接近焦点问题的论文，避免注意力被长列表继续稀释。",
      metaLeft: highPriorityItem.venue,
      metaRight: highPriorityItem.time,
      tone: "warn",
      href: highPriorityItem.href,
    })
  }

  if (now.length === 0) {
    now.push({
      title: "今天没有立即阻塞项",
      copy: "可以直接回到完整工作台推进 Search 或 DailyPaper，把首页留给判断和交接。",
      metaLeft: "Calm mode",
      metaRight: "继续工作",
      tone: "good",
      href: "/workflows",
    })
  }

  if (usageSummary.totals.calls > 0 || usageSummary.totals.total_cost_usd > 0) {
    later.push({
      title: `近 ${usageSummary.window_days} 天 ${usageSummary.totals.calls} 次模型调用`,
      copy: `累计成本 ${formatCurrency(usageSummary.totals.total_cost_usd)}。成本保留在侧栏与设置入口，不再挤占热点区域。`,
      metaLeft: "Usage",
      metaRight: "Settings",
      tone: "info",
      href: "/settings",
    })
  }

  if (signalCount > 0) {
    later.push({
      title: `${signalCount} 条社区信号需要回看`,
      copy: "趋势信息已经压缩到雷达区，只有当它影响当前判断时才需要继续下钻。",
      metaLeft: "Signals",
      metaRight: "Trend Radar",
      tone: "info",
      href: "#trend-radar",
    })
  }

  if (tracks.length > 1) {
    later.push({
      title: `${tracks.length - 1} 个非焦点 Track 等待回顾`,
      copy: "非焦点主题继续存在，但不会再和今天的主问题争抢首页主区。",
      metaLeft: "Research",
      metaRight: "稍后处理",
      tone: "info",
      href: "/research",
    })
  }

  if (later.length === 0) {
    later.push({
      title: "当前没有额外维护项",
      copy: "今天可以把注意力集中在热点候选和趋势判断，不需要切换页面清理杂项。",
      metaLeft: "Workspace",
      metaRight: "清爽",
      tone: "info",
      href: "/workflows",
    })
  }

  return {
    now: now.slice(0, 3),
    later: later.slice(0, 3),
  }
}

function buildHotPapers(args: {
  digest: Awaited<ReturnType<typeof fetchLatestDashboardDailyBrief>>
  trackFeedItems: TrackFeedItem[]
  activeTrack: ResearchTrackSummary | null
}): DashboardHotPaper[] {
  const { digest, trackFeedItems, activeTrack } = args

  if (digest?.highlights.length) {
    return digest.highlights.map((item) => ({
      id: item.id,
      title: item.title,
      href: item.href,
      sourceLabel: digest.sourceLabel,
      queryLabel: item.queryLabel,
      metricLabel: item.metricLabel,
      summary: item.summary,
      metaLabel: item.venueLabel,
      tags: item.tags.length ? item.tags : item.sourceBadges,
      recommendation: item.recommendation,
    }))
  }

  return trackFeedItems.slice(0, 4).map((item) => {
    const recommendation = item.latest_judge?.recommendation
      ? String(item.latest_judge.recommendation)
      : null
    const judgeScore = item.latest_judge?.overall

    return {
      id: String(item.paper.id),
      title: item.paper.title,
      href: `/papers/${item.paper.id}`,
      sourceLabel: activeTrack?.name || "Track feed",
      queryLabel: item.matched_terms.slice(0, 2).join(" · ") || "Personalized feed",
      metricLabel: judgeScore != null ? `Judge ${Number(judgeScore).toFixed(1)}` : `Feed ${item.feed_score.toFixed(1)}`,
      summary: recommendation
        ? `当前推荐等级：${recommendation}。这条结果已经进入焦点 Track 的个性化候选流。`
        : "这条结果已经进入焦点 Track 的个性化候选流，适合作为首页热点候选。",
      metaLabel: [item.paper.venue, item.paper.year].filter(Boolean).join(" · ") || "Research feed",
      tags: item.matched_terms.slice(0, 3),
      recommendation,
    }
  })
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)))
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
    digestResult,
  ] = await Promise.allSettled([
    fetchDashboardTracks("default"),
    fetchPipelineTasks(),
    fetchDashboardReadingQueue("default", 8),
    fetchLLMUsage(14),
    fetchDeadlineRadar("default"),
    fetchIntelligenceFeed("default", 6, { sortBy: "delta", sortOrder: "desc" }),
    fetchPapers(),
    fetchLatestDashboardDailyBrief(),
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
  const digest = digestResult.status === "fulfilled" ? digestResult.value : null

  const deadlines = [...deadlinesRaw].sort((left, right) => left.days_left - right.days_left)
  const activeTrack = tracks.find((track) => track.is_active) || tracks[0] || null
  const activeTrackFeedResult = activeTrack
    ? await fetchDashboardTrackFeed(activeTrack.id, "default", 4).catch(() => ({ items: [], total: 0 }))
    : { items: [], total: 0 }
  const queueItems = buildQueuePreviewItems(
    readingQueue,
    new Map(papers.map((paper) => [String(paper.id), paper])),
    activeTrack,
  )
  const hotPapers = buildHotPapers({
    digest,
    trackFeedItems: activeTrackFeedResult.items,
    activeTrack,
  })

  const highPriorityQueue = queueItems.filter((item) => item.priority === "high").length
  const urgentDeadlines = deadlines.filter((deadline) => deadline.days_left <= 14)
  const failedTasks = tasks.filter((task) => task.status === "failed")
  const signalCount = intelligenceFeed.items.length
  const alertCount = failedTasks.length + urgentDeadlines.length + (highPriorityQueue > 0 ? 1 : 0)
  const lanes = buildActionLanes({
    tasks,
    deadlines,
    queueItems,
    usageSummary,
    signalCount,
    tracks,
  })

  const trendCards = buildDashboardIntelligenceCards(intelligenceFeed.items)
  const intelligenceSourceSummary = signalCount > 0
    ? Array.from(
        trendCards.reduce((map, item) => {
          map.set(item.sourceLabel, (map.get(item.sourceLabel) || 0) + 1)
          return map
        }, new Map<string, number>()),
      )
        .map(([label, count]) => ({ label, count }))
        .sort((left, right) => right.count - left.count)
    : []

  const trendTopics = uniqueStrings([
    ...(digest?.queryPulse || []).map((item) => item.query),
    ...(activeTrack?.keywords || []),
    ...(intelligenceFeed.keywords || []),
  ]).slice(0, 8)

  const destinationCards: DashboardDestinationCard[] = [
    {
      title: "Research Workspace",
      description: activeTrack
        ? `继续在 ${activeTrack.name} 里查看完整 Track feed、anchors 和深度研究上下文。`
        : "在 Research 里建立 Track、整理上下文并继续深度探索。",
      metric: activeTrack ? `${activeTrackFeedResult.total} 条更新` : "建立焦点",
      href: activeTrack ? `/research?track_id=${activeTrack.id}` : "/research",
      icon: FlaskConical,
    },
    {
      title: "Papers Library",
      description: "保存库、导出和 BibTeX 继续留在 Papers，不再和首页主路径抢同一层级。",
      metric: `${papers.length} 篇文献`,
      href: "/papers",
      icon: FileText,
    },
    {
      title: "Settings & Delivery",
      description: "推送渠道、Provider 和模型成本都收回配置页，首页只保留影响决策的摘要。",
      metric: `${usageSummary.totals.calls} calls · ${formatCurrency(usageSummary.totals.total_cost_usd)}`,
      href: "/settings",
      icon: Settings,
    },
  ]

  return (
    <DashboardDailyBriefView
      greeting={getGreeting()}
      focusLabel={activeTrack?.name || "Global Radar"}
      alertCount={alertCount}
      signalCount={signalCount}
      libraryCount={papers.length}
      workflowCostLabel={`${usageSummary.totals.calls} calls · ${formatCurrency(usageSummary.totals.total_cost_usd)}`}
      workflowHref="/workflows"
      researchHref={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
      digest={digest}
      hotPapers={hotPapers}
      trendCards={trendCards}
      hasSignals={signalCount > 0}
      trendTopics={trendTopics}
      intelligenceRefreshedAt={intelligenceFeed.refreshed_at}
      intelligenceSourceSummary={intelligenceSourceSummary}
      watchedRepos={intelligenceFeed.watch_repos || []}
      watchedSubreddits={intelligenceFeed.subreddits || []}
      nowItems={lanes.now}
      laterItems={lanes.later}
      queueItems={queueItems}
      highPriorityQueue={highPriorityQueue}
      destinations={destinationCards}
    />
  )
}
