import Link from "next/link"
import {
  Activity,
  ArrowRight,
  Clock,
  FileText,
  FlaskConical,
  Layers,
  Settings,
  Sparkles,
  type LucideIcon,
} from "lucide-react"

import TopicWorkflowDashboard from "@/components/research/TopicWorkflowDashboard"
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

type DashboardPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>
}

type PriorityLevel = "high" | "medium" | "low"
type Tone = "good" | "warn" | "bad" | "info"

type QueuePreview = {
  id: string
  title: string
  venue: string
  tags: string[]
  time: string
  priority: PriorityLevel
  href: string
}

type BriefCardData = {
  label: string
  pill: string
  tone: Tone
  title: string
  copy: string
  metaLeft: string
  metaRight: string
}

type LaneItemData = {
  title: string
  copy: string
  metaLeft: string
  metaRight: string
  tone: Tone
  href?: string
}

type DestinationCardData = {
  title: string
  description: string
  metric: string
  href: string
  icon: LucideIcon
}

const TONE_PILL_CLASSES: Record<Tone, string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-700",
  warn: "border-amber-200 bg-amber-50 text-amber-700",
  bad: "border-rose-200 bg-rose-50 text-rose-700",
  info: "border-indigo-200 bg-indigo-50 text-indigo-700",
}

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

function buildQueuePreviewItems(
  readingQueue: ReadingQueueItem[],
  paperMap: Map<string, Paper>,
  activeTrack: ResearchTrackSummary | null,
): QueuePreview[] {
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

function buildBriefCards(args: {
  activeTrack: ResearchTrackSummary | null
  activeTrackFeed: TrackFeedItem[]
  activeTrackFeedTotal: number
  tasks: PipelineTask[]
  deadlines: DeadlineRadarItem[]
  queueItems: QueuePreview[]
}): BriefCardData[] {
  const { activeTrack, activeTrackFeed, activeTrackFeedTotal, tasks, deadlines, queueItems } = args
  const failedTasks = tasks.filter((task) => task.status === "failed")
  const runningTasks = tasks.filter((task) => task.status !== "success" && task.status !== "failed")
  const latestTask = tasks[0]
  const nearestDeadline = deadlines[0]
  const highPriorityQueue = queueItems.filter((item) => item.priority === "high").length
  const latestTrackPaper = activeTrackFeed[0]?.paper.title

  return [
    {
      label: "Focus",
      pill: activeTrack ? "当前焦点" : "待设置",
      tone: activeTrack ? (activeTrackFeedTotal > 0 ? "good" : "info") : "warn",
      title: activeTrack ? activeTrack.name : "还没有活跃的 Focus Track",
      copy: activeTrack
        ? activeTrackFeedTotal > 0
          ? latestTrackPaper
            ? `最近 ${activeTrackFeedTotal} 条相关更新已经进入这个 Track，最新命中是《${latestTrackPaper}》。`
            : `最近 ${activeTrackFeedTotal} 条相关更新已经进入这个 Track。`
          : activeTrack.description || "这个 Track 暂时没有新的 feed，可以直接从工作台继续 Search。"
        : "先去 Research 创建一个 Focus Track，再回到首页继续推进今天的主题。",
      metaLeft: activeTrack ? `${activeTrackFeedTotal} 条更新` : "前往 Research",
      metaRight: activeTrack ? "Research" : "未设置",
    },
    {
      label: "Pipeline",
      pill: failedTasks.length > 0 ? "需要处理" : runningTasks.length > 0 ? "进行中" : "平稳",
      tone: failedTasks.length > 0 ? "bad" : runningTasks.length > 0 ? "info" : "good",
      title:
        failedTasks.length > 0
          ? `${failedTasks.length} 个后台任务需要处理`
          : runningTasks.length > 0
            ? `${runningTasks.length} 个后台任务正在运行`
            : "后台任务当前平稳",
      copy:
        failedTasks.length > 0
          ? `最近失败任务：${failedTasks[0]?.paper_title || "unknown task"}。`
          : latestTask
            ? `最近任务：${latestTask.paper_title}。`
            : "近期开启的抓取或预处理任务会显示在这里。",
      metaLeft: latestTask ? `最近任务 ${latestTask.started_at}` : "暂无最近任务",
      metaRight:
        failedTasks.length > 0
          ? `${failedTasks.length} 失败`
          : runningTasks.length > 0
            ? `${runningTasks.length} 运行中`
            : "无阻塞",
    },
    {
      label: "Deadline",
      pill:
        nearestDeadline && nearestDeadline.days_left <= 14
          ? "临近"
          : nearestDeadline
            ? "已跟踪"
            : "平静",
      tone:
        nearestDeadline && nearestDeadline.days_left <= 14
          ? "warn"
          : nearestDeadline
            ? "info"
            : "good",
      title: nearestDeadline
        ? `${nearestDeadline.name} 还剩 ${nearestDeadline.days_left} 天`
        : "近期没有紧迫截稿",
      copy: nearestDeadline
        ? `${nearestDeadline.field} 相关的提交窗口已经进入 Radar。`
        : "Radar 中没有 14 天内需要优先处理的会议或 workshop 截止。",
      metaLeft: nearestDeadline ? "Deadline Radar" : "保持当前节奏",
      metaRight: nearestDeadline ? nearestDeadline.ccf_level || "Tracked" : "无紧迫项",
    },
    {
      label: "Queue",
      pill: queueItems.length > 0 ? "待处理" : "空队列",
      tone:
        queueItems.length === 0
          ? "info"
          : highPriorityQueue > 0
            ? "warn"
            : "good",
      title:
        queueItems.length > 0
          ? `今天有 ${queueItems.length} 篇候选等待处理`
          : "今天的待读队列还是空的",
      copy:
        queueItems[0]
          ? `优先看看《${queueItems[0].title}》，决定是继续 Analyze 还是送入 Papers。`
          : "可以从主工作台运行 Search 或 DailyPaper，先把候选池建立起来。",
      metaLeft: `${highPriorityQueue} 篇高优`,
      metaRight: queueItems[0] ? `最近 ${queueItems[0].time}` : "等待候选",
    },
  ]
}

function buildActionLanes(args: {
  tasks: PipelineTask[]
  deadlines: DeadlineRadarItem[]
  queueItems: QueuePreview[]
  usageSummary: LLMUsageSummary
  signalCount: number
  tracks: ResearchTrackSummary[]
}): { now: LaneItemData[]; later: LaneItemData[] } {
  const { tasks, deadlines, queueItems, usageSummary, signalCount, tracks } = args
  const failedTask = tasks.find((task) => task.status === "failed")
  const urgentDeadline = deadlines.find((deadline) => deadline.days_left <= 14)
  const highPriorityItem = queueItems.find((item) => item.priority === "high")

  const now: LaneItemData[] = []
  const later: LaneItemData[] = []

  if (failedTask) {
    now.push({
      title: `修复 ${failedTask.paper_title}`,
      copy: "后台抓取或预处理失败会直接影响下午的候选与 digest 质量。",
      metaLeft: failedTask.started_at,
      metaRight: "阻塞主流程",
      tone: "bad",
      href: "#workflow",
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
      href: "#workflow",
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

function getLaneTone(items: LaneItemData[]): Tone {
  if (items.some((item) => item.tone === "bad")) return "bad"
  if (items.some((item) => item.tone === "warn")) return "warn"
  if (items.some((item) => item.tone === "good")) return "good"
  return "info"
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

function FocusDeadlinesPanel({
  track,
  deadlines,
}: {
  track: ResearchTrackSummary | null
  deadlines: DeadlineRadarItem[]
}) {
  const tone: Tone =
    deadlines.length === 0
      ? track
        ? "good"
        : "info"
      : deadlines[0].days_left <= 14
        ? "warn"
        : "info"

  return (
    <section className="mt-5 border-t border-slate-100 pt-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">Focus Deadlines</p>
          <h4 className="mt-2 text-base font-semibold text-slate-900">当前焦点的 Top 3 DDL</h4>
          <p className="mt-1 text-sm leading-6 text-slate-600">
            {track
              ? "根据 Track 的 keywords、methods 和 venues 自动关联最近会议，让 deadline 直接进入当前工作面。"
              : "先设置 Focus Track，系统才会开始自动关联相关会议 deadline。"}
          </p>
        </div>
        <TonePill tone={tone}>
          {deadlines.length > 0 ? `${deadlines.length} 项` : track ? "暂无命中" : "等待焦点"}
        </TonePill>
      </div>

      {deadlines.length > 0 ? (
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {deadlines.map((deadline) => {
            const matchedTrack = deadline.matched_tracks.find((item) => item.track_id === track?.id)
            const matchedTerms = (
              matchedTrack?.matched_terms?.length
                ? matchedTrack.matched_terms
                : matchedTrack?.matched_keywords || []
            ).slice(0, 3)

            return (
              <Link
                key={`${deadline.name}-${deadline.deadline}`}
                href={deadline.url}
                target="_blank"
                rel="noreferrer"
                className="block rounded-xl bg-slate-50 px-4 py-3 transition-colors hover:bg-slate-100"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                    {deadline.ccf_level || deadline.field}
                  </span>
                  <span className="text-sm font-semibold text-slate-900">{deadline.days_left} 天</span>
                </div>
                <h5 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{deadline.name}</h5>
                <p className="mt-1 text-sm leading-6 text-slate-600">{deadline.field}</p>
                {matchedTerms.length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {matchedTerms.map((term) => (
                      <span
                        key={`${deadline.name}-${term}`}
                        className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600"
                      >
                        {term}
                      </span>
                    ))}
                  </div>
                ) : null}
                <div className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-indigo-600">
                  查看会议
                  <ArrowRight size={14} />
                </div>
              </Link>
            )
          })}
        </div>
      ) : (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50/70 p-4 text-sm leading-6 text-slate-600">
          {track
            ? `当前还没有和 ${track.name} 自动关联的近期开会 DDL。可以补充 venues 或 methods，让匹配更稳定。`
            : "先去 Research 设定 Focus Track，deadline 才会自动收敛到当前主题。"}
        </div>
      )}
    </section>
  )
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

function TonePill({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${TONE_PILL_CLASSES[tone]}`}
    >
      {children}
    </span>
  )
}

function FocusOverviewPanel({ items }: { items: BriefCardData[] }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">Overview</p>
          <h4 className="mt-2 text-base font-semibold text-slate-900">焦点摘要</h4>
          <p className="mt-1 text-sm leading-6 text-slate-600">把焦点、后台状态、deadline 和队列压成一个连续摘要，而不是四张分散卡片。</p>
        </div>
        <TonePill tone="info">{items.length} 项</TonePill>
      </div>

      <div className="mt-4 divide-y divide-slate-200">
        {items.map((item) => (
          <article key={item.label} className="py-4 first:pt-0 last:pb-0">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{item.label}</p>
                <TonePill tone={item.tone}>{item.pill}</TonePill>
              </div>
              <span className="text-xs text-slate-500">{item.metaRight}</span>
            </div>
            <h5 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h5>
            <p className="mt-1 text-sm leading-6 text-slate-600">{item.copy}</p>
            <div className="mt-3 text-xs text-slate-500">{item.metaLeft}</div>
          </article>
        ))}
      </div>
    </section>
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

function TodayRail({
  nowItems,
  laterItems,
  destinations,
  queueItems,
  highPriorityQueue,
}: {
  nowItems: LaneItemData[]
  laterItems: LaneItemData[]
  destinations: DestinationCardData[]
  queueItems: QueuePreview[]
  highPriorityQueue: number
}) {
  function renderLaneItem(item: LaneItemData, key: string) {
    const content = (
      <div className="rounded-xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100">
        <div className="flex items-center justify-between gap-3">
          <TonePill tone={item.tone}>
            {item.tone === "bad"
              ? "阻塞"
              : item.tone === "warn"
                ? "注意"
                : item.tone === "good"
                  ? "清爽"
                  : "信息"}
          </TonePill>
          <span className="text-xs text-slate-500">{item.metaRight}</span>
        </div>
        <h4 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h4>
        <p className="mt-1 text-sm leading-6 text-slate-600">{item.copy}</p>
        <div className="mt-3 flex items-center justify-between gap-3 text-xs text-slate-500">
          <span>{item.metaLeft}</span>
          {item.href ? (
            <span className="inline-flex items-center gap-1 font-semibold text-indigo-600">
              打开
              <ArrowRight size={13} />
            </span>
          ) : null}
        </div>
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
    <aside className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">Today</p>
        <h3 className="mt-2 text-xl font-bold tracking-tight text-slate-900">今日侧栏</h3>
        <p className="mt-1 text-sm leading-6 text-slate-600">把行动项、交接入口和队列收成一条连续的 rail，避免右侧变成卡片堆。</p>
      </div>

      <div className="mt-5 space-y-5">
        <section>
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-900">现在要处理</h4>
            <TonePill tone={getLaneTone(nowItems)}>{nowItems.length} 项</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {nowItems.map((item, index) => renderLaneItem(item, `now-${index}`))}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-900">稍后处理</h4>
            <TonePill tone={getLaneTone(laterItems)}>{laterItems.length} 项</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {laterItems.map((item, index) => renderLaneItem(item, `later-${index}`))}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-900">交接入口</h4>
            <span className="text-xs text-slate-500">{destinations.length} 个空间</span>
          </div>
          <div className="mt-3 space-y-2">
            {destinations.map((item) => {
              const Icon = item.icon

              return (
                <Link
                  key={item.title}
                  href={item.href}
                  className="flex items-start justify-between gap-3 rounded-xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100"
                >
                  <div className="flex items-start gap-3">
                    <span className="mt-0.5 flex size-8 items-center justify-center rounded-xl bg-white text-indigo-600 shadow-sm">
                      <Icon size={16} />
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-slate-900">{item.title}</p>
                      <p className="mt-1 text-sm leading-6 text-slate-600">{item.description}</p>
                    </div>
                  </div>
                  <span className="text-xs font-semibold text-slate-500">{item.metric}</span>
                </Link>
              )
            })}
          </div>
        </section>

        <section className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-900">今日队列</h4>
            <TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} 篇高优</TonePill>
          </div>
          <div className="mt-3 space-y-2">
            {queueItems.length > 0 ? (
              queueItems.map((item) => (
                <Link
                  key={item.id}
                  href={item.href}
                  className="block rounded-xl bg-slate-50 px-3.5 py-3 transition-colors hover:bg-slate-100"
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
              <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50/70 p-4 text-sm leading-6 text-slate-600">
                队列里还没有候选。先从主工作台运行 Search，把今天的候选池建立起来。
              </div>
            )}
          </div>
        </section>
      </div>
    </aside>
  )
}

export default async function DashboardPage({ searchParams }: DashboardPageProps) {
  const params = searchParams ? await searchParams : {}
  const queryValue = Array.isArray(params?.query) ? params.query[0] : params?.query
  const initialQueries = typeof queryValue === "string"
    ? queryValue
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
    : undefined

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
  const activeTrackFeed = activeTrackFeedResult.items || []
  const activeTrackFeedTotal = activeTrackFeedResult.total || 0
  const intelligenceCards = buildDashboardIntelligenceCards(intelligenceFeed.items)
  const focusDeadlines = activeTrack
    ? deadlines.filter((deadline) =>
        deadline.matched_tracks.some((matchedTrack) => matchedTrack.track_id === activeTrack.id),
      ).slice(0, 3)
    : []
  const queueItems = buildQueuePreviewItems(
    readingQueue,
    new Map(papers.map((paper) => [String(paper.id), paper])),
    activeTrack,
  )

  const focusKeywords = (activeTrack?.keywords || []).slice(0, 3)
  const highPriorityQueue = queueItems.filter((item) => item.priority === "high").length
  const failedTasks = tasks.filter((task) => task.status === "failed")
  const urgentDeadlines = deadlines.filter((deadline) => deadline.days_left <= 14)
  const signalCount = intelligenceCards.length
  const alertCount = failedTasks.length + urgentDeadlines.length + (highPriorityQueue > 0 ? 1 : 0)
  const greeting = getGreeting()
  const libraryCount = papers.length

  const focusSummary = activeTrack
    ? activeTrackFeedTotal > 0
      ? activeTrackFeed[0]?.paper.title
        ? `当前焦点 Track「${activeTrack.name}」最近捕获了 ${activeTrackFeedTotal} 条相关更新，最新命中是《${activeTrackFeed[0].paper.title}》。`
        : `当前焦点 Track「${activeTrack.name}」最近捕获了 ${activeTrackFeedTotal} 条相关更新。`
      : activeTrack.description || `当前焦点 Track「${activeTrack.name}」暂时没有新的 feed，可以直接从工作台继续 Search。`
    : "目前还没有活跃的 Focus Track。先去 Research 设定问题域，再回到首页继续推进今天的工作。"

  const briefCards = buildBriefCards({
    activeTrack,
    activeTrackFeed,
    activeTrackFeedTotal,
    tasks,
    deadlines,
    queueItems,
  })
  const lanes = buildActionLanes({
    tasks,
    deadlines,
    queueItems,
    usageSummary,
    signalCount,
    tracks,
  })
  const destinationCards: DestinationCardData[] = [
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
                {greeting}，把今天最重要的问题先推进到可决策状态。
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                首页收敛到焦点、主工作台、提醒和证据。深层研究、文献管理和配置继续留在各自页面。
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className="inline-flex min-h-10 items-center rounded-full border border-slate-200 bg-white px-4 text-xs font-medium text-slate-600">
                Workspace / default
              </span>
              <span className="inline-flex min-h-10 items-center rounded-full border border-slate-200 bg-white px-4 text-xs font-medium text-slate-600">
                {activeTrack ? activeTrack.name : "Focus 未设置"}
              </span>
              <span className="inline-flex min-h-10 items-center rounded-full border border-slate-200 bg-white px-4 text-xs font-medium text-slate-600">
                {alertCount} 项提醒
              </span>
            </div>
          </header>

          <section className="mt-4 grid items-start gap-4 xl:grid-cols-[minmax(0,1.35fr)_360px]" id="workflow">
            <div className="space-y-4">
              <article className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                  <div className="max-w-3xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">Workspace</p>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                      当前焦点工作台
                    </h2>
                    <p className="mt-3 text-sm leading-6 text-slate-600">
                      把焦点摘要、运行概览和执行入口收进同一块区域，减少上下跳转和空白断层。
                    </p>
                  </div>

                  <Link
                    href={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
                    className="inline-flex items-center gap-1.5 text-sm font-semibold text-indigo-600 transition-colors hover:text-indigo-700"
                  >
                    打开完整 Research
                    <ArrowRight size={15} />
                  </Link>
                </div>

                <div className="mt-5 grid items-start gap-5 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
                  <div>
                    <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
                      <Sparkles size={14} />
                      当前焦点
                    </div>
                    <h3 className="mt-4 text-2xl font-bold tracking-tight text-slate-900">
                      {activeTrack?.name || "等待设置当前焦点"}
                    </h3>
                    <p className="mt-3 text-sm leading-6 text-slate-600">{focusSummary}</p>

                    <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50/70">
                      <div className="grid sm:grid-cols-2 xl:grid-cols-4">
                        <div className="border-b border-slate-200 px-4 py-3 sm:border-r xl:border-b-0">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Track 更新</p>
                          <p className="mt-2 text-lg font-semibold text-slate-900">{activeTrackFeedTotal}</p>
                        </div>
                        <div className="border-b border-slate-200 px-4 py-3 xl:border-b-0 xl:border-r">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">候选队列</p>
                          <p className="mt-2 text-lg font-semibold text-slate-900">{readingQueue.length}</p>
                        </div>
                        <div className="border-b border-slate-200 px-4 py-3 sm:border-r sm:border-b-0">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">提醒</p>
                          <p className="mt-2 text-lg font-semibold text-slate-900">{alertCount}</p>
                        </div>
                        <div className="px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">证据信号</p>
                          <p className="mt-2 text-lg font-semibold text-slate-900">{signalCount}</p>
                        </div>
                      </div>
                    </div>

                    <div className="mt-5 flex flex-wrap gap-2">
                      <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                        <Layers size={13} />
                        {activeTrack ? activeTrack.name : "Focus 未设置"}
                      </span>
                      <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                        <Activity size={13} />
                        {signalCount} 条证据已压缩
                      </span>
                      <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                        <Clock size={13} />
                        {highPriorityQueue} 篇高优候选
                      </span>
                    </div>

                    <div className="mt-4 flex flex-wrap gap-2">
                      {focusKeywords.length > 0 ? (
                        focusKeywords.map((keyword) => (
                          <span
                            key={keyword}
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600"
                          >
                            {keyword}
                          </span>
                        ))
                      ) : (
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
                          去 Research 添加关键词
                        </span>
                      )}
                      {activeTrackFeedTotal > 0 ? (
                        <span className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
                          {activeTrackFeedTotal} 条 Track feed
                        </span>
                      ) : null}
                    </div>

                    <FocusDeadlinesPanel track={activeTrack} deadlines={focusDeadlines} />

                    <div className="mt-6 flex flex-wrap gap-3">
                      <Link
                        href="#signals"
                        className="inline-flex min-h-11 items-center justify-center rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50"
                      >
                        查看证据快照
                      </Link>
                      <Link
                        href="/papers"
                        className="inline-flex min-h-11 items-center justify-center rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50"
                      >
                        打开 Papers
                      </Link>
                    </div>
                  </div>

                  <FocusOverviewPanel items={briefCards} />
                </div>
              </article>

              <TopicWorkflowDashboard
                initialQueries={initialQueries}
                compact
                dashboardContext={{
                  activeTrackName: activeTrack?.name ?? null,
                  activeTrackHref: activeTrack ? `/research?track_id=${activeTrack.id}` : "/research",
                  readingQueueCount: readingQueue.length,
                  urgentDeadlineCount: urgentDeadlines.length,
                  signalCount,
                }}
              />
            </div>

            <TodayRail
              nowItems={lanes.now}
              laterItems={lanes.later}
              destinations={destinationCards}
              queueItems={queueItems}
              highPriorityQueue={highPriorityQueue}
            />
          </section>

          <section className="mt-4" id="signals">
            <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="grid gap-4 xl:grid-cols-[220px_minmax(0,1fr)] xl:items-start">
                <div>
                  <SectionIntro
                    eyebrow="Signals"
                    title="证据快照"
                    copy="只保留会影响当前判断的几条信号，让这一段更像工作流后的连续证据带。"
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
                </div>

                <div className="grid auto-rows-fr gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {intelligenceCards.length > 0 ? (
                    intelligenceCards.map((item) => <EvidencePreviewCard key={item.id} item={item} compact />)
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/50 p-5 text-sm leading-6 text-slate-600 md:col-span-2 xl:col-span-3">
                      当前没有需要上浮到首页的社区信号。可以直接在主工作台继续推进当前问题。
                    </div>
                  )}
                </div>
              </div>
            </article>
          </section>
        </div>
      </main>
    </div>
  )
}
