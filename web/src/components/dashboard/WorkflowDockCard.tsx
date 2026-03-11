"use client"

import Link from "next/link"
import { useEffect, useMemo, useState } from "react"
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Clock3,
  Layers3,
  Mail,
  Megaphone,
  Search,
  Sparkles,
  Workflow,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useWorkflowStore, type DailyResult, type WorkflowPhase } from "@/lib/stores/workflow-store"

type WorkflowDockCardProps = {
  initialQueries?: string[]
  activeTrackName?: string | null
  activeTrackHref: string
  readingQueueCount: number
  urgentDeadlineCount: number
  signalCount: number
}

type StageTone = "idle" | "running" | "done"

const PHASE_COPY: Record<WorkflowPhase, string> = {
  idle: "未开始",
  searching: "检索中",
  searched: "候选已就绪",
  reporting: "生成中",
  reported: "报告已就绪",
  error: "需要处理",
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return "尚未运行"

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

function countJudgedPapers(dailyResult: DailyResult | null): number {
  const budgetCount = dailyResult?.report?.judge?.budget?.judged_items
  if (typeof budgetCount === "number") {
    return budgetCount
  }

  const seen = new Set<string>()
  for (const query of dailyResult?.report?.queries || []) {
    for (const item of query.top_items || []) {
      if (!item.judge) continue
      const key = `${item.title}::${item.url || ""}`
      seen.add(key)
    }
  }
  return seen.size
}

function getStageTone(isDone: boolean, isRunning: boolean): StageTone {
  if (isRunning) return "running"
  if (isDone) return "done"
  return "idle"
}

function stageClasses(tone: StageTone): string {
  switch (tone) {
    case "running":
      return "border-amber-200 bg-amber-50 text-amber-700"
    case "done":
      return "border-emerald-200 bg-emerald-50 text-emerald-700"
    default:
      return "border-slate-200 bg-slate-50 text-slate-600"
  }
}

function StageCard({
  title,
  detail,
  tone,
}: {
  title: string
  detail: string
  tone: StageTone
}) {
  const label = tone === "running" ? "运行中" : tone === "done" ? "已完成" : "待启动"

  return (
    <div className={`rounded-2xl border px-4 py-3 ${stageClasses(tone)}`}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold">{title}</p>
        <span className="text-[11px] font-semibold uppercase tracking-[0.14em]">{label}</span>
      </div>
      <p className="mt-2 text-xs leading-5">{detail}</p>
    </div>
  )
}

export default function WorkflowDockCard({
  initialQueries,
  activeTrackName,
  activeTrackHref,
  readingQueueCount,
  urgentDeadlineCount,
  signalCount,
}: WorkflowDockCardProps) {
  const store = useWorkflowStore()
  const { searchResult, dailyResult, phase, lastUpdated, notifyEmail, notifyEnabled, resendEnabled, config } =
    store

  const [subscriberCount, setSubscriberCount] = useState<number | null>(null)

  useEffect(() => {
    if (!resendEnabled) {
      return
    }

    let cancelled = false

    async function loadSubscribers() {
      try {
        const response = await fetch("/api/newsletter/subscribers")
        if (!response.ok) return
        const payload = (await response.json()) as { active?: number }
        if (!cancelled && typeof payload.active === "number") {
          setSubscriberCount(payload.active)
        }
      } catch {
        if (!cancelled) {
          setSubscriberCount(null)
        }
      }
    }

    void loadSubscribers()

    return () => {
      cancelled = true
    }
  }, [resendEnabled])

  const workflowHref = useMemo(() => {
    if (!initialQueries?.length) return "/workflows"
    const params = new URLSearchParams({ query: initialQueries.join(",") })
    return `/workflows?${params.toString()}`
  }, [initialQueries])

  const queryCount = initialQueries?.length || dailyResult?.report?.stats?.query_count || 0
  const candidateCount =
    searchResult?.summary?.unique_items ?? dailyResult?.report?.stats?.unique_items ?? 0
  const judgedCount = countJudgedPapers(dailyResult)
  const artifactCount = [dailyResult?.markdown_path, dailyResult?.json_path].filter(Boolean).length
  const hasSearchData = Boolean(searchResult?.items?.length || dailyResult?.report)
  const hasReportData = Boolean(dailyResult?.report)
  const hasAnalysisData = Boolean(judgedCount || dailyResult?.report?.llm_analysis?.daily_insight)
  const searchTone = getStageTone(hasSearchData, phase === "searching")
  const dailyTone = getStageTone(hasReportData, phase === "reporting" && !hasAnalysisData)
  const analyzeTone = getStageTone(hasAnalysisData, phase === "reporting" && hasReportData)
  const dispatchChannels = [
    notifyEnabled ? "Email override" : null,
    resendEnabled ? "Newsletter / Resend" : null,
  ].filter(Boolean) as string[]
  const visibleSubscriberCount = resendEnabled ? subscriberCount : null
  const nextStep =
    !hasSearchData
      ? "先去 Workflows 跑一次 Search，建立今天的候选池。"
      : !hasReportData
        ? "下一步生成 DailyPaper，把候选压缩成可交付的 digest。"
        : !hasAnalysisData
          ? "接着跑 Analyze，补齐 Judge 评分与 insight，再决定要不要推送。"
          : "结果已经成形，适合回到 Workflows 复核后直接交付。"

  return (
    <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="max-w-2xl">
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
            Workflow Snapshot
          </p>
          <h3 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
            把完整 Workbench 留给 /workflows，首页只保留一次运行的快照。
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            这里汇总当前 workflow 的阶段、交付模式和 daily dispatch 状态，让 Dashboard 继续做总览，而不是再嵌一个完整操作台。
          </p>
        </div>

        <Badge className="w-fit rounded-full bg-slate-900 px-3 py-1 text-xs font-medium text-white">
          {PHASE_COPY[phase]}
        </Badge>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-3">
        <StageCard
          title="Search"
          detail={`${queryCount} 个主题 · ${candidateCount} 个候选`}
          tone={searchTone}
        />
        <StageCard
          title="DailyPaper"
          detail={
            hasReportData
              ? `${artifactCount} 份产物${config.saveDaily ? "已落盘" : "以预览模式存在"}`
              : "等待 digest 生成"
          }
          tone={dailyTone}
        />
        <StageCard
          title="Analyze & Dispatch"
          detail={`${judgedCount} 篇已 Judge · ${dispatchChannels.length || 0} 个交付通道`}
          tone={analyzeTone}
        />
      </div>

      <div className="mt-5 grid gap-3 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
        <section className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                Run Snapshot
              </p>
              <h4 className="mt-2 text-base font-semibold text-slate-900">
                当前 workflow 状态
              </h4>
            </div>
            <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
              最近更新 {formatRelativeTime(lastUpdated)}
            </span>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                Focus Handoff
              </p>
              <p className="mt-2 text-sm font-semibold text-slate-900">
                {activeTrackName || "尚未绑定 Focus Track"}
              </p>
              <p className="mt-1 text-sm leading-6 text-slate-600">
                {readingQueueCount} 篇候选待处理，{urgentDeadlineCount} 个紧迫 deadline，{signalCount} 条信号已压缩进首页。
              </p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                Output Mode
              </p>
              <p className="mt-2 text-sm font-semibold text-slate-900">
                {config.saveDaily ? "持久化 DailyPaper 产物" : "仅生成临时预览"}
              </p>
              <p className="mt-1 text-sm leading-6 text-slate-600">
                输出目录 {config.outputDir}，当前已准备 {artifactCount} 份可复核产物。
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-dashed border-slate-300 bg-white/70 px-4 py-3">
            <p className="flex items-center gap-2 text-sm font-medium text-slate-900">
              <Sparkles className="h-4 w-4 text-indigo-600" />
              下一步
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{nextStep}</p>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                Daily Dispatch
              </p>
              <h4 className="mt-2 text-base font-semibold text-slate-900">
                Workflow 的交付层
              </h4>
            </div>
            <Badge
              variant="secondary"
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600"
            >
              {dispatchChannels.length ? `${dispatchChannels.length} 个通道已开启` : "仅手动复核"}
            </Badge>
          </div>

          <div className="mt-4 space-y-3">
            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
              <p className="flex items-center gap-2 text-sm font-medium text-slate-900">
                <Mail className="h-4 w-4 text-indigo-600" />
                Email Override
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {notifyEnabled
                  ? notifyEmail.trim()
                    ? notifyEmail.trim()
                    : "已启用邮件投递，但本次运行没有覆盖默认收件人。"
                  : "未启用单次 Email override。"}
              </p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
              <p className="flex items-center gap-2 text-sm font-medium text-slate-900">
                <Megaphone className="h-4 w-4 text-indigo-600" />
                Newsletter / Resend
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {resendEnabled
                  ? visibleSubscriberCount !== null
                    ? `会把 digest 广播给 ${visibleSubscriberCount} 个 active subscribers。`
                    : "已启用 newsletter 广播，订阅数会在进入 Workflows 后继续校验。"
                  : "当前没有打开面向订阅者的每日推送。"}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              {dispatchChannels.length ? (
                dispatchChannels.map((channel) => (
                  <span
                    key={channel}
                    className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
                  >
                    {channel}
                  </span>
                ))
              ) : (
                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
                  Delivery off
                </span>
              )}
              {config.enableJudge ? (
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                  Judge enabled
                </span>
              ) : null}
              {config.enableLLM ? (
                <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700">
                  LLM insight enabled
                </span>
              ) : null}
            </div>
          </div>
        </section>
      </div>

      <div className="mt-5 flex flex-wrap gap-3">
        <Button asChild className="rounded-full px-5">
          <Link href={workflowHref}>
            <Workflow className="h-4 w-4" />
            打开 Workflow Workbench
          </Link>
        </Button>
        <Button asChild variant="outline" className="rounded-full px-5">
          <Link href={activeTrackHref}>
            <Layers3 className="h-4 w-4" />
            回到 Focus Research
          </Link>
        </Button>
        <Button asChild variant="ghost" className="rounded-full px-4 text-slate-700">
          <Link href="/papers">
            <BookOpen className="h-4 w-4" />
            检查 Papers
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-3">
          <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
            <Search className="h-3.5 w-3.5" />
            Topic Presets
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            {initialQueries?.length
              ? `${initialQueries.slice(0, 3).join(" · ")}${initialQueries.length > 3 ? "…" : ""}`
              : "没有从首页 query 继承预设主题。"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-3">
          <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
            <Clock3 className="h-3.5 w-3.5" />
            Dispatch Rhythm
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            现在仍然是“在运行时决定是否推送”的模式。调度与每日广播留在完整 Workflows 中配置。
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-3">
          <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Why Here
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            首页只负责告诉你这次 run 到了哪一步，以及值不值得继续推进，不再承载完整配置台。
          </p>
        </div>
      </div>
    </article>
  )
}
