import Link from "next/link"
import type { ReactNode } from "react"
import { ArrowRight, type LucideIcon } from "lucide-react"

export type DashboardDecisionTone = "good" | "warn" | "bad" | "info"

export type DashboardLaneItem = {
  title: string
  copy: string
  metaLeft: string
  metaRight: string
  tone: DashboardDecisionTone
  href?: string
}

export type DashboardQueuePreview = {
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

const TONE_CLASSES: Record<DashboardDecisionTone, string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-700",
  warn: "border-amber-200 bg-amber-50 text-amber-700",
  bad: "border-rose-200 bg-rose-50 text-rose-700",
  info: "border-sky-200 bg-sky-50 text-sky-700",
}

function getLaneTone(items: DashboardLaneItem[]): DashboardDecisionTone {
  if (items.some((item) => item.tone === "bad")) return "bad"
  if (items.some((item) => item.tone === "warn")) return "warn"
  if (items.some((item) => item.tone === "good")) return "good"
  return "info"
}

function TonePill({
  tone,
  children,
}: {
  tone: DashboardDecisionTone
  children: ReactNode
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  )
}

function ActionCard({ item }: { item: DashboardLaneItem }) {
  const content = (
    <div className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm transition-colors hover:bg-slate-50">
      <div className="flex items-center justify-between gap-3">
        <TonePill tone={item.tone}>{item.metaRight}</TonePill>
        <span className="text-xs text-slate-500">{item.metaLeft}</span>
      </div>
      <h4 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h4>
      <p className="mt-1 text-sm leading-6 text-slate-600">{item.copy}</p>
      {item.href ? (
        <span className="mt-auto inline-flex items-center gap-1 pt-4 text-sm font-semibold text-slate-900">
          打开
          <ArrowRight size={15} />
        </span>
      ) : null}
    </div>
  )

  return item.href ? (
    <Link href={item.href} className="block h-full">
      {content}
    </Link>
  ) : (
    content
  )
}

function QueueCard({ item }: { item: DashboardQueuePreview }) {
  return (
    <Link
      href={item.href}
      className="block h-full rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm transition-colors hover:bg-slate-50"
    >
      <div className="flex items-center justify-between gap-3">
        <TonePill tone={item.priority === "high" ? "warn" : item.priority === "medium" ? "info" : "good"}>
          {item.priority === "high" ? "高优" : item.priority === "medium" ? "中优" : "低优"}
        </TonePill>
        <span className="text-xs text-slate-500">{item.time}</span>
      </div>
      <h4 className="mt-3 text-sm font-semibold leading-6 text-slate-900">{item.title}</h4>
      <p className="mt-1 text-sm leading-6 text-slate-600">{item.venue}</p>
      {item.tags.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">
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
    </Link>
  )
}

function DestinationCardItem({ item }: { item: DashboardDestinationCard }) {
  const Icon = item.icon

  return (
    <Link
      href={item.href}
      className="flex h-full items-start justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm transition-colors hover:bg-slate-50"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-9 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
          <Icon size={17} />
        </span>
        <div>
          <p className="text-sm font-semibold text-slate-900">{item.title}</p>
          <p className="mt-1 text-sm leading-6 text-slate-600">{item.description}</p>
        </div>
      </div>
      <span className="text-xs font-semibold text-slate-500">{item.metric}</span>
    </Link>
  )
}

function ActionBand({
  eyebrow,
  title,
  copy,
  countLabel,
  countTone,
  children,
}: {
  eyebrow: string
  title: string
  copy: string
  countLabel: string
  countTone: DashboardDecisionTone
  children: ReactNode
}) {
  return (
    <article className="rounded-[28px] border border-slate-200 bg-white/95 p-5 shadow-sm">
      <div className="grid gap-4 xl:grid-cols-[240px_minmax(0,1fr)] xl:items-start">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{eyebrow}</p>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <h3 className="text-xl font-bold tracking-tight text-slate-900">{title}</h3>
            <TonePill tone={countTone}>{countLabel}</TonePill>
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">{copy}</p>
        </div>

        <div>{children}</div>
      </div>
    </article>
  )
}

export default function DashboardActionBands({
  nowItems,
  laterItems,
  destinations,
  queueItems,
  highPriorityQueue,
}: {
  nowItems: DashboardLaneItem[]
  laterItems: DashboardLaneItem[]
  destinations: DashboardDestinationCard[]
  queueItems: DashboardQueuePreview[]
  highPriorityQueue: number
}) {
  return (
    <section className="space-y-3" id="action-bands">
      <ActionBand
        eyebrow="Action Strip"
        title="现在先推进什么"
        copy="不再把提醒塞进一根右侧 rail，而是摊平成同一层级的动作卡片，让首页先回答“下一步做什么”。"
        countLabel={`${nowItems.length} 项`}
        countTone={getLaneTone(nowItems)}
      >
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {nowItems.map((item, index) => (
            <ActionCard key={`now-${index}`} item={item} />
          ))}
        </div>
      </ActionBand>

      <ActionBand
        eyebrow="Watch Strip"
        title="稍后再回看的事情"
        copy="把成本、次级信号和非焦点主题压成第二层提醒，避免它们和主决策抢同一视觉优先级。"
        countLabel={`${laterItems.length} 项`}
        countTone={getLaneTone(laterItems)}
      >
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {laterItems.map((item, index) => (
            <ActionCard key={`later-${index}`} item={item} />
          ))}
        </div>
      </ActionBand>

      <ActionBand
        eyebrow="Queue Strip"
        title="今天要判断的候选"
        copy="阅读队列继续存在，但只保留最需要你今天判断的几篇，不再占据单独侧栏。"
        countLabel={`${highPriorityQueue} 篇高优`}
        countTone={highPriorityQueue > 0 ? "warn" : "good"}
      >
        {queueItems.length > 0 ? (
          <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
            {queueItems.map((item) => (
              <QueueCard key={item.id} item={item} />
            ))}
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-5 text-sm leading-6 text-slate-600">
            队列里还没有候选。先从 Workflows 或 Research 触发一轮搜索，把今天需要决策的论文拉起来。
          </div>
        )}
      </ActionBand>

      <ActionBand
        eyebrow="Workspace Strip"
        title="下一跳去哪"
        copy="Research、Papers 和 Settings 继续是完整工作台，但在首页只保留扁平的入口，而不是再做一列导航摘要。"
        countLabel={`${destinations.length} 个空间`}
        countTone="info"
      >
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {destinations.map((item) => (
            <DestinationCardItem key={item.title} item={item} />
          ))}
        </div>
      </ActionBand>
    </section>
  )
}
