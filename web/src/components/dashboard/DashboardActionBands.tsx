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

function SubsectionIntro({
  title,
  copy,
  meta,
}: {
  title: string
  copy: string
  meta: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div>
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        <p className="mt-1 text-sm leading-6 text-slate-600">{copy}</p>
      </div>
      <div>{meta}</div>
    </div>
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
    <section id="action-bands">
      <article className="rounded-[28px] border border-slate-200 bg-white/95 p-5 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Next Up</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">下一步只保留一个操作面板</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              首页不再拆成多条独立横栏，而是把“现在处理什么、候选队列、去哪继续做”压成一个统一模块。这样视线只需要顺着页面往下读，不需要在多个次级区块里来回切换。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <TonePill tone={getLaneTone(nowItems)}>{nowItems.length} 项当前动作</TonePill>
            <TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} 篇高优候选</TonePill>
            <TonePill tone="info">{destinations.length} 个工作台入口</TonePill>
          </div>
        </div>

        <div className="mt-5 space-y-5">
          <section>
            <SubsectionIntro
              title="先做这些"
              copy="主决策只保留最该先推进的动作卡片，避免提醒和导航入口混在一起。"
              meta={<TonePill tone={getLaneTone(nowItems)}>{nowItems.length} 项</TonePill>}
            />
            <div className="mt-3 grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {nowItems.map((item, index) => (
                <ActionCard key={`now-${index}`} item={item} />
              ))}
            </div>
          </section>

          <section className="border-t border-slate-100 pt-5">
            <SubsectionIntro
              title="顺手留意"
              copy="次级事项不再展开成第二个大模块，只保留成轻量卡片，提醒你稍后回看。"
              meta={<TonePill tone={getLaneTone(laterItems)}>{laterItems.length} 项</TonePill>}
            />
            <div className="mt-3 grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {laterItems.map((item, index) => (
                <ActionCard key={`later-${index}`} item={item} />
              ))}
            </div>
          </section>

          <section className="border-t border-slate-100 pt-5">
            <div className="grid gap-5 xl:grid-cols-2">
              <div>
                <SubsectionIntro
                  title="今天的候选队列"
                  copy="只保留最值得今天判断的几篇论文，减少“列表很多但不知道从哪篇开始”的负担。"
                  meta={<TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} 篇高优</TonePill>}
                />
                <div className="mt-3 grid gap-3">
                  {queueItems.length > 0 ? (
                    queueItems.map((item) => <QueueCard key={item.id} item={item} />)
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-5 text-sm leading-6 text-slate-600">
                      队列里还没有候选。先从 Workflows 或 Research 触发一轮搜索，把今天需要决策的论文拉起来。
                    </div>
                  )}
                </div>
              </div>

              <div>
                <SubsectionIntro
                  title="继续深入的入口"
                  copy="完整工作台仍然在各自页面里，但这里只保留最常用的下一跳。"
                  meta={<TonePill tone="info">{destinations.length} 个入口</TonePill>}
                />
                <div className="mt-3 grid gap-3">
                  {destinations.map((item) => (
                    <DestinationCardItem key={item.title} item={item} />
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>
      </article>
    </section>
  )
}
