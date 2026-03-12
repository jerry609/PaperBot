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
          Open
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
          {item.priority === "high" ? "High" : item.priority === "medium" ? "Medium" : "Low"}
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
  meta,
}: {
  title: string
  meta: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div className="min-h-6">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
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
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Next Up</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">What to do next</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <TonePill tone={getLaneTone(nowItems)}>{nowItems.length} actions</TonePill>
            <TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} priority papers</TonePill>
            <TonePill tone="info">{destinations.length} destinations</TonePill>
          </div>
        </div>

        <div className="mt-5 space-y-5">
          <section>
            <SubsectionIntro
              title="Now"
              meta={<TonePill tone={getLaneTone(nowItems)}>{nowItems.length}</TonePill>}
            />
            <div className="mt-3 grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {nowItems.map((item, index) => (
                <ActionCard key={`now-${index}`} item={item} />
              ))}
            </div>
          </section>

          <section className="border-t border-slate-100 pt-5">
            <SubsectionIntro
              title="Later"
              meta={<TonePill tone={getLaneTone(laterItems)}>{laterItems.length}</TonePill>}
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
                  title="Queue"
                  meta={<TonePill tone={highPriorityQueue > 0 ? "warn" : "good"}>{highPriorityQueue} high</TonePill>}
                />
                <div className="mt-3 grid gap-3">
                  {queueItems.length > 0 ? (
                    queueItems.map((item) => <QueueCard key={item.id} item={item} />)
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-5 text-sm leading-6 text-slate-600">
                      No queued papers yet.
                    </div>
                  )}
                </div>
              </div>

              <div>
                <SubsectionIntro
                  title="Open"
                  meta={<TonePill tone="info">{destinations.length}</TonePill>}
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
