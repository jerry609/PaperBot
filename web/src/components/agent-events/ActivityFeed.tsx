"use client"

import * as ScrollArea from "@radix-ui/react-scroll-area"
import { useAgentEventStore } from "@/lib/agent-events/store"
import type { ActivityFeedItem } from "@/lib/agent-events/types"
import { getAgentPresentation } from "@/lib/agent-runtime"

const HIDDEN_LIVE_EVENT_TYPES = new Set(["tool_call", "tool_result", "tool_error", "file_change"])

function getTypeColor(eventType: string): string {
  if (
    eventType === "agent_started" ||
    eventType === "agent_working" ||
    eventType === "agent_completed" ||
    eventType === "agent_error"
  ) {
    if (eventType === "agent_error") return "text-rose-700"
    if (eventType === "agent_completed") return "text-emerald-700"
    return "text-sky-700"
  }
  if (eventType === "tool_result") return "text-emerald-700"
  if (eventType === "tool_error") return "text-rose-700"
  return "text-zinc-500"
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toTimeString().slice(0, 8) // HH:MM:SS
  } catch {
    return ts.slice(0, 8)
  }
}

function getWorkerRunId(item: ActivityFeedItem): string | null {
  const payload = item.raw.payload
  if (!payload || typeof payload !== "object") return null
  const workerRunId = (payload as Record<string, unknown>).worker_run_id
  return typeof workerRunId === "string" && workerRunId.trim().length > 0 ? workerRunId : null
}

function ActivityFeedRow({
  item,
  onOpenWorkerRun,
}: {
  item: ActivityFeedItem
  onOpenWorkerRun: (workerRunId: string) => void
}) {
  const timeStr = formatTimestamp(item.ts)
  const colorClass = getTypeColor(item.type)
  const presentation = getAgentPresentation(item.agent_name)
  const workerRunId = getWorkerRunId(item)
  const interactive = workerRunId !== null

  return (
    <li className="border-b border-zinc-200 text-sm last:border-0">
      <button
        type="button"
        className={`flex w-full items-start gap-2 py-1.5 text-left ${
          interactive ? "rounded-md px-1 transition-colors hover:bg-white" : ""
        }`}
        onClick={() => {
          if (workerRunId) {
            onOpenWorkerRun(workerRunId)
          }
        }}
        disabled={!interactive}
        title={interactive ? "Open worker details" : item.agent_name}
      >
        <span className="mt-0.5 shrink-0 font-mono text-xs text-zinc-400">{timeStr}</span>
        <span
          className={`shrink-0 rounded border border-zinc-200 bg-zinc-100 px-1.5 py-0.5 text-xs font-medium ${colorClass}`}
          title={item.agent_name}
        >
          {presentation.shortLabel}
        </span>
        <span className="truncate text-zinc-700">{item.summary}</span>
      </button>
    </li>
  )
}

export function ActivityFeed() {
  const feed = useAgentEventStore((s) => s.feed)
  const openWorkerRun = useAgentEventStore((s) => s.openWorkerRun)
  const visibleFeed = feed.filter((item) => !HIDDEN_LIVE_EVENT_TYPES.has(item.type))

  return (
    <div className="flex h-full flex-col bg-[#f5f5f3]">
      <div className="border-b border-zinc-200 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-zinc-900">Live Activity</h3>
          <span className="text-xs text-zinc-500">{visibleFeed.length} events</span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          High-level lifecycle and delegation updates. Tool and file detail live in their own tabs.
        </p>
      </div>
      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport className="h-full w-full">
          {visibleFeed.length === 0 ? (
            <div className="flex h-20 items-center justify-center text-sm text-zinc-500">
              No high-level events yet
            </div>
          ) : (
            <ul className="px-2 py-1">
              {visibleFeed.map((item) => (
                <ActivityFeedRow
                  key={item.id}
                  item={item}
                  onOpenWorkerRun={openWorkerRun}
                />
              ))}
            </ul>
          )}
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="vertical" className="flex w-1.5 touch-none p-0.5">
          <ScrollArea.Thumb className="flex-1 rounded-full bg-zinc-300" />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </div>
  )
}
