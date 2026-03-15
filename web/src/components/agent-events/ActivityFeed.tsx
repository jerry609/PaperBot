"use client"

import * as ScrollArea from "@radix-ui/react-scroll-area"
import { useAgentEventStore } from "@/lib/agent-events/store"
import type { ActivityFeedItem } from "@/lib/agent-events/types"
import { getAgentPresentation } from "@/lib/agent-runtime"

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

function ActivityFeedRow({ item }: { item: ActivityFeedItem }) {
  const timeStr = formatTimestamp(item.ts)
  const colorClass = getTypeColor(item.type)
  const presentation = getAgentPresentation(item.agent_name)

  return (
    <li className="flex items-start gap-2 border-b border-zinc-200 py-1.5 text-sm last:border-0">
      <span className="mt-0.5 shrink-0 font-mono text-xs text-zinc-400">{timeStr}</span>
      <span
        className={`shrink-0 rounded border border-zinc-200 bg-zinc-100 px-1.5 py-0.5 text-xs font-medium ${colorClass}`}
        title={item.agent_name}
      >
        {presentation.shortLabel}
      </span>
      <span className="truncate text-zinc-700">{item.summary}</span>
    </li>
  )
}

export function ActivityFeed() {
  const feed = useAgentEventStore((s) => s.feed)

  return (
    <div className="flex h-full flex-col bg-[#f5f5f3]">
      <div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2">
        <h3 className="text-sm font-semibold text-zinc-900">Activity Feed</h3>
        <span className="text-xs text-zinc-500">{feed.length} events</span>
      </div>
      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport className="h-full w-full">
          {feed.length === 0 ? (
            <div className="flex h-20 items-center justify-center text-sm text-zinc-500">
              No events yet
            </div>
          ) : (
            <ul className="px-2 py-1">
              {feed.map((item) => (
                <ActivityFeedRow key={item.id} item={item} />
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
