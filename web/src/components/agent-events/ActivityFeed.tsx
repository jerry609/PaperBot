"use client"

import * as ScrollArea from "@radix-ui/react-scroll-area"
import { useAgentEventStore } from "@/lib/agent-events/store"
import type { ActivityFeedItem } from "@/lib/agent-events/types"

function getTypeColor(eventType: string): string {
  if (
    eventType === "agent_started" ||
    eventType === "agent_working" ||
    eventType === "agent_completed" ||
    eventType === "agent_error"
  ) {
    if (eventType === "agent_error") return "text-red-500"
    if (eventType === "agent_completed") return "text-green-600"
    return "text-blue-500"
  }
  if (eventType === "tool_result") return "text-green-600"
  if (eventType === "tool_error") return "text-red-500"
  return "text-gray-400"
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

  return (
    <li className="flex items-start gap-2 py-1 text-sm border-b border-gray-800 last:border-0">
      <span className="text-gray-500 font-mono text-xs shrink-0 mt-0.5">{timeStr}</span>
      <span
        className={`text-xs font-medium px-1.5 py-0.5 rounded bg-gray-800 shrink-0 ${colorClass}`}
      >
        {item.agent_name}
      </span>
      <span className="text-gray-300 truncate">{item.summary}</span>
    </li>
  )
}

export function ActivityFeed() {
  const feed = useAgentEventStore((s) => s.feed)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-200">Activity Feed</h3>
        <span className="text-xs text-gray-500">{feed.length} events</span>
      </div>
      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport className="h-full w-full">
          {feed.length === 0 ? (
            <div className="flex items-center justify-center h-20 text-gray-500 text-sm">
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
          <ScrollArea.Thumb className="flex-1 bg-gray-600 rounded-full" />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </div>
  )
}
