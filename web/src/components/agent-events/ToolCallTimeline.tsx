"use client"

import { useAgentEventStore } from "@/lib/agent-events/store"
import type { ToolCallEntry } from "@/lib/agent-events/types"

function formatDuration(ms: number): string {
  if (ms === 0) return "—"
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

function formatArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args)
  if (keys.length === 0) return "(no args)"
  return keys.join(", ")
}

function truncate(text: string, max = 100): string {
  if (text.length <= max) return text
  return text.slice(0, max) + "…"
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toTimeString().slice(0, 8)
  } catch {
    return ts.slice(0, 8)
  }
}

function ToolCallRow({ entry }: { entry: ToolCallEntry }) {
  const isError = entry.status === "error"

  return (
    <li className="flex items-start gap-3 py-2 border-b border-gray-800 last:border-0">
      <div
        className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${isError ? "bg-red-500" : "bg-green-500"}`}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-gray-100 truncate">{entry.tool}</span>
          <div className="flex items-center gap-2 shrink-0">
            {isError && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/50">
                error
              </span>
            )}
            <span className="text-xs text-gray-400 font-mono">{formatDuration(entry.duration_ms)}</span>
            <span className="text-xs text-gray-600 font-mono">{formatTimestamp(entry.ts)}</span>
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-0.5">
          <span className="text-gray-600">args: </span>
          <span>{formatArgs(entry.arguments)}</span>
        </div>
        {entry.result_summary && !isError && (
          <div className="text-xs text-gray-400 mt-0.5 truncate">
            {truncate(entry.result_summary)}
          </div>
        )}
        {isError && entry.error && (
          <div className="text-xs text-red-400 mt-0.5 truncate">{entry.error}</div>
        )}
      </div>
    </li>
  )
}

export function ToolCallTimeline() {
  const toolCalls = useAgentEventStore((s) => s.toolCalls)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-200">Tool Calls</h3>
        <span className="text-xs text-gray-500">{toolCalls.length} calls</span>
      </div>
      {toolCalls.length === 0 ? (
        <div className="flex items-center justify-center h-20 text-gray-500 text-sm">
          No tool calls yet
        </div>
      ) : (
        <div className="overflow-y-auto flex-1">
          <ul className="px-3 py-1">
            {toolCalls.map((entry) => (
              <ToolCallRow key={entry.id} entry={entry} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
