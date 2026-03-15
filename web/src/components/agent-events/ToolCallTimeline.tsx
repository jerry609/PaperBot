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
    <li className="flex items-start gap-3 border-b border-zinc-200 py-2 last:border-0">
      <div
        className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${isError ? "bg-rose-500" : "bg-emerald-500"}`}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-semibold text-zinc-900">{entry.tool}</span>
          <div className="flex items-center gap-2 shrink-0">
            {isError && (
              <span className="rounded border border-rose-200 bg-rose-50 px-1.5 py-0.5 text-xs text-rose-700">
                error
              </span>
            )}
            <span className="font-mono text-xs text-zinc-500">{formatDuration(entry.duration_ms)}</span>
            <span className="font-mono text-xs text-zinc-400">{formatTimestamp(entry.ts)}</span>
          </div>
        </div>
        <div className="mt-0.5 text-xs text-zinc-500">
          <span className="text-zinc-400">args: </span>
          <span>{formatArgs(entry.arguments)}</span>
        </div>
        {entry.result_summary && !isError && (
          <div className="mt-0.5 truncate text-xs text-zinc-600">
            {truncate(entry.result_summary)}
          </div>
        )}
        {isError && entry.error && (
          <div className="mt-0.5 truncate text-xs text-rose-700">{entry.error}</div>
        )}
      </div>
    </li>
  )
}

export function ToolCallTimeline() {
  const toolCalls = useAgentEventStore((s) => s.toolCalls)

  return (
    <div className="flex h-full flex-col bg-[#f5f5f3]">
      <div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2">
        <h3 className="text-sm font-semibold text-zinc-900">Tool Calls</h3>
        <span className="text-xs text-zinc-500">{toolCalls.length} calls</span>
      </div>
      {toolCalls.length === 0 ? (
        <div className="flex h-20 items-center justify-center text-sm text-zinc-500">
          No tool calls yet
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
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
