"use client"

import { Cpu, FileCheck2, Loader2, TriangleAlert } from "lucide-react"

import { ScrollArea } from "@/components/ui/scroll-area"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { getAgentPresentation } from "@/lib/agent-runtime"

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toTimeString().slice(0, 8)
  } catch {
    return ts.slice(0, 8)
  }
}

function eventAppearance(eventType: string) {
  if (eventType === "codex_failed") {
    return {
      label: "Failed",
      icon: TriangleAlert,
      dotClass: "bg-rose-500",
      textClass: "text-rose-700",
    }
  }
  if (eventType === "codex_completed") {
    return {
      label: "Completed",
      icon: FileCheck2,
      dotClass: "bg-emerald-500",
      textClass: "text-emerald-700",
    }
  }
  if (eventType === "codex_accepted") {
    return {
      label: "Accepted",
      icon: Cpu,
      dotClass: "bg-sky-500",
      textClass: "text-sky-700",
    }
  }
  return {
    label: "Dispatched",
    icon: Loader2,
    dotClass: "bg-amber-500",
    textClass: "text-amber-700",
  }
}

export function SubagentActivityPanel() {
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2">
        <h3 className="text-sm font-semibold text-zinc-900">Subagent Activity</h3>
        <span className="text-xs text-zinc-500">{codexDelegations.length} events</span>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {codexDelegations.length === 0 ? (
          <div className="flex h-20 items-center justify-center text-sm text-zinc-500">
            No subagent dispatches yet
          </div>
        ) : (
          <ul className="space-y-1 px-3 py-2">
            {codexDelegations.map((entry) => {
              const presentation = getAgentPresentation(entry.assignee)
              const appearance = eventAppearance(entry.event_type)
              const Icon = appearance.icon
              const generatedCount = Array.isArray(entry.files_generated) ? entry.files_generated.length : 0

              return (
                <li
                  key={entry.id}
                  className="rounded-lg border border-zinc-200 bg-white px-3 py-2"
                >
                  <div className="flex items-start gap-3">
                    <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${appearance.dotClass}`} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <Icon className={`h-3.5 w-3.5 shrink-0 ${appearance.textClass}`} />
                        <span className="truncate text-xs font-medium text-zinc-900">
                          {presentation.label}
                        </span>
                        <span className={`text-[11px] ${appearance.textClass}`}>
                          {appearance.label}
                        </span>
                      </div>
                      <p className="mt-1 truncate text-xs text-zinc-600" title={entry.task_title}>
                        {entry.task_title || "Untitled task"}
                      </p>
                      <div className="mt-1 flex items-center gap-2 text-[11px] text-zinc-500">
                        <span>{formatTimestamp(entry.ts)}</span>
                        {generatedCount > 0 ? <span>{generatedCount} files</span> : null}
                        {entry.reason_code ? <span>{entry.reason_code}</span> : null}
                      </div>
                    </div>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}
