"use client"

import { useMemo } from "react"
import { Cpu, FileCheck2, Loader2, Terminal, TriangleAlert } from "lucide-react"

import { ScrollArea } from "@/components/ui/scroll-area"
import { getAgentPresentation } from "@/lib/agent-runtime"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { buildSubagentActivityGroups, type SubagentActivityGroup } from "@/lib/agent-events/subagent-groups"

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toTimeString().slice(0, 8)
  } catch {
    return ts.slice(0, 8)
  }
}

function formatDuration(startedAt: string, finishedAt: string | null): string | null {
  const start = new Date(startedAt).getTime()
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now()
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null

  const totalSeconds = Math.round((end - start) / 1000)
  if (totalSeconds < 1) return "<1s"
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}m ${seconds}s`
}

function truncate(text: string, max = 84): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 1)}…`
}

function summarizeToolDetail(group: SubagentActivityGroup, toolIndex: number): string {
  const entry = group.recentTools[toolIndex]
  if (!entry) return ""
  if (entry.error) return entry.error
  if (entry.result_summary.trim()) return entry.result_summary.trim()
  const argKeys = Object.keys(entry.arguments)
  if (argKeys.length > 0) return `args: ${argKeys.join(", ")}`
  return "No summary"
}

function statusAppearance(status: SubagentActivityGroup["status"]) {
  if (status === "failed") {
    return {
      label: "Failed",
      icon: TriangleAlert,
      dotClass: "bg-rose-500",
      chipClass: "border-rose-200 bg-rose-50 text-rose-700",
      iconClass: "text-rose-600",
    }
  }
  if (status === "completed") {
    return {
      label: "Completed",
      icon: FileCheck2,
      dotClass: "bg-emerald-500",
      chipClass: "border-emerald-200 bg-emerald-50 text-emerald-700",
      iconClass: "text-emerald-600",
    }
  }
  if (status === "running") {
    return {
      label: "Running",
      icon: Loader2,
      dotClass: "bg-sky-500",
      chipClass: "border-sky-200 bg-sky-50 text-sky-700",
      iconClass: "text-sky-600",
    }
  }
  return {
    label: "Queued",
    icon: Cpu,
    dotClass: "bg-amber-500",
    chipClass: "border-amber-200 bg-amber-50 text-amber-700",
    iconClass: "text-amber-600",
  }
}

function SubagentCard({ group }: { group: SubagentActivityGroup }) {
  const presentation = getAgentPresentation(group.assignee)
  const appearance = statusAppearance(group.status)
  const StatusIcon = appearance.icon
  const fileCount = group.filesGenerated.length
  const duration = formatDuration(group.startedAt, group.finishedAt)

  return (
    <li className="rounded-[22px] border border-slate-200 bg-white px-3 py-3 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-[#f5f6f1]">
          <StatusIcon className={`h-4 w-4 ${appearance.iconClass} ${group.status === "running" ? "animate-spin" : ""}`} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-semibold text-slate-900">{presentation.label}</span>
            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${appearance.chipClass}`}>
              {appearance.label}
            </span>
            <span className="text-[11px] text-slate-400">{formatTimestamp(group.updatedAt)}</span>
          </div>

          <p className="mt-1 text-[12px] font-medium text-slate-700" title={group.taskTitle}>
            {group.taskTitle || "Untitled task"}
          </p>

          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5">
              {group.toolCount} tools
            </span>
            {group.toolErrorCount > 0 ? (
              <span className="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5 text-rose-700">
                {group.toolErrorCount} errors
              </span>
            ) : null}
            {fileCount > 0 ? (
              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-emerald-700">
                {fileCount} files
              </span>
            ) : null}
            {duration ? (
              <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5">
                {duration}
              </span>
            ) : null}
          </div>

          {group.recentTools.length > 0 ? (
            <div className="mt-2 rounded-2xl border border-slate-200 bg-[#f8faf6] px-2.5 py-2">
              <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                <Terminal className="h-3 w-3" />
                Recent tools
              </div>
              <div className="space-y-1.5">
                {group.recentTools.map((tool, index) => (
                  <div key={tool.id} className="rounded-xl border border-slate-200 bg-white px-2 py-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${tool.status === "error" ? "bg-rose-500" : appearance.dotClass}`} />
                      <span className="text-[11px] font-medium text-slate-800">{tool.tool}</span>
                      <span className="ml-auto text-[10px] text-slate-400">{formatTimestamp(tool.ts)}</span>
                    </div>
                    <p className="mt-1 text-[11px] leading-4 text-slate-500">
                      {truncate(summarizeToolDetail(group, index))}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="mt-2 rounded-2xl border border-dashed border-slate-200 bg-[#fafaf8] px-2.5 py-2 text-[11px] text-slate-500">
              {group.status === "queued"
                ? "Waiting for the first worker action."
                : "No nested worker tool activity was captured for this run."}
            </div>
          )}

          {group.error ? (
            <div className="mt-2 rounded-2xl border border-rose-200 bg-rose-50 px-2.5 py-2 text-[11px] leading-4 text-rose-700">
              {truncate(group.error, 220)}
              {group.reasonCode ? ` (${group.reasonCode})` : ""}
            </div>
          ) : null}
        </div>
      </div>
    </li>
  )
}

export function SubagentActivityPanel() {
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
  const toolCalls = useAgentEventStore((state) => state.toolCalls)
  const groups = useMemo(
    () => buildSubagentActivityGroups(codexDelegations, toolCalls),
    [codexDelegations, toolCalls],
  )

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f5f5f3]">
      <div className="border-b border-zinc-200 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-zinc-900">Delegations</h3>
          <span className="text-xs text-zinc-500">{groups.length} runs</span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          One card per Claude-dispatched subagent. Nested worker tools stay attached to the run.
        </p>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {groups.length === 0 ? (
          <div className="flex h-24 items-center justify-center text-sm text-zinc-500">
            No delegation activity yet
          </div>
        ) : (
          <ul className="space-y-2 px-3 py-3">
            {groups.map((group) => (
              <SubagentCard key={group.id} group={group} />
            ))}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}
