"use client"

import { useMemo } from "react"
import { Loader2, CheckCircle2, XCircle, Circle, Wifi, WifiOff } from "lucide-react"
import { useAgentEventStore } from "@/lib/agent-events/store"
import type { AgentStatusEntry, AgentStatus } from "@/lib/agent-events/types"
import { getAgentPresentation } from "@/lib/agent-runtime"
import { buildSubagentActivityGroups } from "@/lib/agent-events/subagent-groups"

function statusConfig(status: AgentStatus) {
  switch (status) {
    case "working":
      return {
        icon: Loader2,
        label: "Working",
        colorClass: "text-amber-700",
        bgClass: "border-amber-200 bg-amber-50",
        spin: true,
      }
    case "completed":
      return {
        icon: CheckCircle2,
        label: "Completed",
        colorClass: "text-emerald-700",
        bgClass: "border-emerald-200 bg-emerald-50",
        spin: false,
      }
    case "errored":
      return {
        icon: XCircle,
        label: "Errored",
        colorClass: "text-rose-700",
        bgClass: "border-rose-200 bg-rose-50",
        spin: false,
      }
    default:
      return {
        icon: Circle,
        label: "Idle",
        colorClass: "text-zinc-500",
        bgClass: "border-zinc-200 bg-white",
        spin: false,
      }
  }
}

function AgentStatusBadge({
  entry,
  compact,
  workerRunId,
  onOpenWorkerRun,
}: {
  entry: AgentStatusEntry
  compact?: boolean
  workerRunId?: string | null
  onOpenWorkerRun: (workerRunId: string) => void
}) {
  const cfg = statusConfig(entry.status)
  const Icon = cfg.icon
  const presentation = getAgentPresentation(entry.agent_name)
  const fullLabel = presentation.label
  const displayName = compact
    ? presentation.shortLabel
    : fullLabel
  const interactive = presentation.kind === "subagent" && Boolean(workerRunId)

  if (compact) {
    return (
      <button
        type="button"
        className={`inline-flex min-w-0 items-center gap-1 rounded-full border px-1.5 py-0.5 text-left ${cfg.bgClass} ${
          interactive ? "transition-colors hover:bg-white/80" : ""
        }`}
        title={interactive ? `${entry.agent_name} · ${cfg.label} · Open worker details` : `${entry.agent_name} · ${cfg.label}`}
        onClick={() => {
          if (workerRunId) {
            onOpenWorkerRun(workerRunId)
          }
        }}
        disabled={!interactive}
      >
        <Icon
          size={11}
          className={`${cfg.colorClass} shrink-0 ${cfg.spin ? "animate-spin" : ""}`}
        />
        <span className="truncate text-[10px] font-medium text-zinc-900">{displayName}</span>
      </button>
    )
  }

  return (
    <button
      type="button"
      className={`flex min-w-0 items-center gap-2 rounded-lg border px-3 py-2 text-left ${cfg.bgClass} ${
        interactive ? "transition-colors hover:bg-white/80" : ""
      }`}
      title={interactive ? "Open worker details" : entry.agent_name}
      onClick={() => {
        if (workerRunId) {
          onOpenWorkerRun(workerRunId)
        }
      }}
      disabled={!interactive}
    >
      <Icon
        size={14}
        className={`${cfg.colorClass} shrink-0 ${cfg.spin ? "animate-spin" : ""}`}
      />
      <div className="min-w-0">
        <div className="truncate text-xs font-medium text-zinc-900">{displayName}</div>
        <div className={`text-xs ${cfg.colorClass}`}>{cfg.label}</div>
      </div>
    </button>
  )
}

export function AgentStatusPanel({ compact = false }: { compact?: boolean }) {
  const agentStatuses = useAgentEventStore((s) => s.agentStatuses)
  const connected = useAgentEventStore((s) => s.connected)
  const codexDelegations = useAgentEventStore((s) => s.codexDelegations)
  const toolCalls = useAgentEventStore((s) => s.toolCalls)
  const openWorkerRun = useAgentEventStore((s) => s.openWorkerRun)
  const entries = Object.values(agentStatuses)
  const latestWorkerRunByAssignee = useMemo(() => {
    const groups = buildSubagentActivityGroups(codexDelegations, toolCalls)
    return new Map(groups.map((group) => [group.assignee, group.workerRunId]))
  }, [codexDelegations, toolCalls])

  return (
    <div className={compact ? "flex flex-col gap-1" : "flex flex-col gap-2 p-3"}>
      {!compact && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-900">Agent Status</h3>
          <div className="flex items-center gap-1.5">
            {connected ? (
              <>
                <Wifi size={12} className="text-emerald-700" />
                <span className="text-xs text-emerald-700">Connected</span>
              </>
            ) : (
              <>
                <WifiOff size={12} className="text-amber-700" />
                <span className="text-xs text-amber-700">Connecting...</span>
              </>
            )}
          </div>
        </div>
      )}
      {entries.length === 0 ? (
        <div className="text-xs text-zinc-500">{compact ? "No agents" : "No agents active"}</div>
      ) : (
        <div className={compact ? "flex flex-wrap gap-1" : "grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4"}>
          {entries.map((entry) => (
            <AgentStatusBadge
              key={entry.agent_name}
              entry={entry}
              compact={compact}
              workerRunId={latestWorkerRunByAssignee.get(entry.agent_name) ?? null}
              onOpenWorkerRun={openWorkerRun}
            />
          ))}
        </div>
      )}
    </div>
  )
}
