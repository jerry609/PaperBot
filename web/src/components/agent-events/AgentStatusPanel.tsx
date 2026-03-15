"use client"

import { Loader2, CheckCircle2, XCircle, Circle, Wifi, WifiOff } from "lucide-react"
import { useAgentEventStore } from "@/lib/agent-events/store"
import type { AgentStatusEntry, AgentStatus } from "@/lib/agent-events/types"

function statusConfig(status: AgentStatus) {
  switch (status) {
    case "working":
      return {
        icon: Loader2,
        label: "Working",
        colorClass: "text-amber-400",
        bgClass: "bg-amber-950/40 border-amber-800/50",
        spin: true,
      }
    case "completed":
      return {
        icon: CheckCircle2,
        label: "Completed",
        colorClass: "text-green-400",
        bgClass: "bg-green-950/40 border-green-800/50",
        spin: false,
      }
    case "errored":
      return {
        icon: XCircle,
        label: "Errored",
        colorClass: "text-red-400",
        bgClass: "bg-red-950/40 border-red-800/50",
        spin: false,
      }
    default:
      return {
        icon: Circle,
        label: "Idle",
        colorClass: "text-gray-400",
        bgClass: "bg-gray-800/40 border-gray-700/50",
        spin: false,
      }
  }
}

function AgentStatusBadge({ entry, compact }: { entry: AgentStatusEntry; compact?: boolean }) {
  const cfg = statusConfig(entry.status)
  const Icon = cfg.icon
  const displayName = compact
    ? entry.agent_name.slice(0, 12) + (entry.agent_name.length > 12 ? "…" : "")
    : entry.agent_name

  return (
    <div
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${cfg.bgClass} min-w-0`}
    >
      <Icon
        size={14}
        className={`${cfg.colorClass} shrink-0 ${cfg.spin ? "animate-spin" : ""}`}
      />
      <div className="min-w-0">
        <div className="text-xs font-medium text-gray-200 truncate">{displayName}</div>
        <div className={`text-xs ${cfg.colorClass}`}>{cfg.label}</div>
      </div>
    </div>
  )
}

export function AgentStatusPanel({ compact = false }: { compact?: boolean }) {
  const agentStatuses = useAgentEventStore((s) => s.agentStatuses)
  const connected = useAgentEventStore((s) => s.connected)
  const entries = Object.values(agentStatuses)

  return (
    <div className="flex flex-col gap-2 p-3">
      {!compact && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Agent Status</h3>
          <div className="flex items-center gap-1.5">
            {connected ? (
              <>
                <Wifi size={12} className="text-green-400" />
                <span className="text-xs text-green-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff size={12} className="text-amber-400" />
                <span className="text-xs text-amber-400">Connecting...</span>
              </>
            )}
          </div>
        </div>
      )}
      {entries.length === 0 ? (
        <div className="text-xs text-gray-500">No agents active</div>
      ) : (
        <div className={compact ? "grid grid-cols-1 gap-2" : "grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4"}>
          {entries.map((entry) => (
            <AgentStatusBadge key={entry.agent_name} entry={entry} compact={compact} />
          ))}
        </div>
      )}
    </div>
  )
}
