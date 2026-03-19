"use client"

import { type ComponentType, useMemo, useState } from "react"
import {
  Activity,
  ArrowLeft,
  ArrowUpRight,
  Ban,
  ChevronRight,
  Command,
  Cpu,
  FileCheck2,
  FileCode2,
  GitBranch,
  Loader2,
  MessageSquareText,
  PauseCircle,
  PlayCircle,
  Wrench,
  ShieldCheck,
  Terminal,
  TriangleAlert,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { backendUrl } from "@/lib/backend-url"
import { getAgentPresentation } from "@/lib/agent-runtime"
import { useAgentEventStore, type AgentInspectorView } from "@/lib/agent-events/store"
import {
  buildSubagentActivityGroups,
  type SubagentActivityGroup,
} from "@/lib/agent-events/subagent-groups"
import type { ActivityFeedItem, FileTouchedEntry, ToolCallEntry } from "@/lib/agent-events/types"
import { useStudioStore, type AgentAction, type AgentTask } from "@/lib/store/studio-store"
import {
  buildWorkerThreadPreview,
  findRelatedWorkerThread,
  type RelatedWorkerThread,
} from "@/lib/studio-worker-links"
import {
  activityFeedItemMatchesWorker,
  fileTouchedMatchesWorker,
  toolCallMatchesWorker,
} from "@/lib/agent-events/worker-focus"

type WorkerDisplayStatus = SubagentActivityGroup["status"] | "paused" | "cancelled"

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
  return `${text.slice(0, max - 1)}...`
}

function formatBridgeTaskKind(value: string): string {
  if (value === "approval_required") return "Approval"
  if (value === "code") return "Code"
  if (value === "review") return "Review"
  if (value === "research") return "Research"
  if (value === "plan") return "Plan"
  if (value === "ops") return "Ops"
  if (value === "failure") return "Failure"
  return "Result"
}

function formatBridgeStatus(value: string): string {
  if (value === "approval_required") return "Approval"
  if (value === "completed") return "Completed"
  if (value === "partial") return "Partial"
  if (value === "failed") return "Failed"
  return "Unknown"
}

function chatThreadStatusLabel(status: "running" | "completed" | "pending" | "error"): string {
  if (status === "running") return "Live"
  if (status === "completed") return "Done"
  if (status === "error") return "Error"
  return "Draft"
}

function formatCheckpointLabel(thread: RelatedWorkerThread, action: AgentAction): string {
  if (action.type === "approval_request") {
    const approval = action.metadata?.approvalRequest
    return approval?.message?.trim() || "Approval required"
  }
  if (action.type === "function_call") {
    const bridgeResult = action.metadata?.bridgeResult
    return bridgeResult?.summary?.trim() || "Worker result updated"
  }
  return thread.latestBridgeResult?.summary || "Worker checkpoint"
}

function checkpointKindLabel(action: AgentAction): string {
  if (action.type === "approval_request") return "Approval"
  if (action.type === "function_call") return "Result"
  return "Update"
}

function humanizeRuntime(runtime: string): string {
  const normalized = runtime.trim().toLowerCase()
  if (normalized === "codex") return "Codex"
  if (normalized === "opencode") return "OpenCode"
  if (normalized === "claude") return "Claude"
  if (normalized === "worker") return "Worker"
  return runtime || "Worker"
}

function controlModeChipLabel(controlMode: SubagentActivityGroup["controlMode"]): string {
  return controlMode === "managed" ? "studio-controlled" : "claude-controlled"
}

function controlOwnerLabel(controlMode: SubagentActivityGroup["controlMode"]): string {
  return controlMode === "managed" ? "Studio session" : "Claude session"
}

function controlSessionLabel(controlMode: SubagentActivityGroup["controlMode"]): string {
  return controlMode === "managed" ? "Managed session" : "Parent session"
}

function workerControlSummary(
  controlMode: SubagentActivityGroup["controlMode"],
  displayStatus: WorkerDisplayStatus,
  relatedThread: RelatedWorkerThread | null,
): string {
  if (controlMode === "managed") {
    if (displayStatus === "paused") return "Studio owns this worker session. Resume or cancel here."
    if (displayStatus === "cancelled") return "Studio-owned session cancelled."
    if (displayStatus === "completed") return "Studio-owned session completed. Full trace stays here."
    if (displayStatus === "failed") return "Studio-owned session failed. Controls stay here."
    return "Studio owns this worker session."
  }

  if (relatedThread?.pendingApproval) {
    return "Parent Claude session is waiting for approval."
  }
  if (displayStatus === "completed") {
    return "Completed in the parent Claude session."
  }
  if (displayStatus === "failed") {
    return "Failed in the parent Claude session."
  }
  return "Parent Claude session owns this worker."
}

function workerControlStateLabel(
  controlMode: SubagentActivityGroup["controlMode"],
  displayStatus: WorkerDisplayStatus,
  relatedThread: RelatedWorkerThread | null,
): string {
  if (controlMode === "managed") {
    if (displayStatus === "paused") return "Paused in Studio session"
    if (displayStatus === "cancelled") return "Cancelled in Studio session"
    if (displayStatus === "completed") return "Completed in Studio session"
    if (displayStatus === "failed") return "Failed in Studio session"
    if (displayStatus === "queued") return "Queued in Studio session"
    return "Running in Studio session"
  }

  if (relatedThread?.pendingApproval) return "Waiting for approval in parent session"
  if (displayStatus === "completed") return "Completed in parent session"
  if (displayStatus === "failed") return "Failed in parent session"
  if (displayStatus === "queued") return "Queued in parent session"
  return "Running in parent session"
}

function controlCardClassName(controlMode: SubagentActivityGroup["controlMode"]): string {
  return controlMode === "managed"
    ? "border-emerald-200 bg-[linear-gradient(180deg,#fcfdf9_0%,#f3f7ef_100%)]"
    : "border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#f2f5f8_100%)]"
}

function controlStateClassName(
  controlMode: SubagentActivityGroup["controlMode"],
  displayStatus: WorkerDisplayStatus,
  relatedThread: RelatedWorkerThread | null,
): string {
  if (controlMode === "mirrored" && relatedThread?.pendingApproval) {
    return "border-amber-200 bg-amber-50 text-amber-800"
  }
  if (displayStatus === "failed") return "border-rose-200 bg-rose-50 text-rose-700"
  if (displayStatus === "completed") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (displayStatus === "paused") return "border-amber-200 bg-amber-50 text-amber-700"
  if (displayStatus === "cancelled") return "border-slate-200 bg-slate-100 text-slate-600"
  return controlMode === "managed"
    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
    : "border-sky-200 bg-sky-50 text-sky-700"
}

function statusAppearance(status: WorkerDisplayStatus) {
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
  if (status === "paused") {
    return {
      label: "Paused",
      icon: PauseCircle,
      dotClass: "bg-amber-500",
      chipClass: "border-amber-200 bg-amber-50 text-amber-700",
      iconClass: "text-amber-600",
    }
  }
  if (status === "cancelled") {
    return {
      label: "Cancelled",
      icon: Ban,
      dotClass: "bg-slate-400",
      chipClass: "border-slate-200 bg-slate-100 text-slate-600",
      iconClass: "text-slate-500",
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

function summarizeToolDetail(entry: ToolCallEntry): string {
  if (entry.error) return entry.error
  if (entry.result_summary.trim()) return entry.result_summary.trim()
  const argKeys = Object.keys(entry.arguments)
  if (argKeys.length > 0) return `args: ${argKeys.join(", ")}`
  return "No summary"
}

function formatRecentToolStrip(entries: ToolCallEntry[]): string {
  if (entries.length === 0) return "No nested tool activity"

  const labels = entries
    .slice(0, 3)
    .map((entry) => entry.tool.trim())
    .filter((value) => value.length > 0)

  const unique = Array.from(new Set(labels))
  return unique.join(" · ")
}

function WorkerChip({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
      {label}
    </span>
  )
}

function WorkerFileRow({ entry }: { entry: FileTouchedEntry }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-mono text-[11px] text-slate-700" title={entry.path}>
          {entry.path}
        </span>
        <span className="shrink-0 text-[10px] text-slate-400">{formatTimestamp(entry.ts)}</span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
        <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5">
          {entry.status}
        </span>
        {typeof entry.linesAdded === "number" ? (
          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-emerald-700">
            +{entry.linesAdded}
          </span>
        ) : null}
        {typeof entry.linesDeleted === "number" ? (
          <span className="rounded-full border border-rose-200 bg-rose-50 px-1.5 py-0.5 text-rose-700">
            -{entry.linesDeleted}
          </span>
        ) : null}
      </div>
    </div>
  )
}

function WorkerToolRow({ entry, dotClass }: { entry: ToolCallEntry; dotClass: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-2.5 py-2">
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${entry.status === "error" ? "bg-rose-500" : dotClass}`} />
        <span className="text-[11px] font-medium text-slate-800">{entry.tool}</span>
        <span className="ml-auto text-[10px] text-slate-400">{formatTimestamp(entry.ts)}</span>
      </div>
      <p className="mt-1 text-[11px] leading-4 text-slate-500">{truncate(summarizeToolDetail(entry), 180)}</p>
    </div>
  )
}

function WorkerTranscriptRow({ item }: { item: ActivityFeedItem }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium text-slate-800">{item.type}</span>
        <span className="text-[10px] text-slate-400">{formatTimestamp(item.ts)}</span>
      </div>
      <p className="mt-1 text-[11px] leading-4 text-slate-500">{truncate(item.summary, 220)}</p>
    </div>
  )
}

function MonitorSurfaceButton({
  label,
  active,
  onClick,
  icon,
}: {
  label: string
  active: boolean
  onClick: () => void
  icon: ComponentType<{ className?: string }>
}) {
  const Icon = icon

  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[10px] transition-colors ${
        active
          ? "border-slate-900 bg-slate-900 text-white shadow-sm"
          : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900"
      }`}
    >
      <Icon className="h-3 w-3" />
      {label}
    </button>
  )
}

function WorkerListCard({
  group,
  displayStatus,
  onOpen,
}: {
  group: SubagentActivityGroup
  displayStatus: WorkerDisplayStatus
  onOpen: () => void
}) {
  const presentation = getAgentPresentation(group.assignee)
  const appearance = statusAppearance(displayStatus)
  const StatusIcon = appearance.icon
  const fileCount = group.filesGenerated.length
  const duration = formatDuration(group.startedAt, group.finishedAt)
  const recentToolStrip = formatRecentToolStrip(group.recentTools)

  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-start gap-3 rounded-[20px] border border-slate-200 bg-white px-3 py-2.5 text-left shadow-[0_1px_0_rgba(255,255,255,0.75)_inset] transition-colors hover:border-slate-300 hover:bg-[#fbfcf8]"
      >
        <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-[#f5f6f1]">
          <StatusIcon className={`h-4 w-4 ${appearance.iconClass} ${group.status === "running" ? "animate-spin" : ""}`} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-[12px] font-semibold text-slate-900">{presentation.label}</span>
            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${appearance.chipClass}`}>
              {appearance.label}
            </span>
            <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
              {controlModeChipLabel(group.controlMode)}
            </span>
            <span className="text-[10px] text-slate-400">{formatTimestamp(group.updatedAt)}</span>
          </div>

          <p className="mt-1 truncate text-[12px] font-medium text-slate-700" title={group.taskTitle}>
            {group.taskTitle || "Untitled task"}
          </p>

          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            <WorkerChip label={humanizeRuntime(group.runtime)} />
            <WorkerChip label={`${group.toolCount} tools`} />
            {fileCount > 0 ? <WorkerChip label={`${fileCount} files`} /> : null}
            {duration ? <WorkerChip label={duration} /> : null}
          </div>

          <div className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500">
            <Terminal className="h-3.5 w-3.5 shrink-0 text-slate-400" />
            <span className="truncate">
              {group.status === "queued" ? "Waiting for the first worker action." : recentToolStrip}
            </span>
          </div>

          {group.error ? (
            <div className="mt-1.5 rounded-xl border border-rose-200 bg-rose-50 px-2 py-1.5 text-[11px] leading-4 text-rose-700">
              {truncate(group.error, 180)}
              {group.reasonCode ? ` (${group.reasonCode})` : ""}
            </div>
          ) : null}

          <div className="mt-2 flex items-center justify-between gap-2 text-[10px] text-slate-400">
            <span className="truncate font-mono">{group.workerRunId}</span>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-600">
              Open
              <ChevronRight className="h-3.5 w-3.5" />
            </span>
          </div>
        </div>
      </button>
    </li>
  )
}

function WorkerDetail({
  group,
  displayStatus,
  matchedTask,
  onTaskStatusPatched,
  boardSessionId,
  relatedThread,
  onOpenRelatedThread,
  activeInspectorView,
  onOpenInspectorView,
  transcript,
  tools,
  files,
  onBack,
}: {
  group: SubagentActivityGroup
  displayStatus: WorkerDisplayStatus
  matchedTask: AgentTask | null
  onTaskStatusPatched: (status: AgentTask["status"], sessionId: string) => void
  boardSessionId: string | null
  relatedThread: RelatedWorkerThread | null
  onOpenRelatedThread: () => void
  activeInspectorView: AgentInspectorView
  onOpenInspectorView: (view: AgentInspectorView) => void
  transcript: ActivityFeedItem[]
  tools: ToolCallEntry[]
  files: FileTouchedEntry[]
  onBack: () => void
}) {
  const presentation = getAgentPresentation(group.assignee)
  const appearance = statusAppearance(displayStatus)
  const duration = formatDuration(group.startedAt, group.finishedAt)
  const [controlBusy, setControlBusy] = useState<"pause" | "resume" | "cancel" | null>(null)
  const [controlError, setControlError] = useState<string | null>(null)
  const [controlNotice, setControlNotice] = useState<string | null>(null)
  const [resumeCopied, setResumeCopied] = useState(false)
  const isManaged = group.controlMode === "managed" && Boolean(group.sessionId) && !group.sessionId.startsWith("studio-")
  const approvalRequest = relatedThread?.latestApprovalAction?.metadata?.approvalRequest
  const controlSessionId =
    approvalRequest?.cliSessionId?.trim() ||
    (group.sessionId.trim().length > 0 ? group.sessionId.trim() : "") ||
    null
  const controlResumeCommand = controlSessionId ? `claude --resume ${controlSessionId}` : null
  const controlOwner = controlOwnerLabel(group.controlMode)
  const controlState = workerControlStateLabel(group.controlMode, displayStatus, relatedThread)
  const controlSummary = workerControlSummary(group.controlMode, displayStatus, relatedThread)
  const controlCardTone = controlCardClassName(group.controlMode)
  const controlStateTone = controlStateClassName(group.controlMode, displayStatus, relatedThread)
  const relatedThreadSessionId = approvalRequest?.cliSessionId?.trim() || null

  async function performControl(action: "pause" | "resume" | "cancel") {
    if (!isManaged || !group.sessionId) return
    setControlBusy(action)
    setControlError(null)
    setControlNotice(null)
    try {
      const response = await fetch(
        backendUrl(`/api/agent-board/sessions/${group.sessionId}/${action}`),
        {
          method: "POST",
          cache: "no-store",
        },
      )
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Failed to ${action} session (${response.status})`)
      }

      if (matchedTask) {
        if (action === "pause") onTaskStatusPatched("paused", group.sessionId)
        if (action === "resume") onTaskStatusPatched("in_progress", group.sessionId)
        if (action === "cancel") onTaskStatusPatched("cancelled", group.sessionId)
      } else if (boardSessionId === group.sessionId) {
        if (action === "pause") onTaskStatusPatched("paused", group.sessionId)
        if (action === "resume") onTaskStatusPatched("in_progress", group.sessionId)
        if (action === "cancel") onTaskStatusPatched("cancelled", group.sessionId)
      }

      setControlNotice(
        action === "pause"
          ? "Managed session pause requested."
          : action === "resume"
            ? "Managed session resume requested."
            : "Managed session cancel requested.",
      )
    } catch (error) {
      setControlError(error instanceof Error ? error.message : `Failed to ${action} session`)
    } finally {
      setControlBusy(null)
    }
  }

  async function copyResumeCommand() {
    if (!controlResumeCommand) return
    setControlError(null)
    setControlNotice(null)
    try {
      await navigator.clipboard.writeText(controlResumeCommand)
      setResumeCopied(true)
      setControlNotice("Resume command copied.")
      window.setTimeout(() => setResumeCopied(false), 1800)
    } catch (error) {
      setControlError(error instanceof Error ? error.message : "Failed to copy resume command")
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f5f5f3]">
      <div className="border-b border-zinc-200 px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 rounded-full px-2 text-zinc-600"
            onClick={onBack}
          >
            <ArrowLeft className="mr-1 h-3.5 w-3.5" />
            Workers
          </Button>
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${appearance.chipClass}`}>
            {appearance.label}
          </span>
        </div>

        <div className="mt-2">
          <h3 className="text-sm font-semibold text-zinc-900">{presentation.label}</h3>
          <p className="mt-0.5 text-[12px] text-zinc-600">{group.taskTitle || "Untitled task"}</p>
        </div>

        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
          <WorkerChip label={humanizeRuntime(group.runtime)} />
          <WorkerChip label={controlModeChipLabel(group.controlMode)} />
          {duration ? <WorkerChip label={duration} /> : null}
          <WorkerChip label={formatTimestamp(group.startedAt)} />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-zinc-500">
          <div className="rounded-2xl border border-slate-200 bg-white px-2.5 py-2">
            <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">Worker run</div>
            <div className="mt-1 font-mono text-[11px] text-slate-700">{group.workerRunId}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white px-2.5 py-2">
            <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">{controlSessionLabel(group.controlMode)}</div>
            <div className="mt-1 font-mono text-[11px] text-slate-700">{group.sessionId || "Unavailable"}</div>
          </div>
        </div>

        <div className={`mt-3 rounded-[22px] border px-2.5 py-2.5 shadow-[0_1px_0_rgba(255,255,255,0.7)_inset] ${controlCardTone}`}>
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                Session Control
              </div>
              <div className="mt-1 flex items-center gap-2">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-white/80">
                  <ShieldCheck className={`h-4 w-4 ${group.controlMode === "managed" ? "text-emerald-700" : "text-slate-700"}`} />
                </div>
                <div className="min-w-0">
                  <div className="text-[12px] font-semibold text-slate-900">{controlOwner}</div>
                  <div className="text-[10px] text-slate-500">
                    {group.controlMode === "managed" ? "Controls execute directly from Studio." : "Actions route back to the parent Claude session."}
                  </div>
                </div>
              </div>
              <p className="mt-1 text-[11px] leading-4 text-slate-500">{controlSummary}</p>
            </div>
            <div className="flex flex-col items-end gap-1.5">
              <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${controlStateTone}`}>
                {controlState}
              </span>
              <WorkerChip label={controlModeChipLabel(group.controlMode)} />
            </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {group.sessionId ? <WorkerChip label={controlSessionLabel(group.controlMode)} /> : null}
            {group.sessionId ? <WorkerChip label={group.sessionId} /> : null}
            {approvalRequest?.workerAgentId ? <WorkerChip label={`worker ${approvalRequest.workerAgentId}`} /> : null}
            {controlResumeCommand ? <WorkerChip label="resume available" /> : null}
          </div>

          {controlResumeCommand ? (
            <div className="mt-2 rounded-2xl border border-white/80 bg-white/80 px-2.5 py-2">
              <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">Resume command</div>
              <code className="mt-1 block break-all text-[10px] text-slate-700">{controlResumeCommand}</code>
            </div>
          ) : null}

          <div className="mt-2 rounded-2xl border border-white/80 bg-white/80 px-2.5 py-2.5">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Actions</div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
            {isManaged ? (
              <>
                {displayStatus === "paused" ? (
                  <Button
                    type="button"
                    size="sm"
                    className="h-7 rounded-full bg-slate-900 px-2.5 text-[10px] text-white hover:bg-slate-800"
                    onClick={() => void performControl("resume")}
                    disabled={controlBusy !== null}
                  >
                    {controlBusy === "resume" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <PlayCircle className="mr-1.5 h-3.5 w-3.5" />}
                    Resume
                  </Button>
                ) : displayStatus !== "cancelled" && displayStatus !== "completed" && displayStatus !== "failed" ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 rounded-full border-amber-200 px-2.5 text-[10px] text-amber-800 hover:bg-amber-50"
                    onClick={() => void performControl("pause")}
                    disabled={controlBusy !== null}
                  >
                    {controlBusy === "pause" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <PauseCircle className="mr-1.5 h-3.5 w-3.5" />}
                    Pause
                  </Button>
                ) : null}

                {displayStatus !== "cancelled" && displayStatus !== "completed" && displayStatus !== "failed" ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 rounded-full border-rose-200 px-2.5 text-[10px] text-rose-700 hover:bg-rose-50"
                    onClick={() => void performControl("cancel")}
                    disabled={controlBusy !== null}
                  >
                    {controlBusy === "cancel" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Ban className="mr-1.5 h-3.5 w-3.5" />}
                    Cancel
                  </Button>
                ) : null}
              </>
            ) : (
              <>
                {relatedThread ? (
                  <Button
                    type="button"
                    size="sm"
                    className="h-7 rounded-full bg-slate-900 px-2.5 text-[10px] text-white hover:bg-slate-800"
                    onClick={onOpenRelatedThread}
                  >
                    <ArrowUpRight className="mr-1.5 h-3.5 w-3.5" />
                    Open thread
                  </Button>
                ) : null}
                {controlResumeCommand ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 rounded-full border-slate-200 px-2.5 text-[10px] text-slate-700"
                    onClick={() => void copyResumeCommand()}
                  >
                    <Command className="mr-1.5 h-3.5 w-3.5" />
                    {resumeCopied ? "Copied" : "Copy resume"}
                  </Button>
                ) : null}
                {!relatedThread && controlResumeCommand ? (
                  <span className="text-[10px] text-slate-500">
                    Resume from Claude CLI if the parent thread is not linked here.
                  </span>
                ) : null}
              </>
            )}
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <MonitorSurfaceButton
                label="Live"
                icon={Activity}
                active={activeInspectorView === "live"}
                onClick={() => onOpenInspectorView("live")}
              />
              <MonitorSurfaceButton
                label="Tools"
                icon={Wrench}
                active={activeInspectorView === "tools"}
                onClick={() => onOpenInspectorView("tools")}
              />
              <MonitorSurfaceButton
                label="Files"
                icon={FileCode2}
                active={activeInspectorView === "files"}
                onClick={() => onOpenInspectorView("files")}
              />
              <MonitorSurfaceButton
                label="Graph"
                icon={GitBranch}
                active={activeInspectorView === "graph"}
                onClick={() => onOpenInspectorView("graph")}
              />
            </div>
          </div>

          {group.controlMode === "mirrored" && !relatedThread && !controlResumeCommand ? (
            <div className="mt-2 rounded-2xl border border-slate-200 bg-[#fafaf8] px-2.5 py-2 text-[11px] leading-4 text-slate-500">
              Studio is mirroring this worker run, but the controlling Claude session is not linked in this workspace.
            </div>
          ) : null}

          {controlNotice ? (
            <p className="mt-2 text-[11px] text-emerald-700">{controlNotice}</p>
          ) : null}
          {controlError ? (
            <p className="mt-2 text-[11px] text-rose-700">{controlError}</p>
          ) : null}
        </div>

        {relatedThread ? (
          <div
            className={`mt-3 rounded-[22px] border px-2.5 py-2.5 shadow-[0_1px_0_rgba(255,255,255,0.7)_inset] ${
              relatedThread.pendingApproval
                ? "border-amber-200 bg-[linear-gradient(180deg,#fffdf8_0%,#f9f2dd_100%)]"
                : "border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#f5f7fa_100%)]"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                  {group.controlMode === "managed" ? "Linked Chat Thread" : "Parent Claude Session"}
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-2xl border border-white/80 bg-white/80">
                    <MessageSquareText className={`h-4 w-4 ${relatedThread.pendingApproval ? "text-amber-700" : "text-slate-700"}`} />
                  </div>
                  <div className="min-w-0">
                    <div className="truncate text-[12px] font-semibold text-slate-900">
                      {relatedThread.task.name}
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {group.controlMode === "managed"
                        ? "Worker status mirrors into the linked Studio thread."
                        : "This parent Claude session owns approval and continuation."}
                    </div>
                  </div>
                </div>
                <p className="mt-1.5 text-[11px] leading-4 text-slate-500">
                  {truncate(buildWorkerThreadPreview(relatedThread.task), 180)}
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className={`h-7 shrink-0 rounded-full px-2.5 text-[10px] ${
                  relatedThread.pendingApproval
                    ? "border-amber-200 bg-white text-amber-800 hover:bg-amber-50"
                    : "border-slate-200 bg-white text-slate-700"
                }`}
                onClick={onOpenRelatedThread}
              >
                <ArrowUpRight className="mr-1.5 h-3.5 w-3.5" />
                {relatedThread.pendingApproval ? "Open approval" : "Open thread"}
              </Button>
            </div>

            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <WorkerChip label={chatThreadStatusLabel(relatedThread.task.status)} />
              {relatedThread.latestBridgeResult ? (
                <WorkerChip label={formatBridgeTaskKind(relatedThread.latestBridgeResult.taskKind)} />
              ) : null}
              {relatedThread.latestBridgeResult ? (
                <WorkerChip label={formatBridgeStatus(relatedThread.latestBridgeResult.status)} />
              ) : null}
              {relatedThreadSessionId ? <WorkerChip label={relatedThreadSessionId} /> : null}
              {relatedThread.pendingApproval ? <WorkerChip label="approval pending" /> : null}
            </div>

            {relatedThread.pendingApproval && relatedThread.latestApprovalAction?.metadata?.approvalRequest?.command ? (
              <div className="mt-2 rounded-2xl border border-amber-200 bg-amber-50 px-2.5 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-amber-700">
                  Pending command
                </div>
                <code className="mt-1 block break-all text-[11px] text-amber-950">
                  {relatedThread.latestApprovalAction.metadata.approvalRequest.command}
                </code>
              </div>
            ) : null}

            {relatedThread.latestBridgeResult ? (
              <div className="mt-2 rounded-2xl border border-white/80 bg-white/80 px-2.5 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                  Latest handoff
                </div>
                <div className="text-[11px] leading-4 text-slate-700">
                  {relatedThread.latestBridgeResult.summary}
                </div>
                {relatedThread.latestBridgeResult.artifacts.length > 0 ? (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {relatedThread.latestBridgeResult.artifacts.slice(0, 4).map((artifact) => (
                      <WorkerChip key={`${artifact.kind}:${artifact.label}:${artifact.path ?? artifact.value ?? ""}`} label={artifact.label} />
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            {relatedThread.matchedActions.length > 1 ? (
              <div className="mt-2 rounded-2xl border border-white/80 bg-white/80 px-2.5 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                  Session timeline
                </div>
                <div className="mt-2 space-y-1.5">
                  {relatedThread.matchedActions.slice(0, 4).map((action) => (
                    <div
                      key={action.id}
                      className="flex items-start justify-between gap-2 rounded-xl border border-slate-200 bg-[#fcfcfb] px-2 py-1.5"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                            {checkpointKindLabel(action)}
                          </span>
                        </div>
                        <div className="mt-1 text-[11px] leading-4 text-slate-700">
                          {formatCheckpointLabel(relatedThread, action)}
                        </div>
                      </div>
                      <div className="shrink-0 text-[10px] text-slate-400">
                        {formatTimestamp(action.timestamp instanceof Date ? action.timestamp.toISOString() : String(action.timestamp))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {relatedThread.pendingApproval ? (
              <div className="mt-2 rounded-2xl border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] leading-4 text-amber-800">
                Approval is pending in the linked Claude thread.
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-3 px-3 py-3">
          <section className="rounded-[20px] border border-slate-200 bg-white px-2.5 py-2.5">
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              <MessageSquareText className="h-3.5 w-3.5" />
              Transcript
            </div>
            {transcript.length === 0 ? (
              <div className="text-[11px] text-slate-500">No worker lifecycle transcript captured yet.</div>
            ) : (
              <div className="space-y-1.5">
                {transcript.map((item) => (
                  <WorkerTranscriptRow key={item.id} item={item} />
                ))}
              </div>
            )}
          </section>

          <section className="rounded-[20px] border border-slate-200 bg-white px-2.5 py-2.5">
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              <Terminal className="h-3.5 w-3.5" />
              Tool activity
            </div>
            {tools.length === 0 ? (
              <div className="text-[11px] text-slate-500">No nested worker tools were captured.</div>
            ) : (
              <div className="space-y-1.5">
                {tools.map((tool) => (
                  <WorkerToolRow key={tool.id} entry={tool} dotClass={appearance.dotClass} />
                ))}
              </div>
            )}
          </section>

          <section className="rounded-[20px] border border-slate-200 bg-white px-2.5 py-2.5">
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              <FileCode2 className="h-3.5 w-3.5" />
              File changes
            </div>
            {files.length === 0 ? (
              <div className="text-[11px] text-slate-500">No file changes were attributed to this worker run.</div>
            ) : (
              <div className="space-y-1.5">
                {files.map((file) => (
                  <WorkerFileRow key={`${file.run_id}-${file.agent_name}-${file.path}`} entry={file} />
                ))}
              </div>
            )}
          </section>
        </div>
      </ScrollArea>
    </div>
  )
}

export function SubagentActivityPanel({
  onOpenRelatedThreadTask,
}: {
  onOpenRelatedThreadTask?: (taskId: string, paperId: string | null) => void
} = {}) {
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
  const toolCalls = useAgentEventStore((state) => state.toolCalls)
  const feed = useAgentEventStore((state) => state.feed)
  const filesTouched = useAgentEventStore((state) => state.filesTouched)
  const selectedWorkerRunId = useAgentEventStore((state) => state.selectedWorkerRunId)
  const setSelectedWorkerRunId = useAgentEventStore((state) => state.setSelectedWorkerRunId)
  const inspectorView = useAgentEventStore((state) => state.inspectorView)
  const setInspectorView = useAgentEventStore((state) => state.setInspectorView)
  const requestWorkspaceSurface = useAgentEventStore((state) => state.requestWorkspaceSurface)
  const chatTasks = useStudioStore((state) => state.tasks)
  const selectedPaperId = useStudioStore((state) => state.selectedPaperId)
  const setActiveTask = useStudioStore((state) => state.setActiveTask)
  const agentTasks = useStudioStore((state) => state.agentTasks)
  const boardSessionId = useStudioStore((state) => state.boardSessionId)
  const updateAgentTask = useStudioStore((state) => state.updateAgentTask)
  const setPipelinePhase = useStudioStore((state) => state.setPipelinePhase)
  const groups = useMemo(
    () => buildSubagentActivityGroups(codexDelegations, toolCalls),
    [codexDelegations, toolCalls],
  )

  const managedTaskStatusById = useMemo(
    () => new Map(agentTasks.map((task) => [task.id, task.status])),
    [agentTasks],
  )

  function resolveDisplayStatus(group: SubagentActivityGroup): WorkerDisplayStatus {
    const taskStatus = managedTaskStatusById.get(group.taskId)
    if (taskStatus === "paused") return "paused"
    if (taskStatus === "cancelled") return "cancelled"
    return group.status
  }

  const selectedGroup = useMemo(
    () =>
      selectedWorkerRunId
        ? groups.find((group) => group.workerRunId === selectedWorkerRunId) ?? null
        : null,
    [groups, selectedWorkerRunId],
  )
  const selectedTask = useMemo(
    () => (selectedGroup ? agentTasks.find((task) => task.id === selectedGroup.taskId) ?? null : null),
    [agentTasks, selectedGroup],
  )
  const selectedDisplayStatus = selectedGroup ? resolveDisplayStatus(selectedGroup) : null
  const relatedThread = useMemo(
    () =>
      selectedGroup
        ? findRelatedWorkerThread(chatTasks, {
            paperId: selectedPaperId,
            delegationTaskId: selectedGroup.taskId,
            workerRunId: selectedGroup.workerRunId,
          })
        : null,
    [chatTasks, selectedGroup, selectedPaperId],
  )

  const selectedTranscript = useMemo(() => {
    if (!selectedGroup) return []
    return [...feed]
      .filter((item) => activityFeedItemMatchesWorker(item, selectedGroup))
      .filter((item) => item.type !== "tool_call" && item.type !== "tool_result" && item.type !== "tool_error" && item.type !== "file_change")
      .reverse()
  }, [feed, selectedGroup])

  const selectedTools = useMemo(() => {
    if (!selectedGroup) return []
    return toolCalls.filter((entry) => toolCallMatchesWorker(entry, selectedGroup))
  }, [selectedGroup, toolCalls])

  const selectedFiles = useMemo(() => {
    if (!selectedGroup) return []
    return Object.values(filesTouched)
      .flat()
      .filter((entry) => fileTouchedMatchesWorker(entry, selectedGroup))
      .sort((left, right) => right.ts.localeCompare(left.ts))
  }, [filesTouched, selectedGroup])

  function patchManagedTaskStatus(status: AgentTask["status"], sessionId: string) {
    if (selectedGroup?.taskId) {
      updateAgentTask(selectedGroup.taskId, { status })
    }
    if (boardSessionId === sessionId) {
      if (status === "paused") setPipelinePhase("paused")
      else if (status === "cancelled") setPipelinePhase("cancelled")
      else setPipelinePhase("executing")
    }
  }

  function openRelatedThread() {
    if (!relatedThread) return
    setActiveTask(relatedThread.task.id)
    requestWorkspaceSurface("log")
    onOpenRelatedThreadTask?.(relatedThread.task.id, selectedPaperId)
  }

  if (selectedGroup) {
    return (
      <WorkerDetail
        group={selectedGroup}
        displayStatus={selectedDisplayStatus ?? selectedGroup.status}
        matchedTask={selectedTask}
        onTaskStatusPatched={patchManagedTaskStatus}
        boardSessionId={boardSessionId}
        relatedThread={relatedThread}
        onOpenRelatedThread={openRelatedThread}
        activeInspectorView={inspectorView}
        onOpenInspectorView={setInspectorView}
        transcript={selectedTranscript}
        tools={selectedTools}
        files={selectedFiles}
        onBack={() => setSelectedWorkerRunId(null)}
      />
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f5f5f3]">
      <div className="border-b border-zinc-200 px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-zinc-900">Workers</h3>
          <span className="text-[11px] text-zinc-500">{groups.length}</span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          Claude-dispatched worker runs.
        </p>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {groups.length === 0 ? (
          <div className="flex h-24 items-center justify-center text-sm text-zinc-500">
            No worker activity yet
          </div>
        ) : (
          <ul className="space-y-2 px-3 py-3">
            {groups.map((group) => (
              <WorkerListCard
                key={group.id}
                group={group}
                displayStatus={resolveDisplayStatus(group)}
                onOpen={() => setSelectedWorkerRunId(group.workerRunId)}
              />
            ))}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}
