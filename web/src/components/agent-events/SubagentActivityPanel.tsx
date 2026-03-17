"use client"

import { useMemo, useState } from "react"
import {
  ArrowLeft,
  Ban,
  Cpu,
  FileCheck2,
  FileCode2,
  Loader2,
  MessageSquareText,
  PauseCircle,
  PlayCircle,
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

function humanizeRuntime(runtime: string): string {
  const normalized = runtime.trim().toLowerCase()
  if (normalized === "codex") return "Codex"
  if (normalized === "opencode") return "OpenCode"
  if (normalized === "claude") return "Claude"
  if (normalized === "worker") return "Worker"
  return runtime || "Worker"
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

function WorkerChip({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
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
            <WorkerChip label={humanizeRuntime(group.runtime)} />
            <WorkerChip label={group.controlMode === "managed" ? "interruptible" : "view only"} />
            <WorkerChip label={`${group.toolCount} tools`} />
            {fileCount > 0 ? <WorkerChip label={`${fileCount} files`} /> : null}
            {duration ? <WorkerChip label={duration} /> : null}
          </div>

          {group.recentTools.length > 0 ? (
            <div className="mt-2 rounded-2xl border border-slate-200 bg-[#f8faf6] px-2.5 py-2">
              <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                <Terminal className="h-3 w-3" />
                Recent tools
              </div>
              <div className="space-y-1.5">
                {group.recentTools.map((tool) => (
                  <WorkerToolRow key={tool.id} entry={tool} dotClass={appearance.dotClass} />
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

          <div className="mt-3 flex items-center justify-between gap-2">
            <div className="min-w-0 text-[11px] text-slate-400">
              <span className="font-mono">{group.workerRunId}</span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
              onClick={onOpen}
            >
              Open
            </Button>
          </div>
        </div>
      </div>
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
  const isManaged = group.controlMode === "managed" && Boolean(group.sessionId) && !group.sessionId.startsWith("studio-")

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

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f5f5f3]">
      <div className="border-b border-zinc-200 px-3 py-2.5">
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
          <p className="mt-1 text-[12px] text-zinc-600">{group.taskTitle || "Untitled task"}</p>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <WorkerChip label={humanizeRuntime(group.runtime)} />
          <WorkerChip label={group.controlMode === "managed" ? "interruptible" : "view only"} />
          {duration ? <WorkerChip label={duration} /> : null}
          <WorkerChip label={formatTimestamp(group.startedAt)} />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-zinc-500">
          <div className="rounded-2xl border border-slate-200 bg-white px-2.5 py-2">
            <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">Worker run</div>
            <div className="mt-1 font-mono text-[11px] text-slate-700">{group.workerRunId}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white px-2.5 py-2">
            <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">Session</div>
            <div className="mt-1 font-mono text-[11px] text-slate-700">{group.sessionId}</div>
          </div>
        </div>

        <div className="mt-3 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-[11px] text-zinc-500">
          {isManaged
            ? "This worker run belongs to a managed PaperBot session. Controls below act on the whole managed session."
            : "This worker run is mirrored from Claude Code. You can inspect it, but Studio cannot interrupt it yet."}
        </div>

        <div className="mt-3 rounded-[22px] border border-slate-200 bg-white px-3 py-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            Enter Worker Monitor
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
              onClick={() => onOpenInspectorView("live")}
            >
              Live Feed
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
              onClick={() => onOpenInspectorView("tools")}
            >
              Tools
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
              onClick={() => onOpenInspectorView("files")}
            >
              Files
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
              onClick={() => onOpenInspectorView("graph")}
            >
              Graph
            </Button>
          </div>
        </div>

        {isManaged ? (
          <div className="mt-3 rounded-[22px] border border-slate-200 bg-white px-3 py-3">
            <div className="flex flex-wrap items-center gap-2">
              {displayStatus === "paused" ? (
                <Button
                  type="button"
                  size="sm"
                  className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
                  onClick={() => void performControl("resume")}
                  disabled={controlBusy !== null}
                >
                  {controlBusy === "resume" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <PlayCircle className="mr-1.5 h-3.5 w-3.5" />}
                  Resume Session
                </Button>
              ) : displayStatus !== "cancelled" && displayStatus !== "completed" && displayStatus !== "failed" ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-full border-amber-200 px-3 text-[11px] text-amber-800 hover:bg-amber-50"
                  onClick={() => void performControl("pause")}
                  disabled={controlBusy !== null}
                >
                  {controlBusy === "pause" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <PauseCircle className="mr-1.5 h-3.5 w-3.5" />}
                  Pause Session
                </Button>
              ) : null}

              {displayStatus !== "cancelled" && displayStatus !== "completed" && displayStatus !== "failed" ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-full border-rose-200 px-3 text-[11px] text-rose-700 hover:bg-rose-50"
                  onClick={() => void performControl("cancel")}
                  disabled={controlBusy !== null}
                >
                  {controlBusy === "cancel" ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Ban className="mr-1.5 h-3.5 w-3.5" />}
                  Cancel Session
                </Button>
              ) : null}
            </div>
            {controlNotice ? (
              <p className="mt-2 text-[11px] text-emerald-700">{controlNotice}</p>
            ) : null}
            {controlError ? (
              <p className="mt-2 text-[11px] text-rose-700">{controlError}</p>
            ) : null}
          </div>
        ) : null}

        {relatedThread ? (
          <div className="mt-3 rounded-[22px] border border-slate-200 bg-white px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                  Related Chat Thread
                </div>
                <div className="mt-1 truncate text-[12px] font-semibold text-slate-900">
                  {relatedThread.task.name}
                </div>
                <p className="mt-1 text-[11px] leading-4 text-slate-500">
                  {truncate(buildWorkerThreadPreview(relatedThread.task), 180)}
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 shrink-0 rounded-full border-slate-200 px-3 text-[11px] text-slate-700"
                onClick={onOpenRelatedThread}
              >
                {relatedThread.pendingApproval ? "Open Approval" : "Open Thread"}
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
              <div className="mt-2 rounded-2xl border border-slate-200 bg-[#f8faf6] px-2.5 py-2">
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
              <div className="mt-2 rounded-2xl border border-slate-200 bg-[#fafaf8] px-2.5 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                  Thread checkpoints
                </div>
                <div className="mt-2 space-y-1.5">
                  {relatedThread.matchedActions.slice(0, 4).map((action) => (
                    <div
                      key={action.id}
                      className="flex items-start justify-between gap-2 rounded-xl border border-slate-200 bg-white px-2 py-1.5"
                    >
                      <div className="min-w-0 text-[11px] leading-4 text-slate-700">
                        {formatCheckpointLabel(relatedThread, action)}
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
                Approval is pending in the linked Claude chat thread. Open the thread to approve and continue that worker run.
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-3 px-3 py-3">
          <section className="rounded-[22px] border border-slate-200 bg-white px-3 py-3">
            <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
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

          <section className="rounded-[22px] border border-slate-200 bg-white px-3 py-3">
            <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
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

          <section className="rounded-[22px] border border-slate-200 bg-white px-3 py-3">
            <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
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

export function SubagentActivityPanel() {
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
  const toolCalls = useAgentEventStore((state) => state.toolCalls)
  const feed = useAgentEventStore((state) => state.feed)
  const filesTouched = useAgentEventStore((state) => state.filesTouched)
  const selectedWorkerRunId = useAgentEventStore((state) => state.selectedWorkerRunId)
  const setSelectedWorkerRunId = useAgentEventStore((state) => state.setSelectedWorkerRunId)
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
      <div className="border-b border-zinc-200 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-zinc-900">Workers</h3>
          <span className="text-xs text-zinc-500">{groups.length} runs</span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          One card per Claude-dispatched worker run. Open a run to inspect its own transcript, tools, and files.
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
