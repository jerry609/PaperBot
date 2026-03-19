"use client"

import { useEffect, useMemo, useState, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowUpRight,
  Bot,
  FileCode2,
  MessageSquareText,
  ShieldCheck,
  Sparkles,
  Users2,
  Wrench,
  Workflow,
} from "lucide-react"

import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { AgentStatusPanel } from "@/components/agent-events/AgentStatusPanel"
import { SubagentActivityPanel } from "@/components/agent-events/SubagentActivityPanel"
import { ToolCallTimeline } from "@/components/agent-events/ToolCallTimeline"
import { AgentDagPanel } from "@/components/agent-dashboard/AgentDagPanel"
import { FileListPanel } from "@/components/agent-dashboard/FileListPanel"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { getAgentPresentation } from "@/lib/agent-runtime"
import {
  buildSubagentActivityGroups,
  type SubagentActivityGroup,
} from "@/lib/agent-events/subagent-groups"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { useStudioRuntime } from "@/hooks/useStudioRuntime"
import {
  buildWorkerThreadPreview,
  findRelatedWorkerThread,
  type RelatedWorkerThread,
} from "@/lib/studio-worker-links"
import { cn } from "@/lib/utils"
import { useStudioStore, type AgentTask } from "@/lib/store/studio-store"

type SwarmSection = "overview" | "staff" | "collaboration" | "tasks"
type ActivityPane = "live" | "tools" | "files"

function monitorRuntimeSummary(
  loading: boolean,
  refreshing: boolean,
  error: string | null,
  statusLabel: string,
): string {
  if (loading) return "Checking Claude Code..."
  if (error) return refreshing ? "Reconnecting to Studio backend..." : "Studio backend disconnected"
  return statusLabel
}

function formatWorkspaceLabel(path: string | null | undefined): string {
  if (!path) return "Workspace pending"
  const segments = path.split("/").filter(Boolean)
  if (segments.length === 0) return path
  return segments.slice(-2).join("/")
}

function workerStatusMeta(status: SubagentActivityGroup["status"]) {
  if (status === "failed") {
    return {
      label: "Failed",
      chipClass: "border-rose-200 bg-rose-50 text-rose-700",
    }
  }
  if (status === "completed") {
    return {
      label: "Completed",
      chipClass: "border-emerald-200 bg-emerald-50 text-emerald-700",
    }
  }
  if (status === "running") {
    return {
      label: "Running",
      chipClass: "border-sky-200 bg-sky-50 text-sky-700",
    }
  }
  return {
    label: "Queued",
    chipClass: "border-amber-200 bg-amber-50 text-amber-700",
  }
}

function taskStatusMeta(status: AgentTask["status"]) {
  if (status === "in_progress") return "border-sky-200 bg-sky-50 text-sky-700"
  if (status === "repairing") return "border-amber-200 bg-amber-50 text-amber-700"
  if (status === "human_review") return "border-violet-200 bg-violet-50 text-violet-700"
  if (status === "done") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (status === "paused") return "border-zinc-200 bg-zinc-100 text-zinc-600"
  if (status === "cancelled") return "border-zinc-200 bg-zinc-100 text-zinc-500"
  return "border-zinc-200 bg-white text-zinc-600"
}

function collaborationState(thread: RelatedWorkerThread | null, group: SubagentActivityGroup) {
  if (thread?.pendingApproval) {
    return {
      label: "Approval waiting",
      className: "border-amber-200 bg-amber-50 text-amber-800",
      summary: "Parent Claude session is waiting on approval before the worker can continue.",
    }
  }
  if (group.status === "failed") {
    return {
      label: "Needs repair",
      className: "border-rose-200 bg-rose-50 text-rose-700",
      summary: group.error || "This worker failed and needs inspection.",
    }
  }
  if (!thread && group.controlMode === "mirrored") {
    return {
      label: "Unlinked mirror",
      className: "border-slate-200 bg-slate-100 text-slate-700",
      summary: "Studio can see the worker run, but the parent Claude thread is not linked here.",
    }
  }
  if (group.status === "completed") {
    return {
      label: "Synced",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
      summary: thread ? "Worker and thread are aligned." : "Completed worker trace available in Monitor.",
    }
  }
  return {
    label: "Streaming",
    className: "border-sky-200 bg-sky-50 text-sky-700",
    summary: thread ? "Worker updates are flowing into the linked chat thread." : "Worker is active in Monitor.",
  }
}

function MetricCard({
  icon: Icon,
  label,
  value,
  caption,
}: {
  icon: typeof Activity
  label: string
  value: string
  caption: string
}) {
  return (
    <div className="rounded-[20px] border border-slate-200 bg-white px-3 py-3 shadow-sm">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-2xl border border-slate-200 bg-[#f3f5ef]">
          <Icon className="h-4 w-4 text-slate-600" />
        </div>
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            {label}
          </div>
          <div className="mt-0.5 text-[18px] font-semibold text-slate-900">{value}</div>
        </div>
      </div>
      <p className="mt-2 text-[11px] leading-5 text-slate-500">{caption}</p>
    </div>
  )
}

function SectionCard({
  title,
  eyebrow,
  description,
  children,
  action,
  className,
}: {
  title: string
  eyebrow: string
  description: string
  children: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <section className={cn("min-h-0 overflow-hidden rounded-[28px] border border-slate-200 bg-[#f7f7f4] shadow-[0_20px_56px_rgba(15,23,42,0.04)]", className)}>
      <div className="border-b border-slate-200 bg-[#f4f5f1] px-3 py-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {eyebrow}
            </p>
            <p className="mt-0.5 text-sm font-semibold text-slate-900">{title}</p>
            <p className="mt-0.5 text-[11px] text-slate-500">{description}</p>
          </div>
          {action}
        </div>
      </div>
      {children}
    </section>
  )
}

export function StudioMonitorWorkspace({
  initialPaperId = null,
  initialWorkerRunId = null,
}: {
  initialPaperId?: string | null
  initialWorkerRunId?: string | null
}) {
  useAgentEvents()

  const router = useRouter()
  const [activeSection, setActiveSection] = useState<SwarmSection>(() =>
    initialWorkerRunId ? "staff" : "overview",
  )
  const [activityPane, setActivityPane] = useState<ActivityPane>("live")

  const loadPapers = useStudioStore((state) => state.loadPapers)
  const papers = useStudioStore((state) => state.papers)
  const selectedPaperId = useStudioStore((state) => state.selectedPaperId)
  const selectPaper = useStudioStore((state) => state.selectPaper)
  const boardSessionId = useStudioStore((state) => state.boardSessionId)
  const chatTasks = useStudioStore((state) => state.tasks)
  const agentTasks = useStudioStore((state) => state.agentTasks)
  const setActiveTask = useStudioStore((state) => state.setActiveTask)

  const { info: runtimeInfo, loading: runtimeLoading, refreshing: runtimeRefreshing } = useStudioRuntime()
  const agentStatuses = useAgentEventStore((state) => state.agentStatuses)
  const toolCalls = useAgentEventStore((state) => state.toolCalls)
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
  const filesTouched = useAgentEventStore((state) => state.filesTouched)
  const openWorkerRun = useAgentEventStore((state) => state.openWorkerRun)
  const setSelectedWorkerRunId = useAgentEventStore((state) => state.setSelectedWorkerRunId)
  const requestWorkspaceSurface = useAgentEventStore((state) => state.requestWorkspaceSurface)

  useEffect(() => {
    loadPapers()
  }, [loadPapers])

  useEffect(() => {
    if (!initialPaperId) return
    if (selectedPaperId === initialPaperId) return
    selectPaper(initialPaperId)
  }, [initialPaperId, selectPaper, selectedPaperId])

  useEffect(() => {
    if (!initialWorkerRunId) return
    setSelectedWorkerRunId(initialWorkerRunId)
  }, [initialWorkerRunId, setSelectedWorkerRunId])

  const effectivePaperId = selectedPaperId || initialPaperId
  const selectedPaper = useMemo(
    () => (effectivePaperId ? papers.find((paper) => paper.id === effectivePaperId) ?? null : null),
    [effectivePaperId, papers],
  )
  const workerGroups = useMemo(
    () => buildSubagentActivityGroups(codexDelegations, toolCalls),
    [codexDelegations, toolCalls],
  )
  const paperWorkerGroups = useMemo(
    () =>
      effectivePaperId
        ? workerGroups.filter((group) =>
            agentTasks.some((task) => task.id === group.taskId && task.paperId === effectivePaperId),
          )
        : workerGroups,
    [agentTasks, effectivePaperId, workerGroups],
  )
  const collaborationItems = useMemo(
    () =>
      paperWorkerGroups.map((group) => ({
        group,
        thread: findRelatedWorkerThread(chatTasks, {
          paperId: effectivePaperId,
          delegationTaskId: group.taskId,
          workerRunId: group.workerRunId,
        }),
      })),
    [chatTasks, effectivePaperId, paperWorkerGroups],
  )
  const boardTasks = useMemo(
    () => (effectivePaperId ? agentTasks.filter((task) => task.paperId === effectivePaperId) : agentTasks),
    [agentTasks, effectivePaperId],
  )

  const activeAgents = Object.values(agentStatuses).filter((entry) => entry.status === "working").length
  const filesTouchedCount = useMemo(
    () => Object.values(filesTouched).reduce((total, entries) => total + entries.length, 0),
    [filesTouched],
  )
  const linkedThreadCount = collaborationItems.filter((item) => item.thread).length
  const pendingApprovalCount = collaborationItems.filter((item) => item.thread?.pendingApproval).length
  const failedWorkerCount = paperWorkerGroups.filter((group) => group.status === "failed").length
  const queuedWorkerCount = paperWorkerGroups.filter((group) => group.status === "queued").length
  const managedWorkerCount = paperWorkerGroups.filter((group) => group.controlMode === "managed").length
  const mirroredWorkerCount = paperWorkerGroups.filter((group) => group.controlMode === "mirrored").length
  const runtimeSummary = monitorRuntimeSummary(
    runtimeLoading,
    runtimeRefreshing,
    runtimeInfo.error,
    runtimeInfo.statusLabel,
  )
  const attentionItems = useMemo(() => {
    const items: Array<{
      id: string
      label: string
      summary: string
      tone: string
      actionLabel: string
      onClick: () => void
    }> = []

    if (pendingApprovalCount > 0) {
      items.push({
        id: "approvals",
        label: `${pendingApprovalCount} approval ${pendingApprovalCount === 1 ? "gate" : "gates"}`,
        summary: "A parent Claude thread is blocked and waiting for a decision.",
        tone: "border-amber-200 bg-amber-50 text-amber-800",
        actionLabel: "Open collaboration",
        onClick: () => setActiveSection("collaboration"),
      })
    }

    if (failedWorkerCount > 0) {
      items.push({
        id: "failures",
        label: `${failedWorkerCount} failed ${failedWorkerCount === 1 ? "worker" : "workers"}`,
        summary: "Repair these runs before trusting the current swarm output.",
        tone: "border-rose-200 bg-rose-50 text-rose-700",
        actionLabel: "Open staff",
        onClick: () => setActiveSection("staff"),
      })
    }

    if (queuedWorkerCount > 0) {
      items.push({
        id: "queued",
        label: `${queuedWorkerCount} queued ${queuedWorkerCount === 1 ? "worker" : "workers"}`,
        summary: "Delegations exist but have not emitted useful runtime activity yet.",
        tone: "border-slate-200 bg-slate-100 text-slate-700",
        actionLabel: "Inspect tasks",
        onClick: () => setActiveSection("tasks"),
      })
    }

    if (items.length === 0) {
      items.push({
        id: "stable",
        label: "Swarm looks healthy",
        summary: "No blocked approvals, failed workers, or silent queues in the current paper.",
        tone: "border-emerald-200 bg-emerald-50 text-emerald-700",
        actionLabel: "View staff",
        onClick: () => setActiveSection("staff"),
      })
    }

    return items
  }, [failedWorkerCount, pendingApprovalCount, queuedWorkerCount])

  function openStudioChat() {
    if (effectivePaperId) {
      router.push(`/studio?paperId=${encodeURIComponent(effectivePaperId)}`)
      return
    }
    router.push("/studio")
  }

  function handleOpenRelatedThreadTask(taskId: string, paperId: string | null) {
    setActiveTask(taskId)
    requestWorkspaceSurface("log")
    if (paperId) {
      router.push(`/studio?paperId=${encodeURIComponent(paperId)}`)
      return
    }
    router.push("/studio")
  }

  function handleFocusWorker(workerRunId: string, nextSection: SwarmSection = "staff") {
    openWorkerRun(workerRunId)
    setActiveSection(nextSection)
  }

  if (!effectivePaperId && papers.length === 0) {
    return (
      <div className="flex h-screen min-h-0 items-center justify-center bg-[#f3f4ef] px-6">
        <div className="max-w-md rounded-[28px] border border-slate-200 bg-white px-6 py-6 text-center shadow-[0_22px_60px_rgba(15,23,42,0.06)]">
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Agent Swarm</p>
          <h1 className="mt-2 text-lg font-semibold text-slate-900">Select a paper first</h1>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            Monitor is a separate operator surface for staff, collaboration, and task flow.
          </p>
          <Button className="mt-5 rounded-full px-4" onClick={() => router.push("/studio")}>
            Back to Studio
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen min-h-0 flex-col bg-[#f1f2ed] text-slate-900">
      <div className="border-b border-slate-200 bg-[#f6f7f3]">
        <div className="flex min-h-12 flex-wrap items-center gap-2 px-4 py-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 rounded-full border border-transparent px-2.5 text-slate-600 hover:bg-white hover:text-slate-900"
            onClick={openStudioChat}
          >
            <ArrowLeft className="mr-1.5 h-3.5 w-3.5" />
            Chat
          </Button>

          <div className="min-w-0 flex flex-1 flex-wrap items-center gap-2 overflow-hidden">
            <div className="truncate text-[15px] font-semibold text-slate-900">
              {selectedPaper?.title ?? "Agent Swarm"}
            </div>
            <Badge variant="outline" className="h-5 border-slate-200 bg-white text-[10px] text-slate-600">
              Agent Swarm
            </Badge>
            <Badge variant="outline" className="h-5 border-slate-200 bg-white text-[10px] text-slate-600">
              OpenAkita + OpenClaw
            </Badge>
            {selectedPaper?.outputDir ? (
              <Badge variant="outline" className="hidden h-5 border-slate-200 bg-white text-[10px] text-slate-600 md:inline-flex">
                {formatWorkspaceLabel(selectedPaper.outputDir)}
              </Badge>
            ) : null}
            {boardSessionId ? (
              <Badge variant="outline" className="hidden h-5 border-slate-200 bg-white font-mono text-[10px] text-slate-600 lg:inline-flex">
                {boardSessionId.slice(0, 18)}
              </Badge>
            ) : null}
          </div>

          <div className="ml-auto flex flex-wrap items-center gap-1.5">
            <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
              {runtimeSummary}
            </span>
          </div>
        </div>

        <div className="px-4 pb-3">
          <Tabs value={activeSection} onValueChange={(value) => setActiveSection(value as SwarmSection)}>
            <TabsList className="grid w-full max-w-[520px] grid-cols-4 rounded-[18px] border border-slate-200 bg-white p-1">
              <TabsTrigger value="overview" className="rounded-[14px] text-[11px]">
                <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="staff" className="rounded-[14px] text-[11px]">
                <Users2 className="mr-1.5 h-3.5 w-3.5" />
                Staff
              </TabsTrigger>
              <TabsTrigger value="collaboration" className="rounded-[14px] text-[11px]">
                <MessageSquareText className="mr-1.5 h-3.5 w-3.5" />
                Collaboration
              </TabsTrigger>
              <TabsTrigger value="tasks" className="rounded-[14px] text-[11px]">
                <Workflow className="mr-1.5 h-3.5 w-3.5" />
                Tasks
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-3 space-y-3">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
                <MetricCard
                  icon={Users2}
                  label="Workers"
                  value={String(paperWorkerGroups.length)}
                  caption="Active swarm roster for the selected paper."
                />
                <MetricCard
                  icon={ShieldCheck}
                  label="Managed"
                  value={String(managedWorkerCount)}
                  caption="Sessions Studio can pause, resume, or cancel directly."
                />
                <MetricCard
                  icon={MessageSquareText}
                  label="Linked Threads"
                  value={String(linkedThreadCount)}
                  caption="Worker runs already mapped back to chat threads."
                />
                <MetricCard
                  icon={AlertTriangle}
                  label="Approvals"
                  value={String(pendingApprovalCount)}
                  caption="Parent-thread gates waiting for an explicit decision."
                />
                <MetricCard
                  icon={Wrench}
                  label="Tool Calls"
                  value={String(toolCalls.length)}
                  caption="All runtime tool actions captured in this workspace."
                />
                <MetricCard
                  icon={FileCode2}
                  label="Artifacts"
                  value={String(filesTouchedCount)}
                  caption="Files and code outputs touched by the active swarm."
                />
              </div>

              <div className="grid min-h-0 gap-3 xl:grid-cols-[minmax(0,1.15fr)_360px]">
                <SectionCard
                  eyebrow="Operator Overview"
                  title="Swarm Graph"
                  description="Task topology and focused worker routing."
                >
                  <div className="h-[420px] min-h-[320px]">
                    <AgentDagPanel />
                  </div>
                </SectionCard>

                <div className="grid min-h-0 gap-3">
                  <SectionCard
                    eyebrow="Operator Overview"
                    title="Needs Attention"
                    description="Items that deserve operator review before trusting the swarm output."
                  >
                    <div className="space-y-2 px-3 py-3">
                      {attentionItems.map((item) => (
                        <div
                          key={item.id}
                          className={cn("rounded-[20px] border px-3 py-2.5", item.tone)}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-[12px] font-semibold">{item.label}</div>
                              <div className="mt-1 text-[11px] leading-5">{item.summary}</div>
                            </div>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7 rounded-full bg-white/80 px-2.5 text-[10px]"
                              onClick={item.onClick}
                            >
                              {item.actionLabel}
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </SectionCard>

                  <SectionCard
                    eyebrow="Runtime"
                    title="Session State"
                    description="Claude Code transport, skill availability, and session health."
                  >
                    <div className="grid gap-2 px-3 py-3 text-[11px] text-slate-600">
                      <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                        <div className="text-[10px] uppercase tracking-[0.12em] text-slate-400">Claude Code</div>
                        <div className="mt-1 text-sm font-semibold text-slate-900">{runtimeInfo.label}</div>
                        <div className="mt-1 leading-5 text-slate-500">{runtimeSummary}</div>
                      </div>
                      <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                        <div className="grid gap-2">
                          <div className="flex items-center justify-between gap-2">
                            <span>Transport</span>
                            <span className="font-medium text-slate-900">{runtimeInfo.chatTransport}</span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Skills</span>
                            <span className="font-medium text-slate-900">{runtimeInfo.skills.length}</span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Model hint</span>
                            <span className="font-medium text-slate-900">{runtimeInfo.detectedDefaultModel ?? "Unavailable"}</span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Mirrored workers</span>
                            <span className="font-medium text-slate-900">{mirroredWorkerCount}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </SectionCard>
                </div>
              </div>

              <SectionCard
                eyebrow="Overview"
                title="Activity Snapshot"
                description="Fast scan of lifecycle, tools, and file output."
                action={
                  <Tabs value={activityPane} onValueChange={(value) => setActivityPane(value as ActivityPane)}>
                    <TabsList className="grid w-[240px] grid-cols-3 rounded-[16px] border border-slate-200 bg-white p-1">
                      <TabsTrigger value="live" className="rounded-[12px] text-[10px]">
                        Live
                      </TabsTrigger>
                      <TabsTrigger value="tools" className="rounded-[12px] text-[10px]">
                        Tools
                      </TabsTrigger>
                      <TabsTrigger value="files" className="rounded-[12px] text-[10px]">
                        Files
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                }
              >
                <div className="h-[300px] min-h-[220px] bg-slate-50">
                  {activityPane === "live" ? (
                    <ActivityFeed />
                  ) : activityPane === "tools" ? (
                    <ToolCallTimeline />
                  ) : (
                    <FileListPanel />
                  )}
                </div>
              </SectionCard>
            </TabsContent>

            <TabsContent value="staff" className="mt-3">
              <div className="grid min-h-0 gap-3 xl:grid-cols-[420px_minmax(0,1fr)]">
                <SectionCard
                  eyebrow="Staff"
                  title="Worker Roster"
                  description="Each worker remains drillable and controllable from this surface."
                  className="min-h-[720px]"
                >
                  <div className="h-[calc(100%-57px)] min-h-[620px]">
                    <SubagentActivityPanel onOpenRelatedThreadTask={handleOpenRelatedThreadTask} />
                  </div>
                </SectionCard>

                <div className="grid min-h-0 gap-3 xl:grid-rows-[auto_minmax(0,1fr)]">
                  <div className="grid gap-3 lg:grid-cols-2">
                    <SectionCard
                      eyebrow="Staff"
                      title="Role Coverage"
                      description="How the current swarm is staffed and controlled."
                    >
                      <div className="grid gap-2 px-3 py-3">
                        <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="text-slate-500">Managed sessions</span>
                            <span className="font-medium text-slate-900">{managedWorkerCount}</span>
                          </div>
                        </div>
                        <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="text-slate-500">Mirrored sessions</span>
                            <span className="font-medium text-slate-900">{mirroredWorkerCount}</span>
                          </div>
                        </div>
                        <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="text-slate-500">Queued workers</span>
                            <span className="font-medium text-slate-900">{queuedWorkerCount}</span>
                          </div>
                        </div>
                        <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="text-slate-500">Active agents</span>
                            <span className="font-medium text-slate-900">{activeAgents}</span>
                          </div>
                        </div>
                      </div>
                    </SectionCard>

                    <SectionCard
                      eyebrow="Staff"
                      title="Agent Status"
                      description="Current agent heartbeat across the workspace."
                    >
                      <div className="px-3 py-3">
                        <div className="rounded-[20px] border border-slate-200 bg-white px-2 py-2">
                          <AgentStatusPanel compact />
                        </div>
                      </div>
                    </SectionCard>
                  </div>

                  <SectionCard
                    eyebrow="Staff"
                    title="Top Workers"
                    description="Quick jump into the most recent worker runs."
                  >
                    <ScrollArea className="h-[360px] px-3 py-3">
                      <div className="space-y-2">
                        {paperWorkerGroups.length === 0 ? (
                          <div className="rounded-[20px] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                            No worker runs have been captured for this paper yet.
                          </div>
                        ) : (
                          paperWorkerGroups.slice(0, 8).map((group) => {
                            const meta = workerStatusMeta(group.status)
                            const presentation = getAgentPresentation(group.assignee)
                            return (
                              <button
                                key={group.workerRunId}
                                type="button"
                                className="flex w-full items-start gap-3 rounded-[20px] border border-slate-200 bg-white px-3 py-2.5 text-left transition-colors hover:border-slate-300 hover:bg-[#fbfcf8]"
                                onClick={() => handleFocusWorker(group.workerRunId, "staff")}
                              >
                                <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-200 bg-[#f4f5ef]">
                                  <Bot className="h-4 w-4 text-slate-600" />
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="flex flex-wrap items-center gap-2">
                                    <span className="truncate text-[12px] font-semibold text-slate-900">
                                      {presentation.label}
                                    </span>
                                    <span className={cn("rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]", meta.chipClass)}>
                                      {meta.label}
                                    </span>
                                  </div>
                                  <div className="mt-1 truncate text-[12px] text-slate-600">
                                    {group.taskTitle || "Untitled task"}
                                  </div>
                                  <div className="mt-1.5 flex flex-wrap gap-1.5 text-[10px] text-slate-500">
                                    <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5">
                                      {group.controlMode}
                                    </span>
                                    <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5">
                                      {group.runtime}
                                    </span>
                                    <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5">
                                      {group.toolCount} tools
                                    </span>
                                  </div>
                                </div>
                              </button>
                            )
                          })
                        )}
                      </div>
                    </ScrollArea>
                  </SectionCard>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="collaboration" className="mt-3">
              <div className="grid min-h-0 gap-3 xl:grid-cols-[460px_minmax(0,1fr)]">
                <SectionCard
                  eyebrow="Collaboration"
                  title="Relay Map"
                  description="How worker runs connect back to Claude threads and approvals."
                >
                  <ScrollArea className="h-[720px] px-3 py-3">
                    <div className="space-y-2">
                      {collaborationItems.length === 0 ? (
                        <div className="rounded-[20px] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                          No collaboration relays exist for this paper yet.
                        </div>
                      ) : (
                        collaborationItems.map(({ group, thread }) => {
                          const presentation = getAgentPresentation(group.assignee)
                          const state = collaborationState(thread, group)
                          const threadPreview = thread ? buildWorkerThreadPreview(thread.task) : null
                          return (
                            <div key={group.workerRunId} className="rounded-[22px] border border-slate-200 bg-white px-3 py-3">
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                  <div className="flex flex-wrap items-center gap-2">
                                    <span className="text-[12px] font-semibold text-slate-900">
                                      {presentation.label}
                                    </span>
                                    <span className={cn("rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]", state.className)}>
                                      {state.label}
                                    </span>
                                  </div>
                                  <div className="mt-1 text-[12px] text-slate-700">{group.taskTitle || "Untitled task"}</div>
                                  <p className="mt-1 text-[11px] leading-5 text-slate-500">{state.summary}</p>
                                </div>
                                <Button
                                  type="button"
                                  variant="outline"
                                  size="sm"
                                  className="h-7 rounded-full px-2.5 text-[10px]"
                                  onClick={() => handleFocusWorker(group.workerRunId, "staff")}
                                >
                                  Focus worker
                                </Button>
                              </div>

                              <div className="mt-2 flex flex-wrap gap-1.5">
                                <Badge variant="outline" className="text-[10px]">{group.controlMode}</Badge>
                                <Badge variant="outline" className="text-[10px]">{group.runtime}</Badge>
                                <Badge variant="outline" className="text-[10px]">{group.toolCount} tools</Badge>
                                {thread ? (
                                  <Badge variant="outline" className="text-[10px]">linked thread</Badge>
                                ) : null}
                              </div>

                              {thread ? (
                                <div className="mt-2 rounded-[18px] border border-slate-200 bg-[#f8faf6] px-3 py-2">
                                  <div className="flex items-start justify-between gap-2">
                                    <div className="min-w-0">
                                      <div className="truncate text-[11px] font-semibold text-slate-800">
                                        {thread.task.name}
                                      </div>
                                      <div className="mt-1 text-[11px] leading-5 text-slate-500">
                                        {threadPreview}
                                      </div>
                                    </div>
                                    <Button
                                      type="button"
                                      size="sm"
                                      className="h-7 rounded-full bg-slate-900 px-2.5 text-[10px] text-white hover:bg-slate-800"
                                      onClick={() => handleOpenRelatedThreadTask(thread.task.id, effectivePaperId)}
                                    >
                                      <ArrowUpRight className="mr-1.5 h-3.5 w-3.5" />
                                      Open thread
                                    </Button>
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          )
                        })
                      )}
                    </div>
                  </ScrollArea>
                </SectionCard>

                <SectionCard
                  eyebrow="Collaboration"
                  title="Runtime Evidence"
                  description="Live stream, tool trace, and file evidence for the current swarm."
                  action={
                    <Tabs value={activityPane} onValueChange={(value) => setActivityPane(value as ActivityPane)}>
                      <TabsList className="grid w-[240px] grid-cols-3 rounded-[16px] border border-slate-200 bg-white p-1">
                        <TabsTrigger value="live" className="rounded-[12px] text-[10px]">
                          Live
                        </TabsTrigger>
                        <TabsTrigger value="tools" className="rounded-[12px] text-[10px]">
                          Tools
                        </TabsTrigger>
                        <TabsTrigger value="files" className="rounded-[12px] text-[10px]">
                          Files
                        </TabsTrigger>
                      </TabsList>
                    </Tabs>
                  }
                >
                  <div className="h-[720px] min-h-[480px] bg-slate-50">
                    {activityPane === "live" ? (
                      <ActivityFeed />
                    ) : activityPane === "tools" ? (
                      <ToolCallTimeline />
                    ) : (
                      <FileListPanel />
                    )}
                  </div>
                </SectionCard>
              </div>
            </TabsContent>

            <TabsContent value="tasks" className="mt-3">
              <div className="grid min-h-0 gap-3 xl:grid-cols-[420px_minmax(0,1fr)]">
                <SectionCard
                  eyebrow="Tasks"
                  title="Task Board"
                  description="Claude-directed execution tasks for this paper."
                >
                  <ScrollArea className="h-[720px] px-3 py-3">
                    <div className="space-y-2">
                      {boardTasks.length === 0 ? (
                        <div className="rounded-[20px] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                          No structured tasks have been recorded for this paper yet.
                        </div>
                      ) : (
                        boardTasks.map((task) => (
                          <div key={task.id} className="rounded-[20px] border border-slate-200 bg-white px-3 py-3">
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="truncate text-[12px] font-semibold text-slate-900">{task.title}</div>
                                {task.description ? (
                                  <div className="mt-1 text-[11px] leading-5 text-slate-500">{task.description}</div>
                                ) : null}
                              </div>
                              <span className={cn("rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]", taskStatusMeta(task.status))}>
                                {task.status}
                              </span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1.5">
                              <Badge variant="outline" className="text-[10px]">{task.assignee}</Badge>
                              <Badge variant="outline" className="text-[10px]">{task.progress}%</Badge>
                              <Badge variant="outline" className="text-[10px]">{task.subtasks.length} subtasks</Badge>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </SectionCard>

                <div className="grid min-h-0 gap-3 xl:grid-rows-[minmax(320px,0.95fr)_minmax(320px,1.05fr)]">
                  <SectionCard
                    eyebrow="Tasks"
                    title="Task Graph"
                    description="Dependency and delegation structure for the active plan."
                  >
                    <div className="h-[360px] min-h-[280px]">
                      <AgentDagPanel />
                    </div>
                  </SectionCard>

                  <SectionCard
                    eyebrow="Tasks"
                    title="Artifacts"
                    description="Files touched and outputs produced by the current swarm."
                  >
                    <div className="h-[340px] min-h-[260px] bg-slate-50">
                      <FileListPanel />
                    </div>
                  </SectionCard>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
