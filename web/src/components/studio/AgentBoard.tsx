"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useStudioStore, type AgentTask, type PipelinePhase, type SandboxFileEntry } from "@/lib/store/studio-store"
import { readSSE } from "@/lib/sse"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
// Tabs removed — task detail now uses TaskDetailPanel
import { WorkspaceSetupDialog } from "./WorkspaceSetupDialog"
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  type Node,
  type Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import {
  nodeTypes,
  type CommanderNodeData,
  type TaskNodeData,
  type E2ENodeData,
  type OpenVSCodeNodeData,
} from "./AgentBoardNodes"
import { edgeTypes, EdgeAnimationStyles } from "./AgentBoardEdges"
import { AgentBoardSidebar } from "./AgentBoardSidebar"
import { TaskDetailPanel } from "./TaskDetailPanel"
import {
  ArrowLeft,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
  Square,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { backendUrl } from "@/lib/backend-url"

// Idle timeout: abort SSE if no event arrives within this window.
// Unlike a total timeout, this resets on every received event, so long-running
// pipelines that are still producing events will never be aborted.
const SSE_IDLE_TIMEOUT_MS = 10 * 60 * 1000
const AGENT_BOARD_SURFACE = "#f3f3f2"
const AGENT_BOARD_SIDEBAR_SURFACE = "#f7f7f8"

interface Props {
  paperId: string | null
  focusMode?: boolean
  onBack?: () => void
  showSidebar?: boolean
}

// LOG_LEVEL_STYLE, toLogTimestamp, extractPossibleFiles removed — now in TaskDetailPanel

function normalizeSandboxPath(path: string): string {
  return path
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\.?\//, "")
    .replace(/\/{2,}/g, "/")
    .replace(/\/$/, "")
}

function collectSandboxFilePaths(entries: SandboxFileEntry[], parent = ""): string[] {
  const paths: string[] = []
  for (const entry of entries) {
    const current = parent ? `${parent}/${entry.name}` : entry.name
    if (entry.type === "file") {
      paths.push(current)
      continue
    }
    if (entry.children && entry.children.length > 0) {
      paths.push(...collectSandboxFilePaths(entry.children, current))
    }
  }
  return paths
}

function sortSandboxEntries(entries: SandboxFileEntry[]): void {
  entries.sort((a, b) => {
    if (a.type !== b.type) return a.type === "directory" ? -1 : 1
    return a.name.localeCompare(b.name)
  })
  for (const entry of entries) {
    if (entry.type === "directory" && entry.children && entry.children.length > 0) {
      sortSandboxEntries(entry.children)
    }
  }
}

function buildSandboxFileTree(paths: string[]): SandboxFileEntry[] {
  const root: SandboxFileEntry[] = []

  for (const rawPath of paths) {
    const cleanPath = normalizeSandboxPath(rawPath)
    if (!cleanPath) continue

    const parts = cleanPath.split("/").filter(Boolean)
    if (parts.length === 0) continue

    let cursor = root
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const isLeaf = i === parts.length - 1

      if (isLeaf) {
        if (!cursor.some(entry => entry.type === "file" && entry.name === part)) {
          cursor.push({ name: part, type: "file" })
        }
        continue
      }

      let dir = cursor.find(
        (entry) => entry.type === "directory" && entry.name === part,
      ) as SandboxFileEntry | undefined
      if (!dir) {
        dir = { name: part, type: "directory", children: [] }
        cursor.push(dir)
      }
      if (!dir.children) {
        dir.children = []
      }
      cursor = dir.children
    }
  }

  sortSandboxEntries(root)
  return root
}

function mergeSandboxFileTree(current: SandboxFileEntry[], addedPaths: string[]): SandboxFileEntry[] {
  const currentPaths = collectSandboxFilePaths(current)
  const merged = new Set<string>([
    ...currentPaths.map(normalizeSandboxPath),
    ...addedPaths.map(normalizeSandboxPath),
  ])
  return buildSandboxFileTree(Array.from(merged).filter(Boolean))
}

function normalizeTaskStatus(rawStatus: unknown): AgentTask["status"] {
  const status = typeof rawStatus === "string" ? rawStatus : "planning"
  if (status === "ai_review") return "in_progress"
  if (
    status === "planning" ||
    status === "in_progress" ||
    status === "repairing" ||
    status === "human_review" ||
    status === "done" ||
    status === "paused" ||
    status === "cancelled"
  ) {
    return status
  }
  return "planning"
}

function normalizeAgentTaskFromBackend(rawTask: Record<string, unknown>, fallbackPaperId: string | null): AgentTask {
  return {
    id: (rawTask.id as string) || `task-${Date.now()}`,
    title: (rawTask.title as string) || "Untitled",
    description: (rawTask.description as string) || "",
    status: normalizeTaskStatus(rawTask.status),
    assignee: (rawTask.assignee as string) || "claude",
    progress: (rawTask.progress as number) || 0,
    tags: (rawTask.tags as string[]) || [],
    subtasks: (rawTask.subtasks as AgentTask["subtasks"]) || [],
    codexOutput: (rawTask.codexOutput as string) || (rawTask.codex_output as string) || undefined,
    generatedFiles:
      (rawTask.generatedFiles as string[]) || (rawTask.generated_files as string[]) || [],
    reviewFeedback:
      (rawTask.reviewFeedback as string) || (rawTask.review_feedback as string) || undefined,
    lastError: (rawTask.lastError as string) || (rawTask.last_error as string) || undefined,
    executionLog:
      (rawTask.executionLog as AgentTask["executionLog"]) ||
      (rawTask.execution_log as AgentTask["executionLog"]) ||
      [],
    paperId: (rawTask.paperId as string) || (rawTask.paper_id as string) || fallbackPaperId || undefined,
    createdAt: (rawTask.createdAt as string) || (rawTask.created_at as string) || new Date().toISOString(),
    updatedAt: (rawTask.updatedAt as string) || (rawTask.updated_at as string) || new Date().toISOString(),
  }
}

interface SessionSnapshotPayload {
  session_id?: unknown
  tasks?: unknown
  status?: unknown
  checkpoint?: unknown
  control_state?: unknown
  sandbox_id?: unknown
  context_pack_id?: unknown
}

function inferPipelinePhaseFromSession(
  controlState: unknown,
  status: unknown,
  checkpoint: unknown,
  tasks: AgentTask[],
): PipelinePhase {
  const normalizedControlState = typeof controlState === "string" ? controlState : ""
  if (normalizedControlState === "running") return "executing"
  if (normalizedControlState === "paused") return "paused"
  if (normalizedControlState === "cancelled") return "cancelled"

  const normalizedStatus = typeof status === "string" ? status : ""
  const normalizedCheckpoint = typeof checkpoint === "string" ? checkpoint : ""
  if (normalizedCheckpoint === "pipeline_cancelled") return "cancelled"
  if (normalizedCheckpoint === "run_complete") {
    const allDoneAtRunComplete = tasks.length > 0 && tasks.every(task => task.status === "done")
    return allDoneAtRunComplete ? "completed" : "idle"
  }
  if (normalizedStatus === "completed") return "completed"
  if (normalizedStatus === "failed") return "failed"
  if (normalizedStatus === "cancelled") return "cancelled"

  const hasPaused = tasks.some(task => task.status === "paused")
  if (hasPaused) return "paused"
  const hasInFlight = tasks.some(task => task.status === "in_progress" || task.status === "repairing")
  if (hasInFlight) return "executing"

  if (normalizedCheckpoint.startsWith("plan")) return "planning"
  // Only infer "executing" from checkpoint if there are actually in-flight tasks.
  // After a server restart or stale session, checkpoint may say "run_*" but
  // nothing is actually running (no RunControl in memory → control_state is null).
  if (
    (normalizedCheckpoint.startsWith("run_") || normalizedCheckpoint.startsWith("task_") || normalizedCheckpoint.startsWith("executor_")) &&
    hasInFlight
  ) {
    return "executing"
  }

  const hasPlanning = tasks.some(task => task.status === "planning")
  if (hasPlanning) return "idle"
  const allDone = tasks.length > 0 && tasks.every(task => task.status === "done")
  if (allDone) return "completed"
  return "idle"
}

// ---------------------------------------------------------------------------
// Node / Edge builder
// ---------------------------------------------------------------------------

const COMMANDER_CARD_WIDTH = 280
const COMMANDER_FOCUS_CARD_WIDTH = 430
const TASK_CARD_WIDTH = 260
const E2E_CARD_WIDTH = 380
const DOWNLOAD_CARD_WIDTH = 320
const CARD_SPACING_X = 300
const ROW_Y = { commander: 0, tasks: 220 }

function buildFlowNodes(
  tasks: AgentTask[],
  paperTitle: string,
  pipelinePhase: string,
  e2eState: ReturnType<typeof useStudioStore.getState>["e2eState"],
  workspaceDir: string,
  showTaskGraph: boolean,
  onCommanderClick: () => void,
  onTaskClick: (taskId: string) => void,
  onE2EClick: () => void,
): Node[] {
  const nodes: Node[] = []

  // Commander node
  const commanderAction =
    pipelinePhase === "planning"
      ? "Decomposing tasks..."
      : pipelinePhase === "executing"
        ? "Monitoring execution..."
        : pipelinePhase === "paused"
          ? "Pipeline paused"
          : pipelinePhase === "cancelled"
            ? "Pipeline cancelled"
            : pipelinePhase === "e2e_running"
              ? "Running full project..."
              : pipelinePhase === "e2e_repairing"
                ? "Directing repair..."
                : pipelinePhase === "completed"
                  ? "All stages complete"
                  : pipelinePhase === "failed"
                    ? "Pipeline failed"
                    : "Ready"

  const commanderStatus =
    pipelinePhase === "idle" || pipelinePhase === "completed" || pipelinePhase === "failed"
      ? "idle"
      : pipelinePhase === "planning" || pipelinePhase === "e2e_repairing"
        ? "working"
        : "ready"

  const commanderWidth = showTaskGraph ? COMMANDER_CARD_WIDTH : COMMANDER_FOCUS_CARD_WIDTH
  nodes.push({
    id: "commander",
    type: "commander",
    position: { x: -commanderWidth / 2, y: ROW_Y.commander },
    data: {
      paperTitle,
      action: commanderAction,
      status: commanderStatus,
      onClick: showTaskGraph ? undefined : onCommanderClick,
      emphasize: !showTaskGraph,
      showExpandHint: !showTaskGraph,
    } satisfies CommanderNodeData,
    draggable: false,
    selectable: true,
  })

  // Initial board: focus on the commander card only.
  if (!showTaskGraph) {
    return nodes
  }

  // Task nodes
  const n = tasks.length
  for (let i = 0; i < n; i++) {
    const x = (i - (n - 1) / 2) * CARD_SPACING_X - TASK_CARD_WIDTH / 2
    nodes.push({
      id: `task-${tasks[i].id}`,
      type: "task",
      position: { x, y: ROW_Y.tasks },
      data: {
        task: tasks[i],
        onClick: () => onTaskClick(tasks[i].id),
      } satisfies TaskNodeData,
      draggable: false,
      selectable: true,
    })
  }

  // Keep task cards in one row; only move E2E lower to keep edges visible.
  const e2eY = ROW_Y.tasks + 340
  const downloadY = e2eY + 320

  // E2E node — centered at x=0 (aligned with Commander)
  nodes.push({
    id: "e2e",
    type: "e2e",
    position: { x: -E2E_CARD_WIDTH / 2, y: e2eY },
    data: {
      e2e: e2eState,
      pipelinePhase,
      onClick: onE2EClick,
    } satisfies E2ENodeData,
    draggable: false,
    selectable: true,
  })

  // VS Code node
  nodes.push({
    id: "download",
    type: "download",
    position: { x: -DOWNLOAD_CARD_WIDTH / 2, y: downloadY },
    data: { directory: workspaceDir, pipelinePhase } satisfies OpenVSCodeNodeData,
    draggable: false,
    selectable: true,
  })

  return nodes
}

function buildFlowEdges(
  tasks: AgentTask[],
  pipelinePhase: string,
  e2eState: ReturnType<typeof useStudioStore.getState>["e2eState"],
  visibleEdgeIds: Set<string>,
  showTaskGraph: boolean,
): Edge[] {
  if (!showTaskGraph) {
    return []
  }
  const edges: Edge[] = []
  const marker = { type: MarkerType.ArrowClosed, width: 14, height: 14, color: "#d4d4d8" }

  // Commander → each task
  for (const task of tasks) {
    const edgeId = `commander-task-${task.id}`
    if (!visibleEdgeIds.has(edgeId)) continue
    edges.push({
      id: edgeId,
      source: "commander",
      target: `task-${task.id}`,
      type: "animated",
      markerEnd: marker,
      data: { variant: "default" },
    })
  }

  // Each task → E2E
  for (const task of tasks) {
    if (task.status !== "done" && task.status !== "human_review" && task.status !== "repairing") continue
    const edgeId = `task-${task.id}-e2e`
    edges.push({
      id: edgeId,
      source: `task-${task.id}`,
      target: "e2e",
      type: "animated",
      markerEnd: { ...marker, color: "#22c55e" },
      data: { variant: "default" },
    })
  }

  // E2E repair → specific task (red dashed loop back)
  if (e2eState?.status === "repairing" || pipelinePhase === "e2e_repairing") {
    // Find tasks being repaired
    for (const task of tasks) {
      if (task.status === "repairing") {
        edges.push({
          id: `repair-${task.id}`,
          source: "e2e",
          target: `task-${task.id}`,
          type: "animated",
          markerEnd: { ...marker, color: "#ef4444" },
          data: { variant: "repair" },
        })
      }
    }
  }

  // E2E → Download (on success)
  if (
    pipelinePhase === "downloading" ||
    pipelinePhase === "completed" ||
    e2eState?.status === "passed" ||
    e2eState?.status === "skipped"
  ) {
    edges.push({
      id: "e2e-download",
      source: "e2e",
      target: "download",
      type: "animated",
      markerEnd: { ...marker, color: "#22c55e" },
      data: { variant: "success" },
    })
  }

  return edges
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function AgentBoard({ paperId, focusMode = false, onBack, showSidebar = true }: Props) {
  const {
    agentTasks,
    boardSessionId,
    setBoardSessionId,
    addAgentTask,
    updateAgentTask,
    replaceAgentTasksForPaper,
    papers,
    updatePaper,
    pipelinePhase,
    setPipelinePhase,
    e2eState,
    setE2EState,
    setSandboxFiles,
  } = useStudioStore()

  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
  const [showE2EDetail, setShowE2EDetail] = useState(false)
  const [showTaskGraph, setShowTaskGraph] = useState(false)
  const [visibleEdgeIds, setVisibleEdgeIds] = useState<Set<string>>(new Set())
  const abortRef = useRef<AbortController | null>(null)
  const hydrateAttemptedRef = useRef<Set<string>>(new Set())

  const filteredTasks = useMemo(
    () => (paperId ? agentTasks.filter(t => t.paperId === paperId) : agentTasks),
    [agentTasks, paperId],
  )

  const selectedTask = useMemo(
    () => (selectedTaskId ? agentTasks.find(t => t.id === selectedTaskId) || null : null),
    [agentTasks, selectedTaskId],
  )
  const selectedPaper = useMemo(
    () => (paperId ? papers.find(p => p.id === paperId) || null : null),
    [paperId, papers],
  )
  const workspaceDir = selectedPaper?.outputDir || null
  const paperTitle = selectedPaper?.title || "Untitled"

  const totalTasks = filteredTasks.length
  const doneCount = filteredTasks.filter(t => t.status === "done" || t.status === "human_review").length
  // cancelledCount removed — cancel now leads to full restart, not selective re-run
  const runnableCount = filteredTasks.filter(
    (task) =>
      task.status === "planning" ||
      task.status === "in_progress" ||
      task.status === "repairing" ||
      task.status === "paused" ||
      task.status === "human_review",
  ).length

  useEffect(() => {
    setShowTaskGraph(false)
  }, [paperId])

  useEffect(() => {
    const shouldExpandByState =
      pipelinePhase !== "idle" || filteredTasks.some(task => task.status !== "planning")
    if (shouldExpandByState) {
      setShowTaskGraph(true)
    }
  }, [filteredTasks, pipelinePhase])

  // Stagger commander→task edges only during live planning.
  // On page refresh (tasks already hydrated), show all edges immediately.
  const prevTaskCountRef = useRef(filteredTasks.length)
  useEffect(() => {
    if (filteredTasks.length === 0) return
    const ids = filteredTasks.map(t => `commander-task-${t.id}`)

    // Tasks appeared all at once (hydration / refresh) — show immediately
    if (prevTaskCountRef.current > 0 || pipelinePhase !== "planning") {
      setVisibleEdgeIds(prev => new Set([...prev, ...ids]))
      prevTaskCountRef.current = filteredTasks.length
      return
    }

    // Live planning — stagger edges one by one
    prevTaskCountRef.current = filteredTasks.length
    let i = 0
    const interval = setInterval(() => {
      if (i >= ids.length) {
        clearInterval(interval)
        return
      }
      setVisibleEdgeIds(prev => new Set([...prev, ids[i]]))
      i++
    }, 300)
    return () => clearInterval(interval)
  }, [filteredTasks.length, pipelinePhase]) // eslint-disable-line react-hooks/exhaustive-deps

  const onCommanderClick = useCallback(() => {
    setShowTaskGraph(true)
  }, [])
  const onTaskClick = useCallback((taskId: string) => setSelectedTaskId(taskId), [])
  const onE2EClick = useCallback(() => setShowE2EDetail(true), [])
  const refreshSandboxFilesFromServer = useCallback(async () => {
    if (!boardSessionId) return
    try {
      const resp = await fetch(
        backendUrl(`/api/agent-board/sessions/${boardSessionId}/sandbox/tree`),
      )
      if (!resp.ok) return
      const payload = (await resp.json()) as { files?: unknown }
      const files = Array.isArray(payload.files)
        ? payload.files.filter((item): item is string => typeof item === "string")
        : []
      setSandboxFiles(buildSandboxFileTree(files))
    } catch {
      // Keep existing sidebar state on transient fetch failures.
    }
  }, [boardSessionId, setSandboxFiles])

  const hydrateFromSnapshot = useCallback((payload: SessionSnapshotPayload, fallbackSessionId: string | null) => {
    if (!paperId) return

    const nextSessionId = typeof payload.session_id === "string" ? payload.session_id.trim() : ""
    const resolvedSessionId = nextSessionId || fallbackSessionId || ""
    if (resolvedSessionId && resolvedSessionId !== boardSessionId) {
      setBoardSessionId(resolvedSessionId)
    }

    const contextPackId =
      typeof payload.context_pack_id === "string" ? payload.context_pack_id.trim() : ""
    if (contextPackId) {
      updatePaper(paperId, { contextPackId })
    }

    const rawTasks = Array.isArray(payload.tasks)
      ? payload.tasks.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
      : []
    const normalizedTasks = rawTasks.map((task) => normalizeAgentTaskFromBackend(task, paperId))
    replaceAgentTasksForPaper(paperId, normalizedTasks)

    const restoredPhase = inferPipelinePhaseFromSession(
      payload.control_state,
      payload.status,
      payload.checkpoint,
      normalizedTasks,
    )
    setPipelinePhase(restoredPhase)

    // Restore sandbox file tree whenever the session has a sandbox_id,
    // regardless of pipeline phase — completed/cancelled runs still have
    // useful files in the sidebar.
    const hasSandboxId =
      typeof payload.sandbox_id === "string" && payload.sandbox_id.trim().length > 0
    if (hasSandboxId) {
      void refreshSandboxFilesFromServer()
    }
  }, [
    boardSessionId,
    paperId,
    refreshSandboxFilesFromServer,
    replaceAgentTasksForPaper,
    setBoardSessionId,
    setPipelinePhase,
    updatePaper,
  ])

  useEffect(() => {
    if (!paperId) return
    const attemptKey = `${paperId}:${boardSessionId ?? "__none__"}`
    if (hydrateAttemptedRef.current.has(attemptKey)) return
    hydrateAttemptedRef.current.add(attemptKey)

    let cancelled = false

    const hydrateFromSession = async () => {
      try {
        if (boardSessionId) {
          const resp = await fetch(backendUrl(`/api/agent-board/sessions/${boardSessionId}`))
          if (resp.ok) {
            const payload = (await resp.json()) as SessionSnapshotPayload
            if (cancelled) return
            hydrateFromSnapshot(payload, boardSessionId)
            return
          }
        }

        const latestResp = await fetch(
          backendUrl(`/api/agent-board/sessions/latest/by-paper?paper_id=${encodeURIComponent(paperId)}`),
        )
        if (!latestResp.ok) return
        const latestPayload = (await latestResp.json()) as SessionSnapshotPayload
        if (cancelled) return
        const latestSessionId =
          typeof latestPayload.session_id === "string" ? latestPayload.session_id.trim() : null
        hydrateFromSnapshot(latestPayload, latestSessionId)
      } catch {
        // Ignore session rehydrate errors; live SSE updates can still drive UI.
      }
    }

    void hydrateFromSession()
    return () => {
      cancelled = true
    }
  }, [
    boardSessionId,
    hydrateFromSnapshot,
    paperId,
  ])

  // Build ReactFlow nodes and edges
  const nodes = useMemo(
    () =>
      buildFlowNodes(
        filteredTasks,
        paperTitle,
        pipelinePhase,
        e2eState,
        workspaceDir || "",
        showTaskGraph,
        onCommanderClick,
        onTaskClick,
        onE2EClick,
      ),
    [
      filteredTasks,
      paperTitle,
      pipelinePhase,
      e2eState,
      workspaceDir,
      showTaskGraph,
      onCommanderClick,
      onTaskClick,
      onE2EClick,
    ],
  )

  const edges = useMemo(
    () => buildFlowEdges(filteredTasks, pipelinePhase, e2eState, visibleEdgeIds, showTaskGraph),
    [filteredTasks, pipelinePhase, e2eState, visibleEdgeIds, showTaskGraph],
  )



  // -------------------------------------------------------------------------
  // Task log helper
  // -------------------------------------------------------------------------

  const appendInlineTaskLog = (
    taskId: string,
    event: string,
    phase: string,
    message: string,
    level: "info" | "warning" | "error" | "success" = "info",
  ) => {
    const task = useStudioStore.getState().agentTasks.find(item => item.id === taskId)
    if (!task) return
    const current = task.executionLog || []
    updateAgentTask(taskId, {
      executionLog: [
        ...current,
        {
          id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          timestamp: new Date().toISOString(),
          event,
          phase,
          level,
          message,
        },
      ],
    })
  }

  const upsertTaskFromEvent = (taskId: string, rawTask: Record<string, unknown>) => {
    const existing = useStudioStore.getState().agentTasks.some(t => t.id === taskId)
    const normalized = normalizeAgentTaskFromBackend(rawTask, paperId)

    if (existing) {
      const updates: Partial<AgentTask> = {
        status: normalized.status,
        assignee: normalized.assignee,
        progress: normalized.progress,
        subtasks: normalized.subtasks,
        codexOutput: normalized.codexOutput,
        generatedFiles: normalized.generatedFiles,
        reviewFeedback: normalized.reviewFeedback,
        lastError: normalized.lastError,
        executionLog: normalized.executionLog,
      }
      updateAgentTask(taskId, updates)
      return
    }

    addAgentTask(normalized)
  }

  // -------------------------------------------------------------------------
  // SSE run handler
  // -------------------------------------------------------------------------

  const runAllWithWorkspace = async (targetWorkspaceDir: string, opts?: { resetCancelled?: boolean; restart?: boolean; continueRun?: boolean }) => {
    if (!boardSessionId || running) return
    setRunning(true)
    setRunError(null)
    setPipelinePhase("executing")

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    // Idle timeout: resets every time an SSE event arrives.
    let idleTimer = setTimeout(() => controller.abort(), SSE_IDLE_TIMEOUT_MS)
    const resetIdleTimer = () => {
      clearTimeout(idleTimer)
      idleTimer = setTimeout(() => controller.abort(), SSE_IDLE_TIMEOUT_MS)
    }

    try {
      const res = await fetch(
        backendUrl(`/api/agent-board/sessions/${boardSessionId}/run`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            workspace_dir: targetWorkspaceDir,
            reset_cancelled: opts?.resetCancelled ?? false,
            restart: opts?.restart ?? false,
            continue_run: opts?.continueRun ?? false,
          }),
          signal: controller.signal,
        },
      )
      if (!res.ok || !res.body) throw new Error(`Run failed (${res.status})`)

      for await (const evt of readSSE(res.body)) {
        if (controller.signal.aborted) break
        resetIdleTimer()

        if (evt?.type === "progress") {
          const data = (evt.data ?? {}) as Record<string, unknown>
          const eventName = (data.event as string) || "progress"
          const taskData = data.task as Record<string, unknown> | undefined
          const taskId = (data.task_id as string) || (taskData?.id as string)
          const mergeFilesToSidebar = (paths: string[]) => {
            if (paths.length === 0) return
            const currentFiles = useStudioStore.getState().sandboxFiles
            setSandboxFiles(mergeSandboxFileTree(currentFiles, paths))
          }

          // Task lifecycle events
          if (taskId && taskData) {
            upsertTaskFromEvent(taskId, taskData)
          } else if (taskId && eventName === "task_codex_done") {
            updateAgentTask(taskId, { status: "in_progress", progress: 70 })
            appendInlineTaskLog(
              taskId,
              "task_codex_done",
              "codex_running",
              "Subagent output received.",
              "success",
            )
          }

          if (taskId && !taskData && eventName === "task_failed") {
            const error = (data.error as string) || "Task failed."
            updateAgentTask(taskId, { status: "done", lastError: error, progress: 100 })
            appendInlineTaskLog(taskId, "task_failed", "codex_running", error, "error")
          }

          // Verification events → silent, just log
          if (eventName === "verify_started" || eventName === "verify_finished" || eventName === "verify_failed") {
            if (taskId) {
              const msg = eventName === "verify_started"
                ? "Verification started..."
                : eventName === "verify_finished"
                  ? `Verification ${(data.passed as boolean) ? "passed" : "failed"}.`
                  : "Verification failed."
              appendInlineTaskLog(taskId, eventName, "verification", msg,
                eventName === "verify_finished" && (data.passed as boolean) ? "success" : "warning")
            }
          }

          // E2E events
          if (eventName === "e2e_started") {
            setPipelinePhase("e2e_running")
            setE2EState({
              status: "running",
              attempt: 0,
              maxAttempts: (data.max_attempts as number) || 4,
              entryPoint: (data.entry_point as string) || null,
              command: (data.command as string) || null,
              lastExitCode: null,
              lastStdout: "",
              lastStderr: "",
              history: [],
            })
          }

          if (eventName === "e2e_attempt") {
            const success = data.success as boolean
            const attempt = (data.attempt as number) || 0
            setE2EState({
              status: success ? "passed" : attempt < ((e2eState?.maxAttempts ?? 3)) ? "repairing" : "failed",
              attempt,
              lastExitCode: (data.exit_code as number) ?? null,
              lastStdout: (data.stdout_preview as string) || "",
              lastStderr: (data.stderr_preview as string) || "",
            })
            if (!success && attempt < ((e2eState?.maxAttempts ?? 3))) {
              setPipelinePhase("e2e_repairing")
            }
          }

          if (eventName === "e2e_finished") {
            const success = data.success as boolean
            setE2EState({
              status: success ? "passed" : "failed",
              lastStdout: (data.stdout_preview as string) || "",
            })
            if (success) {
              setPipelinePhase("downloading")
            } else {
              setPipelinePhase("failed")
            }
          }

          if (eventName === "e2e_error") {
            setE2EState({ status: "failed" })
            setPipelinePhase("failed")
          }

          if (eventName === "e2e_skipped") {
            // Skip E2E, go straight to download
            setPipelinePhase("downloading")
            setE2EState({
              status: "skipped",
              entryPoint: null,
              command: null,
              lastStdout: (data.reason as string) || "No entry point detected — E2E execution was skipped.",
            })
          }

          // Download events
          if (eventName === "download_complete") {
            const downloadedFiles = Array.isArray(data.files_downloaded)
              ? (data.files_downloaded as string[])
              : []
            mergeFilesToSidebar(downloadedFiles)
            void refreshSandboxFilesFromServer()
            setPipelinePhase("completed")
          }

          if (eventName === "download_skipped") {
            setPipelinePhase("completed")
          }

          // Pipeline control events
          if (eventName === "pipeline_paused") {
            setPipelinePhase("paused")
            // Update task statuses from backend
            const pausedTasks = data.tasks as Array<Record<string, unknown>> | undefined
            if (pausedTasks) {
              for (const t of pausedTasks) {
                const tid = t.id as string
                if (tid) upsertTaskFromEvent(tid, t)
              }
            }
          }
          if (eventName === "pipeline_resumed") {
            setPipelinePhase("executing")
            const resumedTasks = data.tasks as Array<Record<string, unknown>> | undefined
            if (resumedTasks) {
              for (const t of resumedTasks) {
                const tid = t.id as string
                if (tid) upsertTaskFromEvent(tid, t)
              }
            }
          }
          if (eventName === "pipeline_cancelled") {
            setPipelinePhase("cancelled")
            // Update all tasks from backend cancel event
            const cancelledTasks = data.tasks as Array<Record<string, unknown>> | undefined
            if (cancelledTasks) {
              for (const t of cancelledTasks) {
                const tid = t.id as string
                if (tid) upsertTaskFromEvent(tid, t)
              }
            }
          }
          if (eventName === "sandbox_auto_released") {
            // Sandbox has been released — could update a sandbox indicator
          }

          // Sandbox file updates from executor/knowledge steps.
          const filesWritten = Array.isArray(data.files_written)
            ? (data.files_written as string[])
            : []
          mergeFilesToSidebar(filesWritten)
        } else if (evt?.type === "result") {
          if (pipelinePhase !== "completed" && pipelinePhase !== "failed") {
            setPipelinePhase("completed")
          }
          break
        } else if (evt?.type === "error") {
          setRunError(evt.message || "Execution failed")
          setPipelinePhase("failed")
          break
        }
      }
    } catch (err) {
      if (controller.signal.aborted) {
        setRunError("Connection lost — no events received for 10 minutes")
      } else {
        setRunError(err instanceof Error ? err.message : "Run failed")
      }
      setPipelinePhase("failed")
    } finally {
      clearTimeout(idleTimer)
      abortRef.current = null
      setRunning(false)
    }
  }

  const handleWorkspaceConfirm = (directory: string) => {
    setShowWorkspaceSetup(false)
    setRunError(null)
    if (paperId) {
      updatePaper(paperId, { outputDir: directory })
    }
    void runAllWithWorkspace(directory)
  }

  const handleRunAll = async () => {
    if (!boardSessionId || running) return
    const hasPlanningTasks = filteredTasks.some(t => t.status === "planning")
    const hasCompletedTasks = filteredTasks.some(t => t.status === "done" || t.status === "human_review")
    // Mix of completed + incomplete → continue from where we left off
    if (!hasPlanningTasks && hasCompletedTasks && filteredTasks.length > 0) {
      await handleContinue()
      return
    }
    // All completed, no planning → full restart
    if (!hasPlanningTasks && filteredTasks.length > 0) {
      await handleRestart()
      return
    }
    setShowTaskGraph(true)
    if (!workspaceDir) {
      if (selectedPaper) {
        setShowWorkspaceSetup(true)
        return
      }
      setRunError("Set up a workspace directory first.")
      return
    }
    await runAllWithWorkspace(workspaceDir)
  }

  const handleCancel = async () => {
    if (!boardSessionId) return
    const finalize = () => {
      abortRef.current?.abort()
      setPipelinePhase("cancelled")
      setRunning(false)
      // Mark running tasks as cancelled; keep completed tasks as-is.
      for (const task of filteredTasks) {
        if (task.status === "in_progress" || task.status === "repairing") {
          updateAgentTask(task.id, { status: "cancelled" as AgentTask["status"] })
        }
      }
    }
    try {
      const res = await fetch(
        backendUrl(`/api/agent-board/sessions/${boardSessionId}/cancel`),
        { method: 'POST' },
      )
      if (res.ok) finalize()
    } catch {
      // Network error — still reset UI so user isn't stuck
      finalize()
    }
  }

  const handleContinue = async () => {
    if (!boardSessionId || running) return
    // Reset only incomplete tasks; keep done/human_review as-is
    for (const task of filteredTasks) {
      if (task.status !== "done" && task.status !== "human_review") {
        updateAgentTask(task.id, {
          status: "planning" as AgentTask["status"],
          progress: 0,
          lastError: undefined,
          codexOutput: undefined,
          generatedFiles: [],
          executionLog: [],
        })
      }
    }
    setPipelinePhase("executing")
    setE2EState({ status: "waiting", attempt: 0, lastExitCode: null, lastStdout: "", lastStderr: "", entryPoint: null, command: null })

    setRunError(null)
    setShowTaskGraph(true)
    if (!workspaceDir) {
      if (selectedPaper) {
        setShowWorkspaceSetup(true)
        return
      }
      setRunError("Set up a workspace directory first.")
      return
    }
    await runAllWithWorkspace(workspaceDir, { continueRun: true })
  }

  const handleRestart = async () => {
    if (!boardSessionId || running) return
    const now = new Date().toISOString()
    // Reset ALL tasks in local store for immediate UI feedback
    for (const task of filteredTasks) {
      updateAgentTask(task.id, {
        status: "planning" as AgentTask["status"],
        progress: 0,
        lastError: undefined,
        codexOutput: undefined,
        reviewFeedback: undefined,
        generatedFiles: [],
        executionLog: [],
        createdAt: now,
        updatedAt: now,
      })
    }
    setPipelinePhase("executing")
    setE2EState({ status: "waiting", attempt: 0, lastExitCode: null, lastStdout: "", lastStderr: "", entryPoint: null, command: null })

    setSandboxFiles([])
    setRunError(null)
    setShowTaskGraph(true)
    if (!workspaceDir) {
      if (selectedPaper) {
        setShowWorkspaceSetup(true)
        return
      }
      setRunError("Set up a workspace directory first.")
      return
    }
    // Backend resets all tasks to planning and starts a fresh pipeline
    await runAllWithWorkspace(workspaceDir, { restart: true })
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div
        className={cn(
          "px-4 py-3 border-b flex items-center justify-between shrink-0",
          focusMode ? "border-zinc-200 bg-[#f3f3f2]" : "border-zinc-100 bg-white",
        )}
      >
        <div className="flex items-center gap-3 min-w-0">
          {focusMode && (
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-xs gap-1.5 shrink-0"
              onClick={onBack}
              title="Back to Studio"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          )}
          <h2 className="text-sm font-semibold text-zinc-800 truncate">
            {focusMode ? paperTitle : "Agent Board"}
          </h2>
          {totalTasks > 0 && (
            <span className="text-xs text-zinc-500 shrink-0">
              {doneCount}/{totalTasks} completed
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {runError && (
            <span className="text-xs text-red-500 max-w-[280px] truncate" title={runError}>
              {runError}
            </span>
          )}
          {/* Pipeline controls */}
          {(pipelinePhase === "executing" || pipelinePhase === "e2e_running" || pipelinePhase === "e2e_repairing") && (
            <>
              <Button
                size="sm"
                variant="outline"
                onClick={() => fetch(backendUrl(`/api/agent-board/sessions/${boardSessionId}/pause`), { method: 'POST' })}
                disabled={!boardSessionId}
                className="h-7 text-xs gap-1.5"
              >
                <Pause className="h-3.5 w-3.5" />
                Pause
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleCancel}
                disabled={!boardSessionId}
                className="h-7 text-xs gap-1.5 text-red-600 border-red-200 hover:bg-red-50"
              >
                <Square className="h-3.5 w-3.5" />
                Cancel
              </Button>
            </>
          )}
          {pipelinePhase === "paused" && (
            <>
              <Button
                size="sm"
                onClick={() => fetch(backendUrl(`/api/agent-board/sessions/${boardSessionId}/resume`), { method: 'POST' })}
                disabled={!boardSessionId}
                className="h-7 text-xs gap-1.5"
              >
                <Play className="h-3.5 w-3.5" />
                Run
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleCancel}
                disabled={!boardSessionId}
                className="h-7 text-xs gap-1.5 text-red-600 border-red-200 hover:bg-red-50"
              >
                <Square className="h-3.5 w-3.5" />
                Cancel
              </Button>
            </>
          )}
          {(pipelinePhase === "cancelled" || pipelinePhase === "failed") && (() => {
            const hasIncomplete = filteredTasks.some(t => t.status !== "done" && t.status !== "human_review")
            const hasCompleted = filteredTasks.some(t => t.status === "done" || t.status === "human_review")
            const showContinue = hasIncomplete && hasCompleted
            return (
              <>
                {showContinue && (
                  <Button
                    size="sm"
                    onClick={handleContinue}
                    disabled={running || !boardSessionId}
                    className="h-7 text-xs gap-1.5"
                  >
                    {running ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <SkipForward className="h-3.5 w-3.5" />
                    )}
                    Continue
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRestart}
                  disabled={running || !boardSessionId}
                  className="h-7 text-xs gap-1.5"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Restart
                </Button>
              </>
            )
          })()}
          {runnableCount > 0 && pipelinePhase !== "executing" && pipelinePhase !== "paused" && pipelinePhase !== "cancelled" && pipelinePhase !== "failed" && pipelinePhase !== "e2e_running" && pipelinePhase !== "e2e_repairing" && (
            <Button
              size="sm"
              onClick={handleRunAll}
              disabled={running || !boardSessionId}
              className="h-7 text-xs gap-1.5"
            >
              {running ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {running ? "Running..." : `Run All (${runnableCount})`}
            </Button>
          )}
        </div>
      </div>

      {/* Body: sidebar + canvas */}
      <div className="flex-1 flex min-h-0" style={{ background: AGENT_BOARD_SURFACE }}>
        {showSidebar ? (
          <AgentBoardSidebar backgroundColor={AGENT_BOARD_SIDEBAR_SURFACE} />
        ) : null}
        <div className="flex-1 min-w-0 h-full" style={{ background: AGENT_BOARD_SURFACE }}>
          <EdgeAnimationStyles />
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            fitViewOptions={{
              maxZoom: showTaskGraph ? 1 : 1.35,
              minZoom: showTaskGraph ? 0.2 : 0.6,
              padding: showTaskGraph ? 0.16 : 0.42,
            }}
            panOnDrag
            zoomOnScroll
            nodesConnectable={false}
            nodesDraggable={false}
            elementsSelectable={showTaskGraph}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              gap={20}
              size={1}
              color="#e5e5e3"
              // @ts-expect-error variant dots supported
              variant="dots"
            />
            <Controls showInteractive={false} className="!bg-white !border-zinc-200 !shadow-sm" />
          </ReactFlow>
        </div>
      </div>

      {/* Task detail dialog */}
      <TaskDetailPanel
        task={selectedTask}
        open={!!selectedTask}
        onOpenChange={open => !open && setSelectedTaskId(null)}
      />

      {/* E2E detail dialog */}
      <E2EDetailDialog
        e2e={e2eState}
        open={showE2EDetail}
        onOpenChange={setShowE2EDetail}
      />

      {/* Workspace setup */}
      {selectedPaper && (
        <WorkspaceSetupDialog
          paper={selectedPaper}
          open={showWorkspaceSetup}
          onConfirm={handleWorkspaceConfirm}
          onCancel={() => setShowWorkspaceSetup(false)}
        />
      )}
    </div>
  )
}

// TaskDetailDialog replaced by TaskDetailPanel (imported from ./TaskDetailPanel)

// ---------------------------------------------------------------------------
// E2E Detail Dialog
// ---------------------------------------------------------------------------

function E2EDetailDialog({
  e2e,
  open,
  onOpenChange,
}: {
  e2e: ReturnType<typeof useStudioStore.getState>["e2eState"]
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[97vw] max-w-[97vw] sm:max-w-4xl h-[80vh] p-0 overflow-hidden">
        <div className="h-full min-w-0 overflow-hidden flex flex-col">
          <DialogHeader className="px-5 py-4 border-b">
            <DialogTitle className="text-base">End-to-End Execution</DialogTitle>
            <DialogDescription className="text-xs">
              {e2e
                ? e2e.status === "skipped"
                  ? "E2E execution was skipped"
                  : `${e2e.command || e2e.entryPoint || "unknown"} · Attempt ${e2e.attempt + 1}/${e2e.maxAttempts + 1}`
                : "Not started"}
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 min-h-0 overflow-auto">
            {!e2e ? (
              <div className="flex items-center justify-center h-full text-zinc-400 text-sm">
                End-to-end execution has not started yet.
              </div>
            ) : (
              <div className="p-5 space-y-4">
                {/* Status */}
                <div className="flex items-center gap-3">
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs",
                      e2e.status === "passed"
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                        : e2e.status === "skipped"
                          ? "bg-zinc-50 text-zinc-600 border-zinc-300"
                          : e2e.status === "failed"
                            ? "bg-red-50 text-red-700 border-red-200"
                            : e2e.status === "repairing"
                              ? "bg-amber-50 text-amber-700 border-amber-200"
                              : "bg-blue-50 text-blue-700 border-blue-200",
                    )}
                  >
                    {e2e.status}
                  </Badge>
                  {e2e.lastExitCode !== null && (
                    <span className="text-xs text-zinc-400">
                      Exit code: {e2e.lastExitCode}
                    </span>
                  )}
                </div>

                {/* stdout */}
                {e2e.lastStdout && (
                  <div>
                    <div className="text-xs font-medium text-zinc-600 mb-1">Output</div>
                    <div className="rounded-lg border border-zinc-700 bg-zinc-950 p-3 font-mono text-xs text-zinc-300 max-h-[40vh] overflow-auto whitespace-pre-wrap">
                      {e2e.lastStdout}
                    </div>
                  </div>
                )}

                {/* stderr */}
                {e2e.lastStderr && (
                  <div>
                    <div className="text-xs font-medium text-red-600 mb-1">Errors</div>
                    <div className="rounded-lg border border-red-200 bg-red-50 p-3 font-mono text-xs text-red-700 max-h-[20vh] overflow-auto whitespace-pre-wrap">
                      {e2e.lastStderr}
                    </div>
                  </div>
                )}

                {/* Repair history */}
                {e2e.history.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-zinc-600 mb-2">
                      Attempt History
                    </div>
                    <div className="space-y-2">
                      {e2e.history.map((h) => (
                        <div
                          key={h.attempt}
                          className="rounded-md border p-2 text-xs flex items-center gap-3"
                        >
                          <span className="font-medium">#{h.attempt + 1}</span>
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-[10px]",
                              h.success
                                ? "bg-emerald-50 text-emerald-700"
                                : "bg-red-50 text-red-700",
                            )}
                          >
                            {h.success ? "passed" : "failed"}
                          </Badge>
                          <span className="text-zinc-400">
                            exit={h.exitCode} · {(h.duration / 1000).toFixed(1)}s
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
