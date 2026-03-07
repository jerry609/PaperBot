"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useStudioStore, type AgentTask } from "@/lib/store/studio-store"
import { readSSE } from "@/lib/sse"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { WorkspaceSetupDialog } from "./WorkspaceSetupDialog"
import {
  Bot,
  CheckCircle2,
  Clock,
  Cpu,
  ExternalLink,
  Loader2,
  Play,
  Terminal,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { backendUrl } from "@/lib/backend-url"

const COLUMNS = [
  { id: "planning", label: "Planning", color: "border-t-yellow-500" },
  { id: "in_progress", label: "In Progress", color: "border-t-blue-500" },
  { id: "ai_review", label: "AI Review", color: "border-t-purple-500" },
  { id: "human_review", label: "Human Review", color: "border-t-orange-500" },
  { id: "done", label: "Done", color: "border-t-green-500" },
] as const
const RUN_ALL_TIMEOUT_MS = 10 * 60 * 1000

type ColumnId = (typeof COLUMNS)[number]["id"]
type HumanReviewDecision = "approve" | "request_changes"

interface Props {
  paperId: string | null
}

const LOG_LEVEL_STYLE: Record<string, string> = {
  info: "text-zinc-300",
  warning: "text-amber-300",
  error: "text-red-300",
  success: "text-emerald-300",
}

function toLogTimestamp(value?: string): string {
  if (!value) return "--:--:--"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return "--:--:--"
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
}

function taskStatusBadge(task: AgentTask) {
  if (task.status === "done") {
    return (
      <Badge variant="outline" className="text-[10px] bg-green-50 text-green-700 border-green-200">
        Done
      </Badge>
    )
  }
  if (task.status === "in_progress") {
    return (
      <Badge variant="outline" className="text-[10px] bg-blue-50 text-blue-700 border-blue-200">
        Running
      </Badge>
    )
  }
  if (task.status === "ai_review") {
    return (
      <Badge variant="outline" className="text-[10px] bg-purple-50 text-purple-700 border-purple-200">
        AI Review
      </Badge>
    )
  }
  if (task.status === "human_review") {
    return (
      <Badge variant="outline" className="text-[10px] bg-orange-50 text-orange-700 border-orange-200">
        Needs Review
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="text-[10px] bg-muted text-muted-foreground">
      Planning
    </Badge>
  )
}

function extractPossibleFiles(codexOutput?: string): string[] {
  if (!codexOutput) return []
  const matches = codexOutput.match(
    /(?:^|\s)([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|txt))/gm
  )
  if (!matches) return []
  const files = new Set(
    matches
      .map(item => item.trim())
      .map(item => item.replace(/^[^A-Za-z0-9_./-]+/, ""))
      .filter(Boolean)
  )
  return Array.from(files).slice(0, 50)
}

function joinWorkspacePath(workspaceDir: string, relativePath: string): string {
  const base = workspaceDir.replace(/[\\/]+$/, "")
  const rel = relativePath.replace(/^[\\/]+/, "")
  return `${base}/${rel}`
}

export function AgentBoard({ paperId }: Props) {
  const { agentTasks, boardSessionId, addAgentTask, updateAgentTask, papers, updatePaper } =
    useStudioStore()
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
  const [reviewNotes, setReviewNotes] = useState("")
  const [reviewSubmitting, setReviewSubmitting] = useState(false)
  const [reviewError, setReviewError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const tasksByColumn = useMemo(() => {
    const filtered = paperId ? agentTasks.filter(task => task.paperId === paperId) : agentTasks
    const map: Record<ColumnId, AgentTask[]> = {
      planning: [],
      in_progress: [],
      ai_review: [],
      human_review: [],
      done: [],
    }
    for (const task of filtered) {
      if (map[task.status]) {
        map[task.status].push(task)
      }
    }
    return map
  }, [agentTasks, paperId])

  const selectedTask = useMemo(
    () => (selectedTaskId ? agentTasks.find(task => task.id === selectedTaskId) || null : null),
    [agentTasks, selectedTaskId]
  )
  const selectedPaper = useMemo(
    () => (paperId ? papers.find(paper => paper.id === paperId) || null : null),
    [paperId, papers]
  )
  const workspaceDir = selectedPaper?.outputDir || null

  const planningCount = tasksByColumn.planning.length
  const totalTasks = agentTasks.filter(t => !paperId || t.paperId === paperId).length
  const doneCount = tasksByColumn.done.length

  useEffect(() => {
    if (!selectedTask || selectedTask.status !== "human_review") {
      setReviewNotes("")
      setReviewError(null)
    }
  }, [selectedTask])

  const appendInlineTaskLog = (
    taskId: string,
    event: string,
    phase: string,
    message: string,
    level: "info" | "warning" | "error" | "success" = "info"
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
    const existing = useStudioStore.getState().agentTasks.some(task => task.id === taskId)
    const status = (rawTask.status as AgentTask["status"]) || "planning"
    const assignee = (rawTask.assignee as string) || "claude"
    const progress = (rawTask.progress as number) || 0
    const subtasks = (rawTask.subtasks as AgentTask["subtasks"]) || []
    const codexOutput =
      (rawTask.codexOutput as string) || (rawTask.codex_output as string) || undefined
    const generatedFiles =
      (rawTask.generatedFiles as string[]) || (rawTask.generated_files as string[]) || undefined
    const reviewFeedback =
      (rawTask.reviewFeedback as string) || (rawTask.review_feedback as string) || undefined
    const lastError = (rawTask.lastError as string) || (rawTask.last_error as string) || undefined
    const executionLog =
      (rawTask.executionLog as AgentTask["executionLog"]) ||
      (rawTask.execution_log as AgentTask["executionLog"]) ||
      undefined
    const humanReviews =
      (rawTask.humanReviews as AgentTask["humanReviews"]) ||
      (rawTask.human_reviews as AgentTask["humanReviews"]) ||
      undefined

    if (existing) {
      const updates: Partial<AgentTask> = { status, assignee, progress, subtasks }
      if (codexOutput !== undefined) updates.codexOutput = codexOutput
      if (generatedFiles !== undefined) updates.generatedFiles = generatedFiles
      if (reviewFeedback !== undefined) updates.reviewFeedback = reviewFeedback
      if (lastError !== undefined) updates.lastError = lastError
      if (executionLog !== undefined) updates.executionLog = executionLog
      if (humanReviews !== undefined) updates.humanReviews = humanReviews
      updateAgentTask(taskId, updates)
      return
    }

    addAgentTask({
      id: taskId,
      title: (rawTask.title as string) || "Untitled",
      description: (rawTask.description as string) || "",
      status,
      assignee,
      progress,
      tags: (rawTask.tags as string[]) || [],
      subtasks,
      codexOutput,
      generatedFiles: generatedFiles || [],
      reviewFeedback,
      lastError,
      executionLog: executionLog || [],
      humanReviews: humanReviews || [],
      paperId: (rawTask.paperId as string) || (rawTask.paper_id as string) || paperId || undefined,
    })
  }

  const runAllWithWorkspace = async (targetWorkspaceDir: string) => {
    if (!boardSessionId || running) return
    setRunning(true)
    setRunError(null)

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), RUN_ALL_TIMEOUT_MS)

    try {
      const res = await fetch(
        backendUrl(`/api/agent-board/sessions/${boardSessionId}/run`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_dir: targetWorkspaceDir }),
          signal: controller.signal,
        }
      )
      if (!res.ok || !res.body) throw new Error(`Run failed (${res.status})`)

      for await (const evt of readSSE(res.body)) {
        if (controller.signal.aborted) break

        if (evt?.type === "progress") {
          const data = (evt.data ?? {}) as Record<string, unknown>
          const eventName = (data.event as string) || "progress"
          const taskData = data.task as Record<string, unknown> | undefined
          const taskId = (data.task_id as string) || (taskData?.id as string)

          if (taskId && taskData) {
            upsertTaskFromEvent(taskId, taskData)
          } else if (taskId && eventName === "task_codex_done") {
            updateAgentTask(taskId, { status: "in_progress", progress: 70 })
            const outputLength = data.output_length as number | undefined
            appendInlineTaskLog(
              taskId,
              "task_codex_done",
              "codex_running",
              `Codex output received${outputLength ? ` (${outputLength} chars)` : ""}.`,
              "success"
            )
          }

          if (taskId && !taskData && eventName === "task_failed") {
            const error = (data.error as string) || "Task failed."
            updateAgentTask(taskId, { status: "human_review", lastError: error, progress: 100 })
            appendInlineTaskLog(taskId, "task_failed", "codex_running", error, "error")
          }
        } else if (evt?.type === "result") {
          break
        } else if (evt?.type === "error") {
          setRunError(evt.message || "Execution failed")
          break
        }
      }
    } catch (err) {
      if (controller.signal.aborted) {
        setRunError("Run timed out")
      } else {
        setRunError(err instanceof Error ? err.message : "Run failed")
      }
    } finally {
      clearTimeout(timeout)
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

  const handleHumanReviewDecision = async (decision: HumanReviewDecision) => {
    if (!selectedTask || selectedTask.status !== "human_review") return
    setReviewSubmitting(true)
    setReviewError(null)
    try {
      const res = await fetch(
        backendUrl(`/api/agent-board/tasks/${selectedTask.id}/human-review`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decision, notes: reviewNotes.trim() || undefined }),
        }
      )
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Review action failed (${res.status})`)
      }
      const updated = (await res.json()) as Record<string, unknown>
      upsertTaskFromEvent(selectedTask.id, updated)
      setReviewNotes("")
      if (decision === "approve") {
        setSelectedTaskId(null)
      }
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "Review action failed")
    } finally {
      setReviewSubmitting(false)
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-3 border-b flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold">Agent Board</h2>
          {totalTasks > 0 && (
            <span className="text-xs text-muted-foreground">
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
          {planningCount > 0 && (
            <Button
              size="sm"
              onClick={handleRunAll}
              disabled={running || !boardSessionId}
              className="h-7 text-xs gap-1.5"
            >
              {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {running ? "Running..." : `Run All (${planningCount})`}
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 flex overflow-x-auto p-4 gap-4 min-h-0">
        {COLUMNS.map(col => (
          <div key={col.id} className="flex-shrink-0 w-72 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold">{col.label}</h3>
                <span className="text-xs text-muted-foreground bg-muted rounded-full px-2 py-0.5">
                  {tasksByColumn[col.id].length}
                </span>
              </div>
            </div>

            <div
              className={cn(
                "flex-1 space-y-3 overflow-y-auto rounded-lg border-t-2 bg-muted/20 p-2",
                col.color
              )}
            >
              {tasksByColumn[col.id].length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                  {col.id === "in_progress" ? (
                    <>
                      <Cpu className="h-5 w-5 mb-2 opacity-30 animate-pulse" />
                      <p className="text-xs">Nothing running</p>
                      <p className="text-[10px]">Click &ldquo;Run All&rdquo; to start</p>
                    </>
                  ) : col.id === "ai_review" ? (
                    <>
                      <Bot className="h-5 w-5 mb-2 opacity-30" />
                      <p className="text-xs">No tasks in review</p>
                      <p className="text-[10px]">Claude reviews Codex output</p>
                    </>
                  ) : (
                    <p className="text-xs">No tasks</p>
                  )}
                </div>
              ) : (
                tasksByColumn[col.id].map(task => (
                  <TaskCard key={task.id} task={task} onClick={() => setSelectedTaskId(task.id)} />
                ))
              )}
            </div>
          </div>
        ))}
      </div>

      <TaskDetailDialog
        task={selectedTask}
        workspaceDir={workspaceDir}
        open={!!selectedTask}
        onOpenChange={open => !open && setSelectedTaskId(null)}
        reviewNotes={reviewNotes}
        onReviewNotesChange={setReviewNotes}
        onReviewDecision={handleHumanReviewDecision}
        reviewSubmitting={reviewSubmitting}
        reviewError={reviewError}
      />

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

function TaskCard({ task, onClick }: { task: AgentTask; onClick: () => void }) {
  const completedSubtasks = task.subtasks.filter(sub => sub.done).length
  const totalSubtasks = task.subtasks.length

  const [relativeTime, setRelativeTime] = useState("")
  useEffect(() => {
    const compute = () => {
      const diff = Date.now() - new Date(task.updatedAt).getTime()
      const mins = Math.floor(diff / 60000)
      if (mins < 60) {
        setRelativeTime(`${mins}m ago`)
        return
      }
      const hours = Math.floor(mins / 60)
      if (hours < 24) {
        setRelativeTime(`${hours}h ago`)
        return
      }
      setRelativeTime(`${Math.floor(hours / 24)}d ago`)
    }
    compute()
    const id = setInterval(compute, 60_000)
    return () => clearInterval(id)
  }, [task.updatedAt])

  return (
    <Card
      className="shadow-sm cursor-pointer hover:border-primary/40 transition-colors"
      onClick={onClick}
      onKeyDown={event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          onClick()
        }
      }}
      role="button"
      tabIndex={0}
    >
      <CardContent className="p-3 space-y-2">
        <div className="flex items-start justify-between">
          <h4 className="text-sm font-medium leading-tight line-clamp-2">{task.title}</h4>
          <div className="flex gap-1 shrink-0 ml-2">{taskStatusBadge(task)}</div>
        </div>

        {task.description && <p className="text-xs text-muted-foreground line-clamp-2">{task.description}</p>}

        {task.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {task.tags.map(tag => (
              <Badge key={tag} variant="secondary" className="text-[10px]">
                {tag}
              </Badge>
            ))}
          </div>
        )}

        {task.progress > 0 && task.progress < 100 && (
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Progress</span>
              <span>{task.progress}%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${task.progress}%` }} />
            </div>
          </div>
        )}

        {totalSubtasks > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            {task.subtasks.slice(0, 10).map(sub => (
              <div
                key={sub.id}
                className={cn("h-2 w-2 rounded-full", sub.done ? "bg-green-500" : "bg-muted-foreground/30")}
                title={sub.title}
              />
            ))}
            {totalSubtasks > 10 && <span className="text-[10px] text-muted-foreground">+{totalSubtasks - 10}</span>}
            <span className="text-[10px] text-muted-foreground ml-1">
              {completedSubtasks}/{totalSubtasks}
            </span>
          </div>
        )}

        <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-1">
          <div className="flex items-center gap-1">
            {task.assignee === "claude" ? <Bot className="h-3 w-3" /> : <Cpu className="h-3 w-3" />}
            <span>{task.assignee}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-2.5 w-2.5" />
            <span>{relativeTime}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function TaskDetailDialog({
  task,
  workspaceDir,
  open,
  onOpenChange,
  reviewNotes,
  onReviewNotesChange,
  onReviewDecision,
  reviewSubmitting,
  reviewError,
}: {
  task: AgentTask | null
  workspaceDir: string | null
  open: boolean
  onOpenChange: (open: boolean) => void
  reviewNotes: string
  onReviewNotesChange: (value: string) => void
  onReviewDecision: (decision: HumanReviewDecision) => Promise<void>
  reviewSubmitting: boolean
  reviewError: string | null
}) {
  const logs = task?.executionLog || []
  const files = task?.generatedFiles?.length ? task.generatedFiles : extractPossibleFiles(task?.codexOutput)
  const reviewDocPath = files.find(path => /(?:^|\/)reviews\/.+-user-review\.md$/.test(path))

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[97vw] max-w-6xl h-[90vh] p-0 overflow-hidden">
        {task && (
          <div className="h-full min-w-0 overflow-hidden flex flex-col">
            <DialogHeader className="px-5 py-4 border-b">
              <div className="flex items-start justify-between gap-2">
                <div className="space-y-1 min-w-0">
                  <DialogTitle className="text-base leading-tight truncate">{task.title}</DialogTitle>
                  <DialogDescription className="text-xs">
                    {task.id} · {task.assignee} · {task.progress}% complete
                  </DialogDescription>
                </div>
                <div className="flex items-center gap-2">
                  {task.status === "human_review" && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 text-xs"
                      onClick={() => {
                        if (workspaceDir) {
                          window.open(`vscode://file${workspaceDir}`, "_blank")
                        }
                      }}
                      disabled={!workspaceDir}
                      title={workspaceDir ? `Open ${workspaceDir} in VS Code` : "Set up workspace first"}
                    >
                      <ExternalLink className="h-3.5 w-3.5 mr-1" />
                      Open in VS Code
                    </Button>
                  )}
                  {taskStatusBadge(task)}
                </div>
              </div>
            </DialogHeader>

            <Tabs defaultValue="logs" className="flex-1 min-h-0 min-w-0 overflow-hidden">
              <div className="px-5 pt-3 shrink-0">
                <TabsList>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="subtasks">Subtasks ({task.subtasks.length})</TabsTrigger>
                  <TabsTrigger value="logs">Logs</TabsTrigger>
                  <TabsTrigger value="files">Files</TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="overview" className="px-5 pb-5 min-w-0 overflow-auto">
                <div className="space-y-4">
                  {task.description && (
                    <div className="rounded-md border bg-muted/30 p-3 text-sm">{task.description}</div>
                  )}

                  {task.reviewFeedback && (
                    <div className="rounded-md border p-3">
                      <div className="text-xs font-medium mb-1">Review Feedback</div>
                      <div className="text-xs text-muted-foreground whitespace-pre-wrap">{task.reviewFeedback}</div>
                    </div>
                  )}

                  {task.lastError && (
                    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-xs text-red-700">
                      {task.lastError}
                    </div>
                  )}

                  {task.status === "human_review" && (
                    <div className="rounded-md border p-3 space-y-3">
                      <div className="text-sm font-medium">Human Review</div>
                      <p className="text-xs text-muted-foreground">
                        Review the logs/output and decide whether to approve this task or send it back to planning.
                      </p>
                      {reviewDocPath && (
                        <div className="rounded border bg-muted/30 p-2 space-y-2">
                          <p className="text-[11px] text-muted-foreground">
                            Review document:
                          </p>
                          <div className="text-xs font-mono break-all">{reviewDocPath}</div>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              if (workspaceDir) {
                                window.open(
                                  `vscode://file${joinWorkspacePath(workspaceDir, reviewDocPath)}`,
                                  "_blank"
                                )
                              }
                            }}
                            disabled={!workspaceDir || reviewSubmitting}
                            className="h-7 text-xs"
                          >
                            <ExternalLink className="h-3.5 w-3.5 mr-1" />
                            Open Review Doc
                          </Button>
                        </div>
                      )}
                      <Textarea
                        value={reviewNotes}
                        onChange={event => onReviewNotesChange(event.target.value)}
                        placeholder="Optional review notes for the agent..."
                        className="min-h-20 text-xs"
                      />
                      {reviewError && <p className="text-xs text-red-600">{reviewError}</p>}
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            if (workspaceDir) {
                              window.open(`vscode://file${workspaceDir}`, "_blank")
                            }
                          }}
                          disabled={!workspaceDir || reviewSubmitting}
                          className="h-7 text-xs"
                          title={workspaceDir ? `Open ${workspaceDir} in VS Code` : "Set up workspace first"}
                        >
                          <ExternalLink className="h-3.5 w-3.5 mr-1" />
                          Open in VS Code
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => void onReviewDecision("approve")}
                          disabled={reviewSubmitting}
                          className="h-7 text-xs"
                        >
                          {reviewSubmitting ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5 mr-1" />}
                          Approve
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void onReviewDecision("request_changes")}
                          disabled={reviewSubmitting}
                          className="h-7 text-xs"
                        >
                          Request Changes
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="subtasks" className="px-5 pb-5 min-w-0 overflow-auto">
                {task.subtasks.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No subtasks.</p>
                ) : (
                  <div className="space-y-2">
                    {task.subtasks.map(subtask => (
                      <div key={subtask.id} className="rounded-md border px-3 py-2 flex items-center gap-2">
                        <span
                          className={cn(
                            "h-2.5 w-2.5 rounded-full",
                            subtask.done ? "bg-green-500" : "bg-muted-foreground/40"
                          )}
                        />
                        <span className="text-sm">{subtask.title}</span>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent
                value="logs"
                className="px-5 pb-5 min-h-0 min-w-0 w-full overflow-hidden data-[state=active]:flex data-[state=inactive]:hidden"
              >
                <div className="flex-1 min-h-0 w-full rounded-md border border-zinc-700 overflow-hidden min-w-0 flex flex-col">
                  <div className="px-3 py-2 border-b border-zinc-700 text-xs font-medium flex items-center gap-1.5 bg-zinc-50">
                    <Terminal className="h-3.5 w-3.5" />
                    Execution Log
                  </div>
                  <div className="flex-1 min-h-0 w-full bg-zinc-950 overflow-auto">
                    <div className="p-3 pr-4 font-mono text-xs space-y-1 min-w-0 max-w-full">
                      {logs.length === 0 ? (
                        <p className="text-zinc-500">No execution logs yet.</p>
                      ) : (
                        logs.map(log => (
                          <div
                            key={log.id}
                            className={cn(
                              LOG_LEVEL_STYLE[log.level] || LOG_LEVEL_STYLE.info,
                              "max-w-full whitespace-pre-wrap break-all leading-5"
                            )}
                          >
                            [{toLogTimestamp(log.timestamp)}] [{log.phase}/{log.event}] {log.message}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="files" className="px-5 pb-5 min-w-0 overflow-auto">
                {files.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No file list detected from current Codex output.
                  </p>
                ) : (
                  <div className="space-y-1">
                    {files.map(file => (
                      <div key={file} className="text-xs font-mono rounded bg-muted px-2 py-1">
                        {file}
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
