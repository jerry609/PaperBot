"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { useStudioStore, AgentAction, type AgentTask as StudioAgentTask } from "@/lib/store/studio-store"
import { useProjectContext } from "@/lib/store/project-context"
import { readSSE } from "@/lib/sse"
import { backendUrl } from "@/lib/backend-url"
import { CodeBlock } from "@/components/ai-elements"
import { DiffModal } from "./DiffViewer"
import { WorkspaceSetupDialog } from "./WorkspaceSetupDialog"
import { ContextDialogPanel } from "./ContextDialogPanel"
import { AgentBoard } from "./AgentBoard"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import type { StudioRuntimeInfo } from "@/lib/studio-runtime"
import { cn } from "@/lib/utils"
import {
    CheckCircle2,
    AlertCircle,
    FileText,
    Bot,
    FileCode,
    Wrench,
    Terminal,
    ChevronDown,
    ChevronRight,
    Clock,
    Loader2,
    X,
    Save,
    Send,
    Code,
    Activity,
    MessageSquare,
    LayoutDashboard,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import Editor from "@monaco-editor/react"
import type { ContextPackSession } from "@/lib/types/p2c"

type StepStatus = "idle" | "running" | "success" | "error"
type Mode = "Code" | "Plan" | "Ask"
export type ReproductionViewMode = "log" | "context" | "board"

interface ReproductionLogProps {
    viewMode: ReproductionViewMode
    onViewModeChange: (mode: ReproductionViewMode) => void
    hideNavigation?: boolean
    onOpenBoardWorkspace?: () => void
    runtimeInfo: StudioRuntimeInfo
    runtimeLoading: boolean
}

const actionIcons: Record<string, React.ElementType> = {
    thinking: Loader2,
    file_change: FileCode,
    function_call: Wrench,
    error: AlertCircle,
    complete: CheckCircle2,
    text: Bot,
    run_command: Terminal,
}

const actionColors: Record<string, { bg: string; text: string; border: string }> = {
    thinking: { bg: "bg-slate-100", text: "text-slate-700", border: "border-slate-200" },
    file_change: { bg: "bg-slate-100", text: "text-slate-700", border: "border-slate-200" },
    function_call: { bg: "bg-stone-100", text: "text-stone-700", border: "border-stone-200" },
    error: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200" },
    complete: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
    text: { bg: "bg-[#eef0ea]", text: "text-slate-800", border: "border-slate-200" },
}

function normalizeBoardTaskStatus(rawStatus: unknown): StudioAgentTask["status"] {
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

function normalizeBoardTaskFromBackend(rawTask: Record<string, unknown>, fallbackPaperId: string | null): StudioAgentTask {
    return {
        id: (rawTask.id as string) || `task-${Date.now()}`,
        title: (rawTask.title as string) || "Untitled",
        description: (rawTask.description as string) || "",
        status: normalizeBoardTaskStatus(rawTask.status),
        assignee: (rawTask.assignee as string) || "claude",
        progress: (rawTask.progress as number) || 0,
        tags: (rawTask.tags as string[]) || [],
        subtasks: (rawTask.subtasks as StudioAgentTask["subtasks"]) || [],
        codexOutput: (rawTask.codexOutput as string) || (rawTask.codex_output as string) || undefined,
        generatedFiles: (rawTask.generatedFiles as string[]) || (rawTask.generated_files as string[]) || [],
        reviewFeedback:
            (rawTask.reviewFeedback as string) || (rawTask.review_feedback as string) || undefined,
        lastError: (rawTask.lastError as string) || (rawTask.last_error as string) || undefined,
        executionLog:
            (rawTask.executionLog as StudioAgentTask["executionLog"]) ||
            (rawTask.execution_log as StudioAgentTask["executionLog"]) ||
            [],
        paperId: (rawTask.paperId as string) || (rawTask.paper_id as string) || fallbackPaperId || undefined,
        createdAt: (rawTask.createdAt as string) || (rawTask.created_at as string) || new Date().toISOString(),
        updatedAt: (rawTask.updatedAt as string) || (rawTask.updated_at as string) || new Date().toISOString(),
    }
}

function buildCodexTaskTitle(message: string): string {
    const singleLine = message.replace(/\s+/g, " ").trim()
    if (!singleLine) return "Studio Codex task"
    return singleLine.length <= 72 ? singleLine : `${singleLine.slice(0, 69)}...`
}

function effectivePermissionMode(
    mode: Mode,
    codeModeEnabled: boolean | null,
): "acceptEdits" | "plan" | "default" {
    if (mode === "Code") return codeModeEnabled === false ? "plan" : "acceptEdits"
    if (mode === "Plan") return "plan"
    return "default"
}

function buildStudioCommandPreview(
    runtimeInfo: StudioRuntimeInfo,
    mode: Mode,
    model: string,
): string {
    if (runtimeInfo.source === "anthropic_api") {
        return "Fallback path: direct Anthropic API call, not Claude Code CLI"
    }
    if (runtimeInfo.source !== "claude_code") {
        return "Resolving Claude Code CLI surface..."
    }

    const normalizedModel = model.trim() || "sonnet"
    const permissionMode = effectivePermissionMode(mode, runtimeInfo.codeModeEnabled)
    const parts = ["claude", "--model", normalizedModel]

    if (permissionMode !== "default") {
        parts.push("--permission-mode", permissionMode)
    }

    parts.push("-p", "--output-format", "stream-json", "--verbose")
    return parts.join(" ")
}

function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

interface ActionItemProps {
    action: AgentAction
    onViewDiff: (action: AgentAction) => void
    isLast: boolean
}

function ActionItem({ action, onViewDiff, isLast }: ActionItemProps) {
    const [expanded, setExpanded] = useState(false)
    const iconKey = action.metadata?.functionName || action.type
    const Icon = actionIcons[iconKey] || actionIcons[action.type] || Bot
    const colors = actionColors[iconKey] || actionColors[action.type] || actionColors.text

    const hasExpandableContent = Boolean(action.metadata?.params || action.metadata?.result)
    const stringifyPayload = (payload: unknown): string =>
        typeof payload === "string" ? payload : JSON.stringify(payload, null, 2) || ""

    return (
        <div className="relative flex gap-2.5">
            {!isLast && (
                <div className="absolute left-2.5 top-6 bottom-0 w-px bg-slate-200" />
            )}

            <div className={cn(
                "relative z-10 w-5 h-5 flex items-center justify-center shrink-0 rounded-md border",
                colors.bg, colors.border
            )}>
                <Icon className={cn("h-2.5 w-2.5", colors.text)} />
            </div>

            <div className="flex-1 min-w-0 pb-3">
                <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                        {action.type === 'file_change' && action.metadata?.filename ? (
                            <div className="space-y-0.5">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <button
                                        onClick={() => onViewDiff(action)}
                                        className={cn("font-mono text-xs hover:underline", colors.text)}
                                    >
                                        {action.metadata.filename}
                                    </button>
                                    <span className="rounded bg-[#e5e8e1] px-1 py-0.5 text-[10px]">
                                        <span className="text-emerald-700">+{action.metadata.linesAdded || 0}</span>
                                        <span className="mx-0.5 text-slate-400">/</span>
                                        <span className="text-rose-700">-{action.metadata.linesDeleted || 0}</span>
                                    </span>
                                </div>
                            </div>
                        ) : action.type === 'function_call' && action.metadata?.functionName ? (
                            <div className="space-y-0.5">
                                <div className="flex items-center gap-2">
                                    <code className={cn("text-[10px] font-mono px-1 py-0.5 rounded", colors.bg, colors.text)}>
                                        {action.metadata.functionName}()
                                    </code>
                                    {hasExpandableContent && (
                                        <button
                                            onClick={() => setExpanded(!expanded)}
                                            className="text-slate-500 transition-colors hover:text-slate-700"
                                        >
                                            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                        </button>
                                    )}
                                </div>
                                {expanded && (
                                    <div className="mt-1.5 space-y-1.5">
                                        {Boolean(action.metadata.params) && (
                                            <CodeBlock title="Args" code={stringifyPayload(action.metadata.params)} />
                                        )}
                                        {Boolean(action.metadata.result) && (
                                            <CodeBlock title="Result" code={stringifyPayload(action.metadata.result)} />
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : action.type === 'error' ? (
                            <div className={cn("text-xs rounded-md border px-2 py-1.5", colors.bg, colors.border)}>
                                <span className={colors.text}>{action.content}</span>
                            </div>
                        ) : action.type === 'complete' ? (
                            <span className={cn("text-xs font-medium", colors.text)}>Completed</span>
                        ) : (
                            <p className="whitespace-pre-wrap text-xs leading-relaxed text-slate-700">{action.content}</p>
                        )}
                    </div>

                    <div className="flex shrink-0 items-center gap-1 text-[9px] text-slate-400">
                        <Clock className="h-2 w-2" />
                        {formatTime(action.timestamp)}
                    </div>
                </div>
            </div>
        </div>
    )
}

export function ReproductionLog({
    viewMode,
    onViewModeChange,
    hideNavigation = false,
    onOpenBoardWorkspace,
    runtimeInfo,
    runtimeLoading,
}: ReproductionLogProps) {
    const router = useRouter()
    const {
        papers,
        tasks,
        activeTaskId,
        selectedPaperId,
        boardSessionId,
        lastGenCodeResult,
        contextPack,
        contextPackLoading,
        contextPackError,
        generationProgress,
        liveObservations,
        addTask,
        addAction,
        appendToLastAction,
        updateTaskStatus,
        updatePaper,
        setBoardSessionId,
        addAgentTask,
        updateAgentTask,
        setPipelinePhase,
    } = useStudioStore()

    const { generate: generateContextPack, status: genStatus } = useContextPackGeneration()
    const { files, activeFile, updateFile, setActiveFile } = useProjectContext()
    const activeFileData = activeFile ? files[activeFile] : null

    const selectedPaper = useMemo(() =>
        selectedPaperId ? papers.find(p => p.id === selectedPaperId) ?? null : null,
        [papers, selectedPaperId]
    )

    const [status, setStatus] = useState<StepStatus>("idle")
    const [mode, setMode] = useState<Mode>("Code")
    const [model, setModel] = useState("sonnet")
    const [lastError, setLastError] = useState<string | null>(null)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)
    const [saving, setSaving] = useState(false)
    const [messageInput, setMessageInput] = useState("")
    const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
    const [pendingAction, setPendingAction] = useState<"chat" | "delegate_codex" | null>(null)
    // Switch to context dialog when generation starts.
    useEffect(() => {
        if (contextPackLoading && viewMode !== "context") {
            onViewModeChange("context")
        }
    }, [contextPackLoading, onViewModeChange, viewMode])

    // Do not auto-switch away from "context"; keep it open until the user changes tabs.

    const activeTask = tasks.find(t => t.id === activeTaskId)
    const projectDir = selectedPaper?.outputDir || lastGenCodeResult?.outputDir || null
    const isBusy = status === "running"
    const runtimeLabel = runtimeLoading
        ? "Studio runtime"
        : runtimeInfo.source === "anthropic_api"
            ? "Anthropic API fallback"
            : "Claude Code"
    const commandPreview = useMemo(
        () => buildStudioCommandPreview(runtimeInfo, mode, model),
        [runtimeInfo, mode, model],
    )
    const messagePlaceholder = runtimeLoading
        ? "Message Studio runtime..."
        : runtimeInfo.source === "anthropic_api"
            ? "Message fallback runtime..."
            : "Message Claude Code..."

    const saveActiveFile = async () => {
        if (!projectDir || !activeFile || !activeFileData) return
        setSaving(true)
        setLastError(null)
        try {
            const res = await fetch(`/api/runbook/file`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ project_dir: projectDir, path: activeFileData.name, content: activeFileData.content }),
            })
            if (!res.ok) {
                const text = await res.text()
                throw new Error(`Failed to save (${res.status}): ${text}`)
            }
        } catch (e) {
            setLastError(e instanceof Error ? e.message : String(e))
        } finally {
            setSaving(false)
        }
    }

    const handleWorkspaceConfirm = (directory: string) => {
        setShowWorkspaceSetup(false)
        if (selectedPaperId) {
            updatePaper(selectedPaperId, { outputDir: directory })
        }
        if (pendingAction === "chat") {
            runChatWithDir(directory)
        } else if (pendingAction === "delegate_codex") {
            runCodexDelegationWithDir(directory)
        }
        setPendingAction(null)
    }

    const runChatWithDir = async (targetDir: string) => {
        // Chat with specified directory - called after workspace setup
        const message = messageInput.trim()
        if (!message) return
        setMessageInput("")
        await handleSendMessageWithDir(message, targetDir)
    }

    const runCodexDelegationWithDir = async (targetDir: string) => {
        const message = messageInput.trim()
        if (!message) return
        setMessageInput("")
        await handleDelegateToCodexWithDir(message, targetDir)
    }

    const handleSendMessage = async () => {
        if (!messageInput.trim() || isBusy) return

        // For Code mode, require a project directory
        if (mode === "Code" && !projectDir) {
            if (!selectedPaper) {
                setLastError("Select or create a paper first.")
                return
            }
            setPendingAction("chat")
            setShowWorkspaceSetup(true)
            return
        }

        const message = messageInput.trim()
        setMessageInput("")
        await handleSendMessageWithDir(message, projectDir || undefined)
    }

    const ensureBoardSession = async (targetDir: string): Promise<string> => {
        if (!selectedPaperId || !selectedPaper) {
            throw new Error("Select or create a paper first.")
        }

        if (boardSessionId) {
            return boardSessionId
        }

        const response = await fetch(backendUrl("/api/agent-board/sessions"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                paper_id: selectedPaperId,
                context_pack_id: contextPack?.context_pack_id || "",
                paper_title: selectedPaper.title,
                workspace_dir: targetDir,
            }),
        })

        if (!response.ok) {
            const text = await response.text()
            throw new Error(text || `Failed to create monitor session (${response.status})`)
        }

        const session = (await response.json()) as { session_id?: string }
        if (!session.session_id) {
            throw new Error("Monitor session response did not include a session_id")
        }

        setBoardSessionId(session.session_id)
        return session.session_id
    }

    const handleSendMessageWithDir = async (message: string, targetDir?: string) => {
        setStatus("running")
        setLastError(null)
        onViewModeChange("log")

        const taskId = addTask(`Chat — ${message.slice(0, 30)}${message.length > 30 ? "…" : ""}`)
        addAction(taskId, { type: "thinking", content: `[${mode}] Sending to ${runtimeLabel}...` })

        try {
            const res = await fetch("/api/studio/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message,
                    mode,
                    model,
                    paper: selectedPaper ? {
                        title: selectedPaper.title,
                        abstract: selectedPaper.abstract,
                        method_section: selectedPaper.methodSection,
                    } : undefined,
                    project_dir: targetDir,
                    context_pack_id: contextPack?.context_pack_id,
                }),
            })

            if (!res.ok || !res.body) {
                throw new Error(`Failed to send message (${res.status})`)
            }

            updateTaskStatus(taskId, "running")

            // Track whether the last action is a text block so we can
            // append to it (producing one continuous bubble) instead of
            // creating a new action per chunk.
            let lastActionIsText = false

            for await (const evt of readSSE(res.body)) {
                if (evt?.type === "progress") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const cliEvent = data.cli_event as string | undefined

                    if (cliEvent === "text") {
                        // Streaming text — append to current text bubble
                        const text = (data.text as string) || ""
                        if (text) {
                            if (lastActionIsText) {
                                appendToLastAction(taskId, text)
                            } else {
                                addAction(taskId, { type: "text", content: text })
                                lastActionIsText = true
                            }
                        }
                    } else if (cliEvent === "tool_use") {
                        lastActionIsText = false
                        addAction(taskId, {
                            type: "function_call",
                            content: `${data.tool_name}()`,
                            metadata: {
                                functionName: data.tool_name as string,
                                params: data.tool_input as Record<string, unknown>,
                            },
                        })
                    } else if (cliEvent === "tool_result") {
                        // Attach result to the most recent function_call action
                        lastActionIsText = false
                        addAction(taskId, {
                            type: "function_call",
                            content: `${data.tool_name}() result`,
                            metadata: {
                                functionName: data.tool_name as string,
                                result: data.content as string,
                            },
                        })
                    } else if (cliEvent === "thinking") {
                        lastActionIsText = false
                        addAction(taskId, { type: "thinking", content: (data.text as string) || "Thinking..." })
                    } else if (data.keepalive) {
                        // Keepalive heartbeat — ignore
                    } else if (data.message) {
                        // Legacy status messages (e.g. "Connecting to Claude CLI...")
                        lastActionIsText = false
                        addAction(taskId, { type: "thinking", content: data.message as string })
                    } else if (data.delta) {
                        // Fallback: legacy plain-text streaming (API fallback path)
                        const text = data.delta as string
                        if (lastActionIsText) {
                            appendToLastAction(taskId, text)
                        } else {
                            addAction(taskId, { type: "text", content: text })
                            lastActionIsText = true
                        }
                    }
                } else if (evt?.type === "result") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const summary = data.num_turns
                        ? `Completed in ${data.num_turns} turns`
                        : "Completed"
                    addAction(taskId, { type: "complete", content: summary })
                    updateTaskStatus(taskId, "completed")
                    setStatus("success")
                } else if (evt?.type === "error") {
                    addAction(taskId, { type: "error", content: evt.message || "Chat failed" })
                    updateTaskStatus(taskId, "error")
                    setLastError(evt.message || "Chat failed")
                    setStatus("error")
                    return
                }
            }
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            addAction(taskId, { type: "error", content: msg })
            updateTaskStatus(taskId, "error")
            setLastError(msg)
            setStatus("error")
        }
    }

    const handleDelegateToCodex = async () => {
        if (!messageInput.trim() || isBusy) return

        if (!selectedPaper) {
            setLastError("Select or create a paper first.")
            return
        }

        if (!projectDir) {
            setPendingAction("delegate_codex")
            setShowWorkspaceSetup(true)
            return
        }

        const message = messageInput.trim()
        setMessageInput("")
        await handleDelegateToCodexWithDir(message, projectDir)
    }

    const handleDelegateToCodexWithDir = async (message: string, targetDir: string) => {
        setStatus("running")
        setLastError(null)

        const taskTitle = buildCodexTaskTitle(message)
        const logTaskId = addTask(`Codex — ${taskTitle}`)
        addAction(logTaskId, { type: "thinking", content: "Preparing Codex delegation..." })

        try {
            const sessionId = await ensureBoardSession(targetDir)
            const createRes = await fetch(backendUrl(`/api/agent-board/sessions/${sessionId}/tasks`), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    title: taskTitle,
                    description: message,
                    workspace_dir: targetDir,
                    assignee: "codex",
                    tags: ["studio_console"],
                }),
            })

            if (!createRes.ok) {
                const text = await createRes.text()
                throw new Error(text || `Failed to create Codex task (${createRes.status})`)
            }

            const rawTask = (await createRes.json()) as Record<string, unknown>
            const task = normalizeBoardTaskFromBackend(rawTask, selectedPaperId)
            addAgentTask(task)
            addAction(logTaskId, {
                type: "text",
                content: `Delegated to Codex in session ${sessionId}. Open Monitor to inspect the live task graph.`,
            })
            updateTaskStatus(logTaskId, "running")
            setPipelinePhase("executing")

            const execRes = await fetch(
                backendUrl(`/api/agent-board/tasks/${encodeURIComponent(task.id)}/execute`),
                { method: "POST" },
            )

            if (!execRes.ok || !execRes.body) {
                const text = await execRes.text()
                throw new Error(text || `Failed to execute Codex task (${execRes.status})`)
            }

            for await (const evt of readSSE(execRes.body)) {
                if (evt?.type === "progress") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const taskData = data.task as Record<string, unknown> | undefined
                    if (taskData) {
                        updateAgentTask(task.id, normalizeBoardTaskFromBackend(taskData, selectedPaperId))
                    }

                    const eventName = data.event as string | undefined
                    if (eventName === "task_failed") {
                        addAction(logTaskId, {
                            type: "error",
                            content: (data.error as string) || "Codex task failed",
                        })
                    } else if (eventName === "task_reviewed" && typeof data.feedback === "string" && data.feedback.trim()) {
                        addAction(logTaskId, {
                            type: "text",
                            content: `Claude review: ${data.feedback as string}`,
                        })
                    } else if (eventName === "task_reviewing") {
                        addAction(logTaskId, {
                            type: "thinking",
                            content: "Claude is reviewing Codex output...",
                        })
                    }
                } else if (evt?.type === "result") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const success = Boolean(data.success)
                    addAction(logTaskId, {
                        type: "complete",
                        content: success ? "Codex task completed" : "Codex task finished and needs review",
                    })
                    updateTaskStatus(logTaskId, success ? "completed" : "error")
                    setStatus(success ? "success" : "error")
                    setPipelinePhase(success ? "completed" : "idle")
                    return
                } else if (evt?.type === "error") {
                    throw new Error(evt.message || "Codex task failed")
                }
            }

            throw new Error("Codex task stream ended before a terminal result was received")
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            addAction(logTaskId, { type: "error", content: msg })
            updateTaskStatus(logTaskId, "error")
            setLastError(msg)
            setStatus("error")
            setPipelinePhase("failed")
            return
        }
    }

    const handleSessionCreated = (session: ContextPackSession) => {
        onViewModeChange("log")
        if (session.initial_prompt) {
            setMessageInput(session.initial_prompt)
        }
    }

    const openAgentBoardWorkspace = () => {
        if (onOpenBoardWorkspace) {
            onOpenBoardWorkspace()
            return
        }
        if (selectedPaperId) {
            router.push(`/studio?paperId=${encodeURIComponent(selectedPaperId)}&surface=board`)
            return
        }
        router.push("/studio?surface=board")
    }

    return (
        <div className="flex h-full min-h-0 w-full flex-1 flex-col bg-[#f5f5f2]">
            {/* Tab Navigation */}
            {!hideNavigation && (
                <div className="flex shrink-0 items-center border-b border-slate-200 bg-[#eef0ea]">
                    {([
                        { key: "context" as const, label: "Context", icon: Activity },
                        { key: "log" as const, label: "Chat", icon: MessageSquare },
                        { key: "board" as const, label: "Monitor", icon: LayoutDashboard },
                    ]).map(({ key, label, icon: TabIcon }) => (
                        <button
                            key={key}
                            onClick={() => {
                                if (key === "board") {
                                    openAgentBoardWorkspace()
                                    return
                                }
                                onViewModeChange(key)
                            }}
                            className={cn(
                                "relative flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors",
                                viewMode === key
                                    ? "text-slate-900"
                                    : "text-slate-500 hover:text-slate-700"
                            )}
                        >
                            <TabIcon className="h-3.5 w-3.5" />
                            {label}
                            {viewMode === key && (
                                <span className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-slate-600" />
                            )}
                        </button>
                    ))}
                </div>
            )}

            {/* Error banner */}
            {(lastError || contextPackError) && (
                <div className="flex shrink-0 items-start gap-2 border-b border-rose-200 bg-rose-50 px-4 py-2 text-rose-700">
                    <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                    <span className="text-xs">{contextPackError || lastError}</span>
                </div>
            )}

            {/* Main content area */}
            <div className="flex-1 min-h-0 overflow-hidden">
                {viewMode === "context" ? (
                    <ContextDialogPanel
                        selectedPaper={
                            selectedPaper
                                ? {
                                    id: selectedPaper.id,
                                    title: selectedPaper.title,
                                    abstract: selectedPaper.abstract,
                                }
                                : null
                        }
                        generationStatus={genStatus}
                        generationProgress={generationProgress}
                        liveObservations={liveObservations}
                        contextPack={contextPack}
                        contextPackLoading={contextPackLoading}
                        contextPackError={contextPackError}
                        onGenerate={(paper) =>
                            generateContextPack({
                                paperId: paper.id,
                                title: paper.title,
                                abstract: paper.abstract,
                            })
                        }
                        onSessionCreated={handleSessionCreated}
                        onDeployToBoard={openAgentBoardWorkspace}
                    />
                ) : viewMode === "board" ? (
                    <AgentBoard paperId={selectedPaperId} monitorMode />
                ) : activeFileData ? (
                    /* File Viewer */
                    <div className="h-full flex flex-col">
                        <div className="flex shrink-0 items-center justify-between border-b border-slate-200 bg-[#eceee8] px-4 py-2">
                            <div className="flex items-center gap-2 text-sm">
                                <FileCode className="h-4 w-4 text-slate-500" />
                                <span className="font-medium text-slate-900">{activeFileData.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Button
                                    variant="default"
                                    size="sm"
                                    className="h-7 text-xs"
                                    onClick={saveActiveFile}
                                    disabled={!projectDir || saving}
                                >
                                    <Save className="h-3.5 w-3.5 mr-1" />
                                    {saving ? "Saving..." : "Save"}
                                </Button>
                                <button
                                    onClick={() => setActiveFile("")}
                                    className="rounded p-1.5 transition-colors hover:bg-slate-200"
                                    title="Close"
                                >
                                    <X className="h-4 w-4" />
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 min-h-0 overflow-hidden">
                            <Editor
                                height="100%"
                                language={activeFileData.language}
                                value={activeFileData.content}
                                theme="light"
                                onChange={(value) => updateFile(activeFileData.name, value || "")}
                                options={{
                                    minimap: { enabled: false },
                                    fontSize: 13,
                                    lineNumbers: "on",
                                    scrollBeyondLastLine: false,
                                    automaticLayout: true,
                                    padding: { top: 12, bottom: 12 },
                                    fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', monospace",
                                }}
                            />
                        </div>
                    </div>
                ) : (
                    /* Chat Timeline */
                    <ScrollArea className="h-full bg-[#f5f5f2]">
                        <div className="p-4">
                            {!activeTask || activeTask.actions.length === 0 ? (
                                <div className="flex flex-col items-center justify-center space-y-4 py-20 text-slate-500">
                                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-[#eceee8]">
                                        <MessageSquare className="h-8 w-8 opacity-30" />
                                    </div>
                                    <div className="text-center space-y-2">
                                        <p className="font-medium text-slate-900">Ready to chat</p>
                                        <p className="text-xs max-w-[280px]">
                                            {selectedPaper
                                                ? `Send a message to start working with ${runtimeLabel} on this paper`
                                                : "Select or create a paper to get started"}
                                        </p>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-0">
                                    {activeTask.actions.map((action, index) => (
                                        <ActionItem
                                            key={action.id}
                                            action={action}
                                            onViewDiff={setDiffAction}
                                            isLast={index === activeTask.actions.length - 1}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                )}
            </div>

            {viewMode === "log" && (
                /* Rich Input Area - CodePilot Style */
                <div className="shrink-0 border-t border-slate-200 bg-[#f1f2ed] p-4">
                    <div className="overflow-hidden rounded-lg border border-slate-200 bg-[#e8ebe4]">
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-[#eef0ea] px-4 py-2 text-xs">
                            <div className="min-w-0">
                                <div className="flex min-w-0 items-center gap-2">
                                    <span
                                        className={cn(
                                            "inline-flex items-center rounded-full border px-2 py-0.5 font-medium",
                                            runtimeInfo.source === "claude_code"
                                                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                                : runtimeInfo.source === "anthropic_api"
                                                    ? "border-amber-200 bg-amber-50 text-amber-700"
                                                    : "border-slate-200 bg-slate-100 text-slate-600",
                                        )}
                                    >
                                        {runtimeLoading ? "Checking runtime" : runtimeInfo.label}
                                    </span>
                                    <span className="truncate text-slate-600">
                                        {runtimeLoading ? "Resolving Claude Code status..." : runtimeInfo.statusLabel}
                                    </span>
                                </div>
                                <div className="mt-1 truncate font-mono text-[10px] text-slate-500">
                                    {commandPreview}
                                </div>
                            </div>
                            <div className="truncate text-[11px] text-slate-500" title={runtimeInfo.cwd || runtimeInfo.actualCwd || undefined}>
                                {runtimeInfo.workspaceLabel}
                            </div>
                        </div>
                        <Textarea
                            value={messageInput}
                            onChange={(e) => setMessageInput(e.target.value)}
                            placeholder={messagePlaceholder}
                            className="min-h-[60px] resize-none border-0 bg-transparent px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus-visible:ring-0"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault()
                                    handleSendMessage()
                                }
                            }}
                        />
                        <div className="flex items-center justify-between border-t border-slate-200 bg-[#f3f4f0] px-3 py-2">
                            <div className="flex items-center gap-2">
                                {/* Paper attachment indicator */}
                                {selectedPaper && (
                                    <div className="flex items-center gap-1.5 rounded-md border border-slate-200 bg-[#f7f7f4] px-2 py-1 text-xs text-slate-600">
                                        <FileText className="h-3.5 w-3.5" />
                                        <span className="max-w-[150px] truncate">{selectedPaper.title}</span>
                                    </div>
                                )}
                                {/* Mode selector */}
                                <Select value={mode} onValueChange={(v) => setMode(v as Mode)}>
                                    <SelectTrigger className="h-7 w-[96px] border-slate-200 bg-[#f7f7f4] text-xs text-slate-700">
                                        <Code className="h-3.5 w-3.5 mr-1" />
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="Code">Code</SelectItem>
                                        <SelectItem value="Plan">Plan</SelectItem>
                                        <SelectItem value="Ask">Ask</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="flex items-center gap-2">
                                {/* Model selector */}
                                <Input
                                    value={model}
                                    onChange={(event) => setModel(event.target.value)}
                                    placeholder="sonnet / opus / claude-sonnet-4-6"
                                    className="h-8 w-[220px] border-slate-200 bg-[#f7f7f4] text-xs text-slate-700"
                                    title="Claude Code model alias or full model name"
                                />
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="h-8 gap-1.5 border-slate-200 bg-[#f7f7f4] text-xs text-slate-700 hover:bg-white"
                                    onClick={handleDelegateToCodex}
                                    disabled={!messageInput.trim() || isBusy}
                                    title="Create a real Codex subagent task from this prompt"
                                >
                                    <Bot className="h-3.5 w-3.5" />
                                    Codex
                                </Button>
                                {/* Send button */}
                                <Button
                                    size="icon"
                                    className="h-8 w-8 rounded-md bg-slate-700 text-white hover:bg-slate-600"
                                    onClick={handleSendMessage}
                                    disabled={!messageInput.trim() || isBusy}
                                >
                                    <Send className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Diff Modal */}
            <DiffModal
                isOpen={!!diffAction}
                oldValue={diffAction?.metadata?.oldContent || '// Original file content'}
                newValue={diffAction?.metadata?.newContent || diffAction?.metadata?.diff || '// Modified file content'}
                filename={diffAction?.metadata?.filename}
                onClose={() => setDiffAction(null)}
                onApply={() => setDiffAction(null)}
                onReject={() => setDiffAction(null)}
            />

            {/* Workspace Setup Dialog */}
            {selectedPaper && (
                <WorkspaceSetupDialog
                    paper={selectedPaper}
                    open={showWorkspaceSetup}
                    onConfirm={handleWorkspaceConfirm}
                    onCancel={() => {
                        setShowWorkspaceSetup(false)
                        setPendingAction(null)
                    }}
                />
            )}
        </div>
    )
}
