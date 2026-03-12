"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { useStudioStore, AgentAction } from "@/lib/store/studio-store"
import { useProjectContext } from "@/lib/store/project-context"
import { readSSE } from "@/lib/sse"
import { CodeBlock } from "@/components/ai-elements"
import { DiffModal } from "./DiffViewer"
import { WorkspaceSetupDialog } from "./WorkspaceSetupDialog"
import { ContextDialogPanel } from "./ContextDialogPanel"
import { AgentBoard } from "./AgentBoard"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
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
import { useTheme } from "next-themes"
import type { ContextPackSession } from "@/lib/types/p2c"

type StepStatus = "idle" | "running" | "success" | "error"
type Mode = "Code" | "Plan" | "Ask"
export type ReproductionViewMode = "log" | "context" | "agent_board"

interface ReproductionLogProps {
    viewMode: ReproductionViewMode
    onViewModeChange: (mode: ReproductionViewMode) => void
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
    thinking: { bg: "bg-purple-50 dark:bg-purple-950/30", text: "text-purple-600 dark:text-purple-400", border: "border-purple-200 dark:border-purple-800" },
    file_change: { bg: "bg-blue-50 dark:bg-blue-950/30", text: "text-blue-600 dark:text-blue-400", border: "border-blue-200 dark:border-blue-800" },
    function_call: { bg: "bg-orange-50 dark:bg-orange-950/30", text: "text-orange-600 dark:text-orange-400", border: "border-orange-200 dark:border-orange-800" },
    error: { bg: "bg-red-50 dark:bg-red-950/30", text: "text-red-600 dark:text-red-400", border: "border-red-200 dark:border-red-800" },
    complete: { bg: "bg-green-50 dark:bg-green-950/30", text: "text-green-600 dark:text-green-400", border: "border-green-200 dark:border-green-800" },
    text: { bg: "bg-muted/50", text: "text-foreground", border: "border-border" },
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
                <div className="absolute left-2.5 top-6 bottom-0 w-px bg-border" />
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
                                    <span className="text-[10px] px-1 py-0.5 rounded bg-muted">
                                        <span className="text-green-600 dark:text-green-400">+{action.metadata.linesAdded || 0}</span>
                                        <span className="text-muted-foreground mx-0.5">/</span>
                                        <span className="text-red-600 dark:text-red-400">-{action.metadata.linesDeleted || 0}</span>
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
                                            className="text-muted-foreground hover:text-foreground transition-colors"
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
                            <p className="text-xs text-foreground/90 whitespace-pre-wrap leading-relaxed">{action.content}</p>
                        )}
                    </div>

                    <div className="flex items-center gap-1 text-[9px] text-muted-foreground shrink-0">
                        <Clock className="h-2 w-2" />
                        {formatTime(action.timestamp)}
                    </div>
                </div>
            </div>
        </div>
    )
}

export function ReproductionLog({ viewMode, onViewModeChange }: ReproductionLogProps) {
    const router = useRouter()
    const { theme } = useTheme()
    const {
        papers,
        tasks,
        activeTaskId,
        selectedPaperId,
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
    const [model, setModel] = useState("claude-sonnet-4-5")
    const [lastError, setLastError] = useState<string | null>(null)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)
    const [saving, setSaving] = useState(false)
    const [messageInput, setMessageInput] = useState("")
    const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
    const [pendingAction, setPendingAction] = useState<"chat" | null>(null)
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

    const handleSendMessageWithDir = async (message: string, targetDir?: string) => {
        setStatus("running")
        setLastError(null)
        onViewModeChange("log")

        const taskId = addTask(`Chat — ${message.slice(0, 30)}${message.length > 30 ? "…" : ""}`)
        addAction(taskId, { type: "thinking", content: `[${mode}] Sending to Claude...` })

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

    const handleSessionCreated = (session: ContextPackSession) => {
        onViewModeChange("log")
        if (session.initial_prompt) {
            setMessageInput(session.initial_prompt)
        }
    }

    const openAgentBoardFocusPage = () => {
        if (selectedPaperId) {
            router.push(`/studio/agent-board/${selectedPaperId}`)
            return
        }
        onViewModeChange("agent_board")
    }

    return (
        <div className="h-full w-full flex-1 flex flex-col min-w-0 min-h-0 bg-background">
            {/* Tab Navigation */}
            <div className="flex items-center shrink-0 border-b">
                {([
                    { key: "context" as const, label: "Context", icon: Activity },
                    { key: "log" as const, label: "Chat", icon: MessageSquare },
                    { key: "agent_board" as const, label: "Agent Board", icon: LayoutDashboard },
                ]).map(({ key, label, icon: TabIcon }) => (
                    <button
                        key={key}
                        onClick={() => {
                            if (key === "agent_board") {
                                openAgentBoardFocusPage()
                                return
                            }
                            onViewModeChange(key)
                        }}
                        className={cn(
                            "flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors relative",
                            viewMode === key
                                ? "text-foreground"
                                : "text-muted-foreground hover:text-foreground/80"
                        )}
                    >
                        <TabIcon className="h-3.5 w-3.5" />
                        {label}
                        {viewMode === key && (
                            <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-primary rounded-full" />
                        )}
                    </button>
                ))}
            </div>

            {/* Error banner */}
            {(lastError || contextPackError) && (
                <div className="px-4 py-2 flex items-start gap-2 shrink-0 bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400">
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
                        onDeployToBoard={openAgentBoardFocusPage}
                    />
                ) : viewMode === "agent_board" ? (
                    <AgentBoard paperId={selectedPaperId} />
                ) : activeFileData ? (
                    /* File Viewer */
                    <div className="h-full flex flex-col">
                        <div className="px-4 py-2 border-b flex items-center justify-between bg-muted/30 shrink-0">
                            <div className="flex items-center gap-2 text-sm">
                                <FileCode className="h-4 w-4 text-muted-foreground" />
                                <span className="font-medium">{activeFileData.name}</span>
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
                                    className="p-1.5 rounded hover:bg-muted transition-colors"
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
                                theme={theme === "dark" ? "vs-dark" : "light"}
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
                    <ScrollArea className="h-full">
                        <div className="p-4">
                            {!activeTask || activeTask.actions.length === 0 ? (
                                <div className="flex flex-col items-center justify-center text-muted-foreground py-20 space-y-4">
                                    <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center">
                                        <MessageSquare className="h-8 w-8 opacity-30" />
                                    </div>
                                    <div className="text-center space-y-2">
                                        <p className="font-medium">Ready to chat</p>
                                        <p className="text-xs max-w-[280px]">
                                            {selectedPaper
                                                ? "Send a message to start working with Claude on this paper"
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
                <div className="border-t p-4 shrink-0">
                    <div className="border rounded-xl bg-muted/30 overflow-hidden">
                        <Textarea
                            value={messageInput}
                            onChange={(e) => setMessageInput(e.target.value)}
                            placeholder="Message Claude..."
                            className="border-0 bg-transparent resize-none min-h-[60px] focus-visible:ring-0 px-4 py-3"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault()
                                    handleSendMessage()
                                }
                            }}
                        />
                        <div className="px-3 py-2 flex items-center justify-between border-t bg-background/50">
                            <div className="flex items-center gap-2">
                                {/* Paper attachment indicator */}
                                {selectedPaper && (
                                    <div className="flex items-center gap-1.5 px-2 py-1 bg-muted rounded-md text-xs text-muted-foreground">
                                        <FileText className="h-3.5 w-3.5" />
                                        <span className="max-w-[150px] truncate">{selectedPaper.title}</span>
                                    </div>
                                )}
                                {/* Mode selector */}
                                <Select value={mode} onValueChange={(v) => setMode(v as Mode)}>
                                    <SelectTrigger className="h-7 w-[90px] text-xs border-0 bg-muted">
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
                                <Select value={model} onValueChange={setModel}>
                                    <SelectTrigger className="h-7 w-[130px] text-xs border-0 bg-muted">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="claude-sonnet-4-5">Sonnet 4.5</SelectItem>
                                        <SelectItem value="claude-opus-4-5">Opus 4.5</SelectItem>
                                        <SelectItem value="claude-haiku-4-5">Haiku 4.5</SelectItem>
                                    </SelectContent>
                                </Select>
                                {/* Send button */}
                                <Button
                                    size="icon"
                                    className="h-8 w-8 rounded-full"
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
