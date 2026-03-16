"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import {
    buildCommandPreview,
    type ActiveCommand as CliActiveCommand,
    type CommandResult as CliCommandResult,
    getCommandPresets,
    type LastCommandOutput as CliLastCommandOutput,
} from "./CliCommandRunner"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import { parseStudioSlashCommand } from "@/lib/studio-slash"
import type { StudioRuntimeInfo } from "@/lib/studio-runtime"
import { cn } from "@/lib/utils"
import {
    CheckCircle2,
    AlertCircle,
    FileText,
    Bot,
    User,
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
    Settings2,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import Editor from "@monaco-editor/react"
import type { ContextPackSession } from "@/lib/types/p2c"

type StepStatus = "idle" | "running" | "success" | "error"
type Mode = "Code" | "Plan" | "Ask"
type EffortOption = "default" | "low" | "medium" | "high" | "max"
export type ReproductionViewMode = "log" | "context" | "board" | "commands"

interface ReproductionLogProps {
    viewMode: ReproductionViewMode
    onViewModeChange: (mode: ReproductionViewMode) => void
    hideNavigation?: boolean
    onOpenBoardWorkspace?: () => void
    runtimeInfo: StudioRuntimeInfo
    runtimeLoading: boolean
}

const actionIcons: Record<string, React.ElementType> = {
    user: User,
    thinking: Loader2,
    file_change: FileCode,
    function_call: Wrench,
    error: AlertCircle,
    complete: CheckCircle2,
    text: Bot,
    run_command: Terminal,
}

const actionColors: Record<string, { bg: string; text: string; border: string }> = {
    user: { bg: "bg-slate-700", text: "text-white", border: "border-slate-700" },
    thinking: { bg: "bg-slate-100", text: "text-slate-700", border: "border-slate-200" },
    file_change: { bg: "bg-slate-100", text: "text-slate-700", border: "border-slate-200" },
    function_call: { bg: "bg-stone-100", text: "text-stone-700", border: "border-stone-200" },
    error: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200" },
    complete: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
    text: { bg: "bg-[#eef0ea]", text: "text-slate-800", border: "border-slate-200" },
}

function buildChatThreadTitle(message: string): string {
    const singleLine = message.replace(/\s+/g, " ").trim()
    if (!singleLine) return "New thread"
    return singleLine.length <= 56 ? singleLine : `${singleLine.slice(0, 53)}...`
}

function normalizeThinkingMessage(value: unknown): string | null {
    if (typeof value !== "string") return null
    const normalized = value.replace(/\s+/g, " ").trim()
    return normalized || null
}

function isGenericThinkingMessage(message: string): boolean {
    return /^(\[[^\]]+\] sending to .+|thinking|connecting(?: to [^.]+)?|working|processing|waiting)\.{0,3}$/i.test(message.trim())
}

function shouldReplaceThinkingMessage(current: string | null, next: string): boolean {
    if (!current) return true
    if (current === next) return false

    const currentIsGeneric = isGenericThinkingMessage(current)
    const nextIsGeneric = isGenericThinkingMessage(next)

    if (!currentIsGeneric && nextIsGeneric) {
        return false
    }
    return true
}

type SlashCommandItem = {
    id: string
    command: string
    label: string
    description: string
    group: "Claude Code" | "Runtime" | "Session"
    keywords: string[]
    icon: React.ElementType
    onSelect: (remainder: string) => void
}

type SlashTriggerMatch = {
    query: string
    token: string
    start: number
    end: number
}

function splitCommaSeparatedValues(value: string): string[] {
    return value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
}

function splitLineValues(value: string): string[] {
    return value
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean)
}

function resolveRequestedModel(modelOption: string, customModel: string): string {
    return modelOption === "custom" ? customModel.trim() : modelOption.trim()
}

function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

type ComposerPillTone = "neutral" | "accent" | "success" | "warning"

const composerPillToneClassName: Record<ComposerPillTone, string> = {
    neutral: "border-slate-200 bg-[#f7f8f4] text-slate-700",
    accent: "border-slate-300 bg-[#edf0e7] text-slate-800",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    warning: "border-amber-200 bg-amber-50 text-amber-700",
}

interface ComposerPillProps {
    label: string
    meta?: string
    tone?: ComposerPillTone
    icon?: React.ElementType
    onRemove?: () => void
}

function ComposerPill({
    label,
    meta,
    tone = "neutral",
    icon: Icon,
    onRemove,
}: ComposerPillProps) {
    return (
        <div
            className={cn(
                "inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs shadow-sm",
                composerPillToneClassName[tone],
            )}
        >
            {Icon ? <Icon className="h-3.5 w-3.5 shrink-0 opacity-80" /> : null}
            {meta ? (
                <span className="shrink-0 text-[10px] uppercase tracking-[0.12em] opacity-60">{meta}</span>
            ) : null}
            <span className="truncate font-medium">{label}</span>
            {onRemove ? (
                <button
                    type="button"
                    className="rounded-full p-0.5 transition-colors hover:bg-white/80"
                    onClick={onRemove}
                    title={`Remove ${label}`}
                >
                    <X className="h-3 w-3" />
                </button>
            ) : null}
        </div>
    )
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
    const hasToolResult = action.metadata?.result !== undefined
    const stringifyPayload = (payload: unknown): string =>
        typeof payload === "string" ? payload : JSON.stringify(payload, null, 2) || ""

    return (
        <div className="group relative flex gap-2.5">
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
                            <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                    <code className={cn("rounded-full px-2 py-0.5 text-[10px] font-mono", colors.bg, colors.text)}>
                                        {action.metadata.functionName}()
                                    </code>
                                    <span
                                        className={cn(
                                            "rounded-full border px-1.5 py-0.5 text-[10px] uppercase tracking-[0.12em]",
                                            hasToolResult
                                                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                                : "border-slate-200 bg-white text-slate-500",
                                        )}
                                    >
                                        {hasToolResult ? "Done" : "Running"}
                                    </span>
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
                        ) : action.type === 'thinking' ? (
                            <div className="inline-flex max-w-full items-start gap-2 rounded-full border border-slate-200 bg-white/80 px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] text-slate-500">
                                <span className="shrink-0 font-medium text-slate-400">Thinking</span>
                                <span className="min-w-0 whitespace-pre-wrap normal-case tracking-normal text-slate-500">
                                    {action.content}
                                </span>
                            </div>
                        ) : action.type === 'error' ? (
                            <div className={cn("text-xs rounded-md border px-2 py-1.5", colors.bg, colors.border)}>
                                <span className={colors.text}>{action.content}</span>
                            </div>
                        ) : action.type === 'user' ? (
                            <div className="ml-auto max-w-[86%] rounded-[20px] bg-slate-700 px-3.5 py-2.5 text-[12px] leading-6 text-white shadow-sm">
                                {action.content}
                            </div>
                        ) : action.type === 'complete' ? (
                            <span className={cn("rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-[10px] font-medium uppercase tracking-[0.12em]", colors.text)}>
                                Run complete
                            </span>
                        ) : (
                            <div className="max-w-[88%] rounded-[20px] border border-slate-200 bg-[#f7f8f4] px-3.5 py-2.5 text-[12px] leading-6 text-slate-800 shadow-[0_1px_0_rgba(255,255,255,0.6)_inset]">
                                <p className="whitespace-pre-wrap">{action.content}</p>
                            </div>
                        )}
                    </div>

                    <div className="flex shrink-0 items-center gap-1 text-[9px] text-slate-400 opacity-0 transition-opacity group-hover:opacity-100">
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
        lastGenCodeResult,
        contextPack,
        contextPackLoading,
        contextPackError,
        generationProgress,
        liveObservations,
        addTask,
        renameTask,
        addAction,
        upsertThinkingAction,
        attachResultToLatestFunctionCall,
        appendToLastAction,
        appendTaskHistory,
        updateTaskStatus,
        updatePaper,
        setActiveTask,
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
    const [modelOption, setModelOption] = useState("sonnet")
    const [customModel, setCustomModel] = useState("")
    const [lastError, setLastError] = useState<string | null>(null)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)
    const [saving, setSaving] = useState(false)
    const [messageInput, setMessageInput] = useState("")
    const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null)
    const [composerCursor, setComposerCursor] = useState(0)
    const [activeCliCommand, setActiveCliCommand] = useState<CliActiveCommand | null>(null)
    const [chatDraftBeforeUtility, setChatDraftBeforeUtility] = useState("")
    const [runningCliCommand, setRunningCliCommand] = useState(false)
    const [lastCommandOutput, setLastCommandOutput] = useState<CliLastCommandOutput | null>(null)
    const [slashSelectedIndex, setSlashSelectedIndex] = useState(0)
    const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
    const [continueLast, setContinueLast] = useState(false)
    const [resumeSession, setResumeSession] = useState("")
    const [cliSessionId, setCliSessionId] = useState("")
    const [agentOverride, setAgentOverride] = useState("")
    const [mcpConfigText, setMcpConfigText] = useState("")
    const [toolsText, setToolsText] = useState("")
    const [allowedToolsText, setAllowedToolsText] = useState("")
    const [addDirsText, setAddDirsText] = useState("")
    const [settingsText, setSettingsText] = useState("")
    const [effort, setEffort] = useState<EffortOption>("default")
    // Switch to context dialog when generation starts.
    useEffect(() => {
        if (contextPackLoading && viewMode !== "context") {
            onViewModeChange("context")
        }
    }, [contextPackLoading, onViewModeChange, viewMode])

    const knownModelAliases = useMemo(() => {
        const aliases = runtimeInfo.knownModelAliases.filter((item) => item.trim().length > 0)
        return aliases.length > 0 ? aliases : ["sonnet", "opus"]
    }, [runtimeInfo.knownModelAliases])

    useEffect(() => {
        if (modelOption === "custom") return
        if (!knownModelAliases.includes(modelOption)) {
            setModelOption(knownModelAliases[0] ?? "sonnet")
        }
    }, [knownModelAliases, modelOption])

    const activeTask = useMemo(
        () =>
            tasks.find(
                (task) =>
                    task.id === activeTaskId &&
                    (!selectedPaperId || task.paperId === selectedPaperId),
            ) ?? null,
        [activeTaskId, selectedPaperId, tasks],
    )
    const activeChatTask = activeTask?.kind === "chat" ? activeTask : null
    const visibleTask = activeChatTask
    const projectDir = selectedPaper?.outputDir || lastGenCodeResult?.outputDir || null
    const isBusy = status === "running"
    const runtimeLabel = runtimeLoading
        ? "Studio runtime"
        : runtimeInfo.source === "anthropic_api"
            ? "Anthropic API fallback"
            : "Claude Code"
    const requestedModel = useMemo(
        () => resolveRequestedModel(modelOption, customModel),
        [customModel, modelOption],
    )
    const parsedMcpConfig = useMemo(() => splitLineValues(mcpConfigText), [mcpConfigText])
    const parsedTools = useMemo(() => splitCommaSeparatedValues(toolsText), [toolsText])
    const parsedAllowedTools = useMemo(() => splitCommaSeparatedValues(allowedToolsText), [allowedToolsText])
    const parsedAddDirs = useMemo(() => splitLineValues(addDirsText), [addDirsText])
    const selectedEffort = effort === "default" ? null : effort
    const missingCustomModel = modelOption === "custom" && requestedModel.length === 0
    const commandMode = Boolean(activeCliCommand)
    const commandRuntimeUnavailable = activeCliCommand
        ? activeCliCommand.runtime === "claude"
            ? runtimeInfo.source !== "claude_code"
            : !runtimeInfo.opencodeAvailable
        : false
    const messagePlaceholder = runtimeLoading
        ? "Message Studio runtime..."
        : runtimeInfo.source === "anthropic_api"
            ? "Message fallback runtime..."
            : "Message Claude Code..."
    const composerPlaceholder = commandMode
        ? `Edit args for ${buildCommandPreview(activeCliCommand!.runtime, activeCliCommand!.preset, "").trim()} and press Enter to run`
        : messagePlaceholder
    const composerInteractionHint = commandMode
        ? "Enter to run · Esc clears command mode"
        : "Enter to send · Shift+Enter for newline · / for commands"

    useEffect(() => {
        if (activeTask && activeTask.kind !== "chat") {
            setActiveTask(null)
        }
    }, [activeTask, setActiveTask])

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
        runChatWithDir(directory)
    }

    const runChatWithDir = async (targetDir: string) => {
        // Chat with specified directory - called after workspace setup
        const message = messageInput.trim()
        if (!message) return
        if (!requestedModel) {
            setLastError("Select a Claude Code model alias or enter a full custom model name.")
            return
        }
        setMessageInput("")
        await handleSendMessageWithDir(message, targetDir)
    }

    const handleSendMessage = async () => {
        if (!messageInput.trim() || isBusy) return
        if (!requestedModel) {
            setLastError("Select a Claude Code model alias or enter a full custom model name.")
            return
        }

        // For Code mode, require a project directory
        if (mode === "Code" && !projectDir) {
            if (!selectedPaper) {
                setLastError("Select or create a paper first.")
                return
            }
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

        const threadTitle = buildChatThreadTitle(message)
        const existingHistory = activeChatTask?.history ?? []
        const taskId =
            activeChatTask?.id ??
            addTask(threadTitle)
        let assistantResponse = ""
        let assistantHistoryCommitted = false

        if (activeChatTask && activeChatTask.history.length === 0 && activeChatTask.actions.length === 0) {
            renameTask(taskId, threadTitle)
        }

        const initialThinking = `[${mode}] Sending to ${runtimeLabel}...`
        addAction(taskId, { type: "user", content: message })
        appendTaskHistory(taskId, { role: "user", content: message })
        upsertThinkingAction(taskId, initialThinking)
        updateTaskStatus(taskId, "running")

        try {
            const res = await fetch("/api/studio/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message,
                    mode,
                    model: requestedModel,
                    paper: selectedPaper ? {
                        title: selectedPaper.title,
                        abstract: selectedPaper.abstract,
                        method_section: selectedPaper.methodSection,
                    } : undefined,
                    history: existingHistory,
                    session_id: taskId,
                    project_dir: targetDir,
                    context_pack_id: contextPack?.context_pack_id,
                    continue_last: continueLast,
                    resume_session: resumeSession.trim() || undefined,
                    cli_session_id: cliSessionId.trim() || undefined,
                    agent: agentOverride.trim() || undefined,
                    mcp_config: parsedMcpConfig,
                    tools: parsedTools,
                    allowed_tools: parsedAllowedTools,
                    add_dirs: parsedAddDirs,
                    settings: settingsText.trim() || undefined,
                    effort: selectedEffort ?? undefined,
                }),
            })

            if (!res.ok || !res.body) {
                throw new Error(`Failed to send message (${res.status})`)
            }
            // Track whether the last action is a text block so we can
            // append to it (producing one continuous bubble) instead of
            // creating a new action per chunk.
            let lastActionIsText = false
            let lastStreamActionType: "thinking" | "text" | "tool" | "other" = "thinking"
            let lastThinkingContent: string | null = initialThinking

            const pushThinking = (value: unknown) => {
                const message = normalizeThinkingMessage(value)
                if (!message) return

                if (lastStreamActionType === "thinking") {
                    if (!shouldReplaceThinkingMessage(lastThinkingContent, message)) {
                        return
                    }
                    upsertThinkingAction(taskId, message)
                } else {
                    addAction(taskId, { type: "thinking", content: message })
                }

                lastStreamActionType = "thinking"
                lastThinkingContent = message
                lastActionIsText = false
            }

            for await (const evt of readSSE(res.body)) {
                if (evt?.type === "progress") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const cliEvent = data.cli_event as string | undefined

                    if (cliEvent === "text") {
                        // Streaming text — append to current text bubble
                        const text = (data.text as string) || ""
                        if (text) {
                            assistantResponse += text
                            if (lastActionIsText) {
                                appendToLastAction(taskId, text)
                            } else {
                                addAction(taskId, { type: "text", content: text })
                                lastActionIsText = true
                            }
                            lastStreamActionType = "text"
                            lastThinkingContent = null
                        }
                    } else if (cliEvent === "tool_use") {
                        lastActionIsText = false
                        lastStreamActionType = "tool"
                        lastThinkingContent = null
                        addAction(taskId, {
                            type: "function_call",
                            content: `${data.tool_name}()`,
                            metadata: {
                                functionName: data.tool_name as string,
                                params: data.tool_input as Record<string, unknown>,
                            },
                        })
                    } else if (cliEvent === "tool_result") {
                        lastActionIsText = false
                        lastStreamActionType = "tool"
                        lastThinkingContent = null
                        const functionName = data.tool_name as string
                        const attached = attachResultToLatestFunctionCall(
                            taskId,
                            functionName,
                            data.content as string,
                        )
                        if (!attached) {
                            addAction(taskId, {
                                type: "function_call",
                                content: `${functionName}()`,
                                metadata: {
                                    functionName,
                                    result: data.content as string,
                                },
                            })
                        }
                    } else if (cliEvent === "thinking") {
                        pushThinking((data.text as string) || "Thinking...")
                    } else if (data.keepalive) {
                        // Keepalive heartbeat — ignore
                    } else if (data.message) {
                        // Legacy status messages (e.g. "Connecting to Claude CLI...")
                        pushThinking(data.message as string)
                    } else if (data.delta) {
                        // Fallback: legacy plain-text streaming (API fallback path)
                        const text = data.delta as string
                        assistantResponse += text
                        if (lastActionIsText) {
                            appendToLastAction(taskId, text)
                        } else {
                            addAction(taskId, { type: "text", content: text })
                            lastActionIsText = true
                        }
                        lastStreamActionType = "text"
                        lastThinkingContent = null
                    }
                } else if (evt?.type === "result") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const finalContent =
                        assistantResponse || (typeof data.content === "string" ? data.content : "")
                    if (finalContent.trim() && !assistantHistoryCommitted) {
                        appendTaskHistory(taskId, { role: "assistant", content: finalContent })
                        assistantHistoryCommitted = true
                    }
                    if (!finalContent.trim()) {
                        const summary = data.num_turns
                            ? `Completed in ${data.num_turns} turns`
                            : "Completed"
                        addAction(taskId, { type: "complete", content: summary })
                    }
                    updateTaskStatus(taskId, "completed")
                    setStatus("success")
                } else if (evt?.type === "error") {
                    if (assistantResponse.trim() && !assistantHistoryCommitted) {
                        appendTaskHistory(taskId, { role: "assistant", content: assistantResponse })
                        assistantHistoryCommitted = true
                    }
                    addAction(taskId, { type: "error", content: evt.message || "Chat failed" })
                    updateTaskStatus(taskId, "error")
                    setLastError(evt.message || "Chat failed")
                    setStatus("error")
                    return
                }
            }
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            if (assistantResponse.trim() && !assistantHistoryCommitted) {
                appendTaskHistory(taskId, { role: "assistant", content: assistantResponse })
            }
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

    const handleSelectCliCommand = (command: CliActiveCommand, nextArgs?: string, restoreDraft?: string) => {
        if (!commandMode) {
            setChatDraftBeforeUtility(restoreDraft ?? messageInput)
        }

        setActiveCliCommand(command)
        setMessageInput(nextArgs ?? command.preset.defaultArgs)
        setLastError(null)

        if (viewMode === "commands") {
            onViewModeChange("log")
        }
    }

    const handleClearCliCommand = () => {
        setActiveCliCommand(null)
        setMessageInput(chatDraftBeforeUtility)
        setChatDraftBeforeUtility("")
    }

    const showSyntheticCommandOutput = (preview: string, stdout: string) => {
        setLastCommandOutput({
            preview,
            error: null,
            result: {
                ok: true,
                command: ["studio-shell"],
                returncode: 0,
                stdout,
                stderr: "",
                cwd: projectDir,
            },
        })
    }

    const clearActiveConversation = () => {
        setActiveTask(null)
        setMessageInput("")
        setChatDraftBeforeUtility("")
        setActiveCliCommand(null)
        setLastError(null)
        setLastCommandOutput(null)
    }

    const executeCliCommand = async (command = activeCliCommand, args = messageInput) => {
        if (!command || runningCliCommand) return

        const selectedRuntimeUnavailable =
            command.runtime === "claude"
                ? runtimeInfo.source !== "claude_code"
                : !runtimeInfo.opencodeAvailable
        if (selectedRuntimeUnavailable) {
            setLastError("Selected command runtime is unavailable")
            return
        }

        const preview = buildCommandPreview(command.runtime, command.preset, args)
        setRunningCliCommand(true)
        setLastError(null)

        if (viewMode === "commands") {
            onViewModeChange("log")
        }

        try {
            const response = await fetch("/api/studio/command", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    runtime: command.runtime,
                    command: command.preset.command,
                    args: args.trim(),
                    project_dir: projectDir ?? undefined,
                }),
            })

            const payload = (await response.json()) as CliCommandResult

            setLastCommandOutput({
                preview,
                result: payload,
                error: payload?.ok ? null : payload?.stderr || `Command failed (${payload?.returncode ?? 500})`,
            })
        } catch (e) {
            const message = e instanceof Error ? e.message : "Failed to run Studio command"
            setLastCommandOutput({
                preview,
                result: null,
                error: message,
            })
        } finally {
            setRunningCliCommand(false)
        }
    }

    const setSlashScaffold = (command: string, value = "") => {
        const nextValue = value.trim() ? `/${command} ${value.trim()}` : `/${command} `
        setMessageInput(nextValue)
        setLastError(null)
    }

    const handleSlashSubmit = async (): Promise<boolean> => {
        const parsed = parseStudioSlashCommand(messageInput, knownModelAliases)
        if (!parsed) return false

        setLastError(null)

        if (parsed.kind === "help") {
            setMessageInput("")
            showSyntheticCommandOutput(
                "Claude Code slash help",
                [
                    "Supported Studio slash commands:",
                    "/help",
                    "/status",
                    "/new",
                    "/clear",
                    "/plan <request>",
                    "/model <alias>",
                    "/agents",
                    "/mcp [args]",
                    "/auth [args]",
                    "/doctor",
                    "",
                    "Chat turns stream Claude Code print mode.",
                    "Runtime entries launch standalone Claude CLI utilities.",
                    "Studio intentionally exposes a focused command subset instead of old local-only slash toggles.",
                ].join("\n"),
            )
            return true
        }

        if (parsed.kind === "status") {
            setMessageInput("")
            showSyntheticCommandOutput(
                "Claude Code status",
                [
                    `runtime: ${runtimeLoading ? "checking" : runtimeLabel}`,
                    `mode: ${mode}`,
                    `model: ${requestedModel || "pending"}`,
                    `workspace: ${projectDir ?? "not set"}`,
                    `paper: ${selectedPaper?.title ?? "none"}`,
                    `session: ${activeChatTask?.name ?? "new thread"}`,
                ].join("\n"),
            )
            return true
        }

        if (parsed.kind === "clear" || parsed.kind === "new_thread") {
            clearActiveConversation()
            return true
        }

        if (parsed.kind === "mode") {
            setMode(parsed.mode)
            setMessageInput(parsed.remainder)
            return true
        }

        if (parsed.kind === "model") {
            setModelOption(parsed.modelOption)
            setCustomModel(parsed.customModel)
            setMessageInput(parsed.remainder)
            return true
        }

        const preset = getCommandPresets(parsed.runtime).find((item) => item.id === parsed.presetId)
        if (!preset) {
            setLastError(`Unknown Claude Code command: ${parsed.presetId}`)
            return true
        }

        const command: CliActiveCommand = {
            runtime: parsed.runtime,
            preset,
        }
        const nextArgs = parsed.args || preset.defaultArgs
        handleSelectCliCommand(command, nextArgs, "")
        await executeCliCommand(command, nextArgs)
        return true
    }

    const handleComposerSubmit = () => {
        if (commandMode) {
            void executeCliCommand()
            return
        }

        if (messageInput.trim().startsWith("/")) {
            void handleSlashSubmit().then((handled) => {
                if (!handled) {
                    if (/^\/[a-z][a-z0-9-]*(?:\s|$)/i.test(messageInput.trim())) {
                        setLastError("Unsupported Studio slash command. Use /help to see the available Claude commands.")
                        focusComposerToEnd()
                    } else {
                        void handleSendMessage()
                    }
                } else {
                    focusComposerToEnd()
                }
            })
            return
        }

        void handleSendMessage()
    }

    const normalizedComposerCursor = Math.min(composerCursor, messageInput.length)
    const activeSlashMatch = useMemo<SlashTriggerMatch | null>(() => {
        if (commandMode) return null
        const beforeCursor = messageInput.slice(0, normalizedComposerCursor)
        const match = beforeCursor.match(/(^|\s)\/([^\s/]*)$/)
        if (!match) return null

        const token = match[2] ?? ""
        return {
            query: token.toLowerCase(),
            token,
            start: normalizedComposerCursor - token.length - 1,
            end: normalizedComposerCursor,
        }
    }, [commandMode, messageInput, normalizedComposerCursor])
    const slashPaletteActive = Boolean(activeSlashMatch)
    const slashQuery = activeSlashMatch?.query ?? ""
    const slashToken = activeSlashMatch?.token ?? ""

    const mergeComposerText = (before: string, inserted: string, after: string) => {
        let nextValue = before
        const normalizedInserted = inserted.trim()

        if (normalizedInserted) {
            if (nextValue && !/\s$/.test(nextValue)) {
                nextValue += " "
            }
            nextValue += normalizedInserted
        }

        if (after) {
            if (nextValue && !/\s$/.test(nextValue) && !/^\s/.test(after)) {
                nextValue += " "
            } else if (!nextValue && /^\s+/.test(after)) {
                nextValue += after.trimStart()
                return nextValue
            }
            nextValue += after
        }

        return nextValue
    }

    const replaceActiveSlashToken = (replacement = "") => {
        if (!activeSlashMatch) return replacement
        const before = messageInput.slice(0, activeSlashMatch.start)
        const after = messageInput.slice(activeSlashMatch.end)
        return mergeComposerText(before, replacement, after)
    }

    const syncComposerCursor = (target: HTMLTextAreaElement) => {
        setComposerCursor(target.selectionStart ?? target.value.length)
    }

    const focusComposerToEnd = () => {
        requestAnimationFrame(() => {
            if (!composerTextareaRef.current) return
            const nextCursor = composerTextareaRef.current.value.length
            composerTextareaRef.current.focus()
            composerTextareaRef.current.setSelectionRange(nextCursor, nextCursor)
            setComposerCursor(nextCursor)
        })
    }

    function openAgentBoardWorkspace() {
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

    const claudeCommandPresets = useMemo(() => {
        const byId = new Map(getCommandPresets("claude").map((preset) => [preset.id, preset]))
        return {
            agents: byId.get("claude-agents") ?? null,
            mcp: byId.get("claude-mcp") ?? null,
            auth: byId.get("claude-auth") ?? null,
            doctor: byId.get("claude-doctor") ?? null,
        }
    }, [])

    const slashCommands: SlashCommandItem[] = [
        {
            id: "slash-help",
            command: "help",
            label: "Slash help",
            description: "Show the Claude-style command subset that Studio currently supports.",
            group: "Claude Code",
            keywords: ["commands", "palette", "claude"],
            icon: MessageSquare,
            onSelect: (remainder) => setSlashScaffold("help", remainder),
        },
        {
            id: "slash-status",
            command: "status",
            label: "Status",
            description: "Show the current runtime, model, workspace, and active thread.",
            group: "Claude Code",
            keywords: ["runtime", "model", "workspace", "session"],
            icon: Activity,
            onSelect: (remainder) => setSlashScaffold("status", remainder),
        },
        {
            id: "slash-plan",
            command: "plan",
            label: "Plan mode",
            description: "Switch Studio into Claude Code plan mode for the next message.",
            group: "Claude Code",
            keywords: ["strategy", "outline", "design"],
            icon: LayoutDashboard,
            onSelect: (remainder) => setSlashScaffold("plan", remainder),
        },
        {
            id: "slash-model",
            command: "model",
            label: "Set model",
            description: "Use /model <alias-or-full-name> to switch the Claude Code model.",
            group: "Claude Code",
            keywords: ["model", "alias", "sonnet", "opus", "custom"],
            icon: Bot,
            onSelect: (remainder) => setSlashScaffold("model", remainder),
        },
        {
            id: "slash-new",
            command: "new",
            label: "New thread",
            description: "Start a fresh Studio thread and clear the current draft.",
            group: "Session",
            keywords: ["clear", "fresh", "reset"],
            icon: X,
            onSelect: (remainder) => setSlashScaffold("new", remainder),
        },
        {
            id: "slash-clear",
            command: "clear",
            label: "Clear conversation",
            description: "Reset the current conversation state without opening another panel.",
            group: "Session",
            keywords: ["new", "reset", "session"],
            icon: X,
            onSelect: (remainder) => setSlashScaffold("clear", remainder),
        },
        {
            id: "runtime-agents",
            command: "agents",
            label: "claude agents",
            description: "Open Claude Code command mode with `claude agents`.",
            group: "Runtime",
            keywords: ["runtime", "cli", "claude", "agents"],
            icon: Terminal,
            onSelect: (remainder) => {
                const preset = claudeCommandPresets.agents
                if (!preset) return
                handleSelectCliCommand({ runtime: "claude", preset }, remainder || preset.defaultArgs, remainder)
            },
        },
        {
            id: "runtime-mcp",
            command: "mcp",
            label: "claude mcp",
            description: "Open Claude Code command mode with `claude mcp list` by default.",
            group: "Runtime",
            keywords: ["runtime", "cli", "claude", "mcp", "servers"],
            icon: Wrench,
            onSelect: (remainder) => {
                const preset = claudeCommandPresets.mcp
                if (!preset) return
                handleSelectCliCommand({ runtime: "claude", preset }, remainder || preset.defaultArgs, remainder)
            },
        },
        {
            id: "runtime-auth",
            command: "auth",
            label: "claude auth",
            description: "Open Claude Code command mode with `claude auth status`.",
            group: "Runtime",
            keywords: ["runtime", "cli", "claude", "auth", "status", "login"],
            icon: Bot,
            onSelect: (remainder) => {
                const preset = claudeCommandPresets.auth
                if (!preset) return
                handleSelectCliCommand({ runtime: "claude", preset }, remainder || preset.defaultArgs, remainder)
            },
        },
        {
            id: "runtime-doctor",
            command: "doctor",
            label: "claude doctor",
            description: "Run Claude Code health checks in command mode.",
            group: "Runtime",
            keywords: ["runtime", "cli", "claude", "doctor", "health"],
            icon: Terminal,
            onSelect: (remainder) => {
                const preset = claudeCommandPresets.doctor
                if (!preset) return
                handleSelectCliCommand({ runtime: "claude", preset }, remainder || preset.defaultArgs, remainder)
            },
        },
    ]

    const filteredSlashCommands = slashCommands.filter((item) => {
        if (!slashQuery) return true
        const haystack = [item.command, item.label, item.description, ...item.keywords]
            .join(" ")
            .toLowerCase()
        return haystack.includes(slashQuery)
    })

    useEffect(() => {
        if (!slashPaletteActive) {
            setSlashSelectedIndex(0)
            return
        }

        setSlashSelectedIndex((current) =>
            Math.min(current, Math.max(filteredSlashCommands.length - 1, 0)),
        )
    }, [filteredSlashCommands.length, slashPaletteActive, slashQuery])

    const handleApplySlashCommand = (command: SlashCommandItem) => {
        command.onSelect(replaceActiveSlashToken())
        setSlashSelectedIndex(0)
        focusComposerToEnd()
    }

    const composerHelperText = commandMode
        ? "Claude Code command selected. Enter runs it, and clear returns the composer to chat."
        : missingCustomModel
            ? "Enter a full Claude Code model name before sending."
            : "Type / for Claude-style commands, runtime checks, and thread controls."
    const activeModeLabel = mode
    const activeModelLabel =
        modelOption === "custom"
            ? requestedModel || "Custom model"
            : requestedModel
    const composerBadges = [
        selectedPaper
            ? {
                id: "paper",
                label: selectedPaper.title,
                meta: "paper",
                tone: "neutral" as const,
                icon: FileText,
            }
            : null,
        {
            id: "mode",
            label: activeModeLabel,
            meta: "mode",
            tone: "accent" as const,
            icon: Code,
        },
        {
            id: "model",
            label: activeModelLabel,
            meta: "model",
            tone: modelOption === "custom" ? "warning" as const : "neutral" as const,
            icon: Bot,
        },
        commandMode
            ? {
                id: "command",
                label: buildCommandPreview(activeCliCommand!.runtime, activeCliCommand!.preset, messageInput),
                meta: activeCliCommand!.runtime,
                tone: "accent" as const,
                icon: Terminal,
                onRemove: handleClearCliCommand,
            }
            : null,
        continueLast
            ? {
                id: "continue",
                label: "Continue session",
                meta: "flow",
                tone: "warning" as const,
                icon: ChevronRight,
                onRemove: () => setContinueLast(false),
            }
            : null,
        resumeSession.trim()
            ? {
                id: "resume",
                label: resumeSession.trim(),
                meta: "resume",
                tone: "neutral" as const,
                icon: Clock,
                onRemove: () => setResumeSession(""),
            }
            : null,
        cliSessionId.trim()
            ? {
                id: "session-id",
                label: cliSessionId.trim(),
                meta: "session-id",
                tone: "neutral" as const,
                icon: Terminal,
                onRemove: () => setCliSessionId(""),
            }
            : null,
        agentOverride.trim()
            ? {
                id: "agent",
                label: agentOverride.trim(),
                meta: "agent",
                tone: "neutral" as const,
                icon: Bot,
                onRemove: () => setAgentOverride(""),
            }
            : null,
        selectedEffort
            ? {
                id: "effort",
                label: selectedEffort,
                meta: "effort",
                tone: "success" as const,
                icon: Activity,
                onRemove: () => setEffort("default"),
            }
            : null,
        parsedTools.length > 0
            ? {
                id: "tools",
                label: `${parsedTools.length} tool${parsedTools.length === 1 ? "" : "s"}`,
                meta: "tools",
                tone: "neutral" as const,
                icon: Wrench,
                onRemove: () => setToolsText(""),
            }
            : null,
        parsedAllowedTools.length > 0
            ? {
                id: "allowed-tools",
                label: `${parsedAllowedTools.length} allow${parsedAllowedTools.length === 1 ? "ed tool" : "ed tools"}`,
                meta: "allow",
                tone: "neutral" as const,
                icon: Wrench,
                onRemove: () => setAllowedToolsText(""),
            }
            : null,
        parsedAddDirs.length > 0
            ? {
                id: "add-dirs",
                label: `${parsedAddDirs.length} extra dir${parsedAddDirs.length === 1 ? "" : "s"}`,
                meta: "dirs",
                tone: "neutral" as const,
                icon: FileCode,
                onRemove: () => setAddDirsText(""),
            }
            : null,
        parsedMcpConfig.length > 0
            ? {
                id: "mcp",
                label: `${parsedMcpConfig.length} config${parsedMcpConfig.length === 1 ? "" : "s"}`,
                meta: "mcp",
                tone: "neutral" as const,
                icon: LayoutDashboard,
                onRemove: () => setMcpConfigText(""),
            }
            : null,
        settingsText.trim()
            ? {
                id: "settings",
                label: "Settings override",
                meta: "config",
                tone: "neutral" as const,
                icon: Settings2,
                onRemove: () => setSettingsText(""),
            }
            : null,
    ].filter(Boolean) as ComposerPillProps[]

    const consoleMode = viewMode === "log" || viewMode === "commands"
    const activeNavigationView = viewMode === "commands" ? "log" : viewMode

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
                                activeNavigationView === key
                                    ? "text-slate-900"
                                    : "text-slate-500 hover:text-slate-700"
                            )}
                        >
                            <TabIcon className="h-3.5 w-3.5" />
                            {label}
                            {activeNavigationView === key && (
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
                            {!visibleTask || visibleTask.actions.length === 0 ? (
                                <div className="flex flex-col items-center justify-center space-y-4 py-20 text-slate-500">
                                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-[#eceee8]">
                                        <MessageSquare className="h-8 w-8 opacity-30" />
                                    </div>
                                    <div className="text-center space-y-2">
                                        <p className="font-medium text-slate-900">Talk to Claude Code</p>
                                        <p className="text-xs max-w-[280px]">
                                            {selectedPaper
                                                ? `Start a thread for ${selectedPaper.title}. Runtime and delegation activity mirror into Monitor.`
                                                : "Select or create a paper to get started"}
                                        </p>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-0">
                                    {visibleTask.actions.map((action, index) => (
                                        <ActionItem
                                            key={action.id}
                                            action={action}
                                            onViewDiff={setDiffAction}
                                            isLast={index === visibleTask.actions.length - 1}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                )}
            </div>

            {consoleMode && (
                /* Rich Input Area - CodePilot Style */
                <div className="shrink-0 border-t border-slate-200 bg-[#f1f2ed] p-2.5">
                    <div className="overflow-hidden rounded-[28px] border border-slate-200 bg-[#e8ebe4] shadow-[0_20px_50px_rgba(15,23,42,0.06)]">
                        <div className="border-b border-slate-200 bg-[#eef1ea] px-4 py-2.5">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="flex min-w-0 items-center gap-2">
                                    <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                                        {commandMode ? "Command" : "Claude Code"}
                                    </span>
                                    <span className="truncate text-[12px] font-medium text-slate-800">
                                        {commandMode
                                            ? "Claude Code command"
                                            : selectedPaper
                                                ? selectedPaper.title
                                                : "Threaded chat"}
                                    </span>
                                </div>
                                <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-500">
                                    /
                                </span>
                            </div>
                        </div>

                        <div className="relative bg-[#eef0ea]">
                            {composerBadges.length > 0 ? (
                                <div className="flex flex-wrap gap-2 px-4 pt-3">
                                    {composerBadges.map((badge) => (
                                        <ComposerPill key={badge.label + badge.meta} {...badge} />
                                    ))}
                                </div>
                            ) : null}

                            {modelOption === "custom" && !commandMode ? (
                                <div className="px-4 pt-3">
                                    <Input
                                        value={customModel}
                                        onChange={(event) => setCustomModel(event.target.value)}
                                        placeholder="claude-sonnet-4-6"
                                        className="h-9 border-slate-200 bg-[#f7f8f4] text-xs text-slate-700"
                                        title="Full Claude Code model name"
                                    />
                                </div>
                            ) : null}

                            <Textarea
                                ref={composerTextareaRef}
                                value={messageInput}
                                onChange={(e) => {
                                    setMessageInput(e.target.value)
                                    syncComposerCursor(e.target)
                                }}
                                onClick={(e) => syncComposerCursor(e.currentTarget)}
                                onSelect={(e) => syncComposerCursor(e.currentTarget)}
                                onKeyUp={(e) => syncComposerCursor(e.currentTarget)}
                                placeholder={composerPlaceholder}
                                className="min-h-[88px] resize-none border-0 bg-transparent px-4 py-3 text-[14px] leading-7 text-slate-800 placeholder:text-slate-400 focus-visible:ring-0"
                                onKeyDown={(e) => {
                                    if (slashPaletteActive) {
                                        if (e.key === "ArrowDown" && filteredSlashCommands.length > 0) {
                                            e.preventDefault()
                                            setSlashSelectedIndex((current) =>
                                                Math.min(current + 1, filteredSlashCommands.length - 1),
                                            )
                                            return
                                        }

                                        if (e.key === "ArrowUp" && filteredSlashCommands.length > 0) {
                                            e.preventDefault()
                                            setSlashSelectedIndex((current) => Math.max(current - 1, 0))
                                            return
                                        }

                                        if (e.key === "Escape") {
                                            e.preventDefault()
                                            setMessageInput(replaceActiveSlashToken())
                                            setSlashSelectedIndex(0)
                                            return
                                        }

                                        if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
                                            e.preventDefault()
                                            const selectedSlashCommand =
                                                filteredSlashCommands[slashSelectedIndex] ?? filteredSlashCommands[0]
                                            if (selectedSlashCommand) {
                                                handleApplySlashCommand(selectedSlashCommand)
                                            }
                                            return
                                        }
                                    }

                                    if ((e.key === "Backspace" || e.key === "Escape") && !messageInput.trim()) {
                                        if (commandMode) {
                                            e.preventDefault()
                                            handleClearCliCommand()
                                            return
                                        }
                                    }

                                    if (e.key === "Enter" && !e.shiftKey) {
                                        e.preventDefault()
                                        handleComposerSubmit()
                                    }
                                }}
                            />

                            {slashPaletteActive ? (
                                <div className="px-4 pb-3">
                                    <div className="max-w-[620px] overflow-hidden rounded-2xl border border-slate-200 bg-[#f7f8f4] shadow-[0_18px_40px_rgba(15,23,42,0.10)]">
                                        <div className="flex items-center justify-between gap-3 border-b border-slate-200 bg-[#f0f2ec] px-3 py-2.5">
                                            <div>
                                                <div className="text-[11px] font-medium text-slate-800">Claude Code commands</div>
                                                <div className="mt-0.5 text-[11px] text-slate-500">
                                                    Slash opens the Studio command surface: Claude-style chat commands plus safe runtime utilities.
                                                </div>
                                            </div>
                                            <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-mono text-[11px] text-slate-500">
                                                /{slashToken || ""}
                                            </span>
                                        </div>

                                        <ScrollArea className="max-h-64">
                                            {filteredSlashCommands.length === 0 ? (
                                                <div className="px-3 py-4 text-sm text-slate-500">
                                                    No matching slash command.
                                                </div>
                                            ) : (
                                                <div className="space-y-2 p-2">
                                                    {(["Claude Code", "Session", "Runtime"] as const).map((group) => {
                                                        const groupItems = filteredSlashCommands.filter((item) => item.group === group)
                                                        if (groupItems.length === 0) return null

                                                        return (
                                                            <div key={group}>
                                                                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                                                                    {group}
                                                                </div>
                                                                <div className="space-y-1">
                                                                    {groupItems.map((item) => {
                                                                        const globalIndex = filteredSlashCommands.findIndex((entry) => entry.id === item.id)
                                                                        const selected = globalIndex === slashSelectedIndex
                                                                        const ItemIcon = item.icon

                                                                        return (
                                                                            <button
                                                                                key={item.id}
                                                                                type="button"
                                                                                className={cn(
                                                                                    "flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors",
                                                                                    selected
                                                                                        ? "border-slate-300 bg-[#edf0e7]"
                                                                                        : "border-transparent bg-transparent hover:border-slate-200 hover:bg-[#eef1ea]",
                                                                                )}
                                                                                onMouseEnter={() => setSlashSelectedIndex(globalIndex)}
                                                                                onClick={() => handleApplySlashCommand(item)}
                                                                            >
                                                                                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white">
                                                                                    <ItemIcon className="h-3.5 w-3.5 text-slate-500" />
                                                                                </div>
                                                                                <div className="min-w-0 flex-1">
                                                                                    <div className="flex items-center gap-2">
                                                                                        <span className="font-mono text-[11px] text-slate-900">
                                                                                            /{item.command}
                                                                                        </span>
                                                                                        <span className="truncate text-[11px] text-slate-500">
                                                                                            {item.label}
                                                                                        </span>
                                                                                    </div>
                                                                                    <div className="mt-1 line-clamp-2 text-[11px] leading-5 text-slate-500">
                                                                                        {item.description}
                                                                                    </div>
                                                                                </div>
                                                                                <span className="shrink-0 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-400">
                                                                                    {item.group}
                                                                                </span>
                                                                            </button>
                                                                        )
                                                                    })}
                                                                </div>
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            )}
                                        </ScrollArea>
                                    </div>
                                </div>
                            ) : null}
                        </div>

                        <div className="border-t border-slate-200 bg-[#f3f4ef] px-3 py-3">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex flex-1 flex-wrap items-center gap-1.5">
                                    <Select value={mode} onValueChange={(value) => setMode(value as Mode)}>
                                        <SelectTrigger className="h-8 w-[104px] rounded-full border-slate-200 bg-white text-xs text-slate-700">
                                            <Code className="mr-1 h-3.5 w-3.5" />
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Code">Code</SelectItem>
                                            <SelectItem value="Plan">Plan</SelectItem>
                                            <SelectItem value="Ask">Ask</SelectItem>
                                        </SelectContent>
                                    </Select>

                                    <Select value={modelOption} onValueChange={setModelOption}>
                                        <SelectTrigger className="h-8 w-[148px] rounded-full border-slate-200 bg-white text-xs text-slate-700">
                                            <Bot className="mr-1 h-3.5 w-3.5" />
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {knownModelAliases.map((alias) => (
                                                <SelectItem key={alias} value={alias}>
                                                    {alias}
                                                </SelectItem>
                                            ))}
                                            <SelectItem value="custom">Custom model…</SelectItem>
                                        </SelectContent>
                                    </Select>

                                </div>

                                <Button
                                    size="icon"
                                    className="h-10 w-10 shrink-0 rounded-full bg-slate-800 text-white shadow-sm hover:bg-slate-700"
                                    onClick={handleComposerSubmit}
                                    disabled={
                                        commandMode
                                            ? runningCliCommand || commandRuntimeUnavailable
                                            : !messageInput.trim() || isBusy || missingCustomModel
                                    }
                                    title={
                                        commandMode
                                            ? commandRuntimeUnavailable
                                                ? "Selected command runtime is unavailable"
                                                : runningCliCommand
                                                    ? "Running command"
                                                    : "Run Claude Code command"
                                            : missingCustomModel
                                                    ? "Enter a full Claude model name first"
                                                    : "Send to Claude Code"
                                    }
                                >
                                    {commandMode ? (
                                        runningCliCommand ? (
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                        ) : (
                                            <Terminal className="h-4 w-4" />
                                        )
                                    ) : (
                                        <Send className="h-4 w-4" />
                                    )}
                                </Button>
                            </div>

                            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
                                <span>{composerHelperText}</span>
                                <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
                                    {composerInteractionHint}
                                </span>
                            </div>

                            {lastCommandOutput ? (
                                <div className="mt-3 rounded-2xl border border-slate-200 bg-[#f7f8f4]">
                                    <div className="flex items-start justify-between gap-3 border-b border-slate-200 bg-[#eef1ea] px-3 py-2.5">
                                        <div className="min-w-0">
                                            <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500">
                                                Recent command output
                                            </div>
                                            <div className="mt-1 truncate font-mono text-[11px] text-slate-700">
                                                {lastCommandOutput.preview}
                                            </div>
                                        </div>
                                        <button
                                            type="button"
                                            className="rounded-full p-1 text-slate-500 transition-colors hover:bg-white hover:text-slate-900"
                                            onClick={() => setLastCommandOutput(null)}
                                            title="Dismiss latest command output"
                                        >
                                            <X className="h-3.5 w-3.5" />
                                        </button>
                                    </div>

                                    <div className="space-y-3 px-3 py-3">
                                        {lastCommandOutput.result ? (
                                            <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-500">
                                                exit {lastCommandOutput.result.returncode} · cwd {lastCommandOutput.result.cwd ?? "n/a"}
                                            </div>
                                        ) : null}

                                        {lastCommandOutput.error ? (
                                            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                                                {lastCommandOutput.error}
                                            </div>
                                        ) : null}

                                        {lastCommandOutput.result?.stdout ? (
                                            <div className="rounded-xl border border-slate-200 bg-white">
                                                <div className="border-b border-slate-200 px-3 py-2 text-[11px] font-medium text-slate-500">
                                                    STDOUT
                                                </div>
                                                <pre className="max-h-48 overflow-auto whitespace-pre-wrap px-3 py-3 text-[12px] leading-5 text-slate-800">
                                                    {lastCommandOutput.result.stdout}
                                                </pre>
                                            </div>
                                        ) : null}

                                        {lastCommandOutput.result?.stderr ? (
                                            <div className="rounded-xl border border-slate-200 bg-white">
                                                <div className="border-b border-slate-200 px-3 py-2 text-[11px] font-medium text-slate-500">
                                                    STDERR
                                                </div>
                                                <pre className="max-h-40 overflow-auto whitespace-pre-wrap px-3 py-3 text-[12px] leading-5 text-slate-700">
                                                    {lastCommandOutput.result.stderr}
                                                </pre>
                                            </div>
                                        ) : null}
                                    </div>
                                </div>
                            ) : null}
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
                    }}
                />
            )}
        </div>
    )
}
