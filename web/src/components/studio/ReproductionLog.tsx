"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
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
import {
    buildCommandPreview,
    CliCommandRunner,
    type ActiveCommand as CliActiveCommand,
    type CommandResult as CliCommandResult,
    getCommandPresets,
    type LastCommandOutput as CliLastCommandOutput,
} from "./CliCommandRunner"
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

type StudioCommandPreviewOptions = {
    mode: Mode
    model: string
    continueLast: boolean
    resumeSession: string
    cliSessionId: string
    agent: string
    mcpConfig: string[]
    tools: string[]
    allowedTools: string[]
    addDirs: string[]
    settings: string
    effort: Exclude<EffortOption, "default"> | null
}

type SlashCommandItem = {
    id: string
    command: string
    label: string
    description: string
    group: "Modes" | "Actions" | "Views" | "Models" | "Options" | "Quick Commands"
    keywords: string[]
    icon: React.ElementType
    onSelect: (remainder: string) => void
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

function formatPreviewToken(token: string): string {
    if (!token) return '""'
    if (/\s|,|\{|\}|\[|\]|"/.test(token)) {
        return JSON.stringify(token)
    }
    return token
}

function slashifyModelAlias(alias: string): string {
    return `model-${alias.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "")}`
}

function appendPreviewOption(parts: string[], flag: string, values: string[]) {
    if (values.length === 0) return
    parts.push(flag, ...values)
}

function appendJoinedPreviewOption(parts: string[], flag: string, values: string[]) {
    if (values.length === 0) return
    parts.push(flag, values.join(","))
}

function buildStudioCommandPreview(
    runtimeInfo: StudioRuntimeInfo,
    options: StudioCommandPreviewOptions,
): string {
    if (runtimeInfo.source === "anthropic_api") {
        return "Fallback path: direct Anthropic API call, not Claude Code CLI"
    }
    if (runtimeInfo.source !== "claude_code") {
        return "Resolving Claude Code CLI surface..."
    }

    const normalizedModel = options.model.trim() || "sonnet"
    const permissionMode = effectivePermissionMode(options.mode, runtimeInfo.codeModeEnabled)
    const parts = ["claude", "--model", normalizedModel]

    if (options.continueLast) {
        parts.push("--continue")
    }

    if (options.resumeSession.trim()) {
        parts.push("--resume", options.resumeSession.trim())
    }

    if (options.cliSessionId.trim()) {
        parts.push("--session-id", options.cliSessionId.trim())
    }

    if (options.agent.trim()) {
        parts.push("--agent", options.agent.trim())
    }

    appendPreviewOption(parts, "--add-dir", options.addDirs)
    appendPreviewOption(parts, "--mcp-config", options.mcpConfig)
    appendJoinedPreviewOption(parts, "--tools", options.tools)
    appendJoinedPreviewOption(parts, "--allowed-tools", options.allowedTools)

    if (options.settings.trim()) {
        parts.push("--settings", options.settings.trim())
    }

    if (options.effort) {
        parts.push("--effort", options.effort)
    }

    if (permissionMode !== "default") {
        parts.push("--permission-mode", permissionMode)
    }

    parts.push("-p", "<prompt>", "--output-format", "stream-json", "--verbose")
    return parts.map(formatPreviewToken).join(" ")
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
    const [modelOption, setModelOption] = useState("sonnet")
    const [customModel, setCustomModel] = useState("")
    const [lastError, setLastError] = useState<string | null>(null)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)
    const [saving, setSaving] = useState(false)
    const [messageInput, setMessageInput] = useState("")
    const [activeCliCommand, setActiveCliCommand] = useState<CliActiveCommand | null>(null)
    const [chatDraftBeforeUtility, setChatDraftBeforeUtility] = useState("")
    const [codexMode, setCodexMode] = useState(false)
    const [runningCliCommand, setRunningCliCommand] = useState(false)
    const [lastCommandOutput, setLastCommandOutput] = useState<CliLastCommandOutput | null>(null)
    const [slashSelectedIndex, setSlashSelectedIndex] = useState(0)
    const [showWorkspaceSetup, setShowWorkspaceSetup] = useState(false)
    const [pendingAction, setPendingAction] = useState<"chat" | "delegate_codex" | null>(null)
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)
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

    const activeTask = tasks.find(t => t.id === activeTaskId)
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
    const utilityMode = commandMode || codexMode
    const commandRuntimeUnavailable = activeCliCommand
        ? activeCliCommand.runtime === "claude"
            ? runtimeInfo.source !== "claude_code"
            : !runtimeInfo.opencodeAvailable
        : false
    const advancedOptionsCount = useMemo(
        () =>
            [
                continueLast,
                Boolean(resumeSession.trim()),
                Boolean(cliSessionId.trim()),
                Boolean(agentOverride.trim()),
                parsedMcpConfig.length > 0,
                parsedTools.length > 0,
                parsedAllowedTools.length > 0,
                parsedAddDirs.length > 0,
                Boolean(settingsText.trim()),
                Boolean(selectedEffort),
            ].filter(Boolean).length,
        [
            agentOverride,
            cliSessionId,
            continueLast,
            parsedAddDirs.length,
            parsedAllowedTools.length,
            parsedMcpConfig.length,
            parsedTools.length,
            resumeSession,
            selectedEffort,
            settingsText,
        ],
    )
    const commandPreview = useMemo(
        () =>
            buildStudioCommandPreview(runtimeInfo, {
                mode,
                model: requestedModel,
                continueLast,
                resumeSession,
                cliSessionId,
                agent: agentOverride,
                mcpConfig: parsedMcpConfig,
                tools: parsedTools,
                allowedTools: parsedAllowedTools,
                addDirs: parsedAddDirs,
                settings: settingsText,
                effort: selectedEffort,
            }),
        [
            agentOverride,
            cliSessionId,
            continueLast,
            mode,
            parsedAddDirs,
            parsedAllowedTools,
            parsedMcpConfig,
            parsedTools,
            requestedModel,
            resumeSession,
            runtimeInfo,
            selectedEffort,
            settingsText,
        ],
    )
    const messagePlaceholder = runtimeLoading
        ? "Message Studio runtime..."
        : runtimeInfo.source === "anthropic_api"
            ? "Message fallback runtime..."
            : "Message Claude Code..."
    const composerPlaceholder = commandMode
        ? `Edit args for ${buildCommandPreview(activeCliCommand!.runtime, activeCliCommand!.preset, "").trim()} and press Enter to run`
        : codexMode
            ? "Describe the Codex subagent task and press Enter to delegate"
        : messagePlaceholder
    const composerPreview = commandMode
        ? buildCommandPreview(activeCliCommand!.runtime, activeCliCommand!.preset, messageInput)
        : codexMode
            ? `codex ${messageInput.trim() || "<task>"}`.trim()
        : commandPreview

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
        if (!requestedModel) {
            setLastError("Select a Claude Code model alias or enter a full custom model name.")
            return
        }
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
                    model: requestedModel,
                    paper: selectedPaper ? {
                        title: selectedPaper.title,
                        abstract: selectedPaper.abstract,
                        method_section: selectedPaper.methodSection,
                    } : undefined,
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

    const handleSelectCliCommand = (command: CliActiveCommand, nextArgs?: string, restoreDraft?: string) => {
        if (!utilityMode) {
            setChatDraftBeforeUtility(restoreDraft ?? messageInput)
        }

        setCodexMode(false)
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

    const activateCodexMode = (nextMessage = "", restoreDraft?: string) => {
        if (!utilityMode) {
            setChatDraftBeforeUtility(restoreDraft ?? messageInput)
        }

        setActiveCliCommand(null)
        setCodexMode(true)
        setMessageInput(nextMessage)
        setLastError(null)
    }

    const handleClearCodexMode = () => {
        setCodexMode(false)
        setMessageInput(chatDraftBeforeUtility)
        setChatDraftBeforeUtility("")
    }

    const executeCliCommand = async () => {
        if (!activeCliCommand || runningCliCommand || commandRuntimeUnavailable) return

        const preview = buildCommandPreview(activeCliCommand.runtime, activeCliCommand.preset, messageInput)
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
                    runtime: activeCliCommand.runtime,
                    command: activeCliCommand.preset.command,
                    args: messageInput.trim(),
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

    const handleComposerSubmit = () => {
        if (commandMode) {
            void executeCliCommand()
            return
        }

        if (codexMode) {
            void handleDelegateToCodex()
            return
        }

        void handleSendMessage()
    }

    const normalizedComposerInput = messageInput.trimStart()
    const slashPaletteActive = !utilityMode && normalizedComposerInput.startsWith("/")
    const slashPayload = slashPaletteActive ? normalizedComposerInput.slice(1) : ""
    const slashFirstSpace = slashPayload.search(/\s/)
    const slashToken =
        slashFirstSpace === -1 ? slashPayload : slashPayload.slice(0, slashFirstSpace)
    const slashRemainder =
        slashFirstSpace === -1 ? "" : slashPayload.slice(slashFirstSpace + 1).trimStart()
    const slashQuery = slashToken.toLowerCase()

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

    const slashCommands: SlashCommandItem[] = [
        {
            id: "mode-code",
            command: "code",
            label: "Code mode",
            description: "Switch the composer to edit and implementation mode.",
            group: "Modes",
            keywords: ["build", "fix", "edit"],
            icon: Code,
            onSelect: (remainder) => {
                setMode("Code")
                setMessageInput(remainder)
            },
        },
        {
            id: "mode-plan",
            command: "plan",
            label: "Plan mode",
            description: "Switch the composer to planning mode before execution.",
            group: "Modes",
            keywords: ["strategy", "outline", "design"],
            icon: LayoutDashboard,
            onSelect: (remainder) => {
                setMode("Plan")
                setMessageInput(remainder)
            },
        },
        {
            id: "mode-ask",
            command: "ask",
            label: "Ask mode",
            description: "Switch the composer to question and discussion mode.",
            group: "Modes",
            keywords: ["question", "discuss", "explain"],
            icon: MessageSquare,
            onSelect: (remainder) => {
                setMode("Ask")
                setMessageInput(remainder)
            },
        },
        {
            id: "action-codex",
            command: "codex",
            label: "Codex delegation",
            description: "Turn the composer into a Codex subagent delegation prompt.",
            group: "Actions",
            keywords: ["delegate", "subagent", "agent", "task"],
            icon: Bot,
            onSelect: (remainder) => {
                activateCodexMode(remainder, remainder)
            },
        },
        {
            id: "view-chat",
            command: "chat",
            label: "Open chat",
            description: "Stay in the console chat timeline.",
            group: "Views",
            keywords: ["console", "log", "messages"],
            icon: MessageSquare,
            onSelect: (remainder) => {
                onViewModeChange("log")
                setMessageInput(remainder)
            },
        },
        {
            id: "view-context",
            command: "context",
            label: "Open context",
            description: "Jump to the paper context and workspace preparation view.",
            group: "Views",
            keywords: ["pack", "paper", "workspace"],
            icon: Activity,
            onSelect: (remainder) => {
                onViewModeChange("context")
                setMessageInput(remainder)
            },
        },
        {
            id: "view-monitor",
            command: "monitor",
            label: "Open monitor",
            description: "Jump to the agent board and delegation monitor.",
            group: "Views",
            keywords: ["board", "agents", "subagents"],
            icon: LayoutDashboard,
            onSelect: (remainder) => {
                openAgentBoardWorkspace()
                setMessageInput(remainder)
            },
        },
        ...knownModelAliases.map((alias) => ({
            id: `model-${alias}`,
            command: slashifyModelAlias(alias),
            label: alias,
            description: `Switch Claude Code model to ${alias}.`,
            group: "Models" as const,
            keywords: ["model", "claude", "alias", alias],
            icon: Bot,
            onSelect: (remainder: string) => {
                setModelOption(alias)
                setCustomModel("")
                setMessageInput(remainder)
            },
        })),
        {
            id: "option-continue",
            command: "continue",
            label: "Continue last session",
            description: "Turn on --continue for the next Claude Code print-mode run.",
            group: "Options",
            keywords: ["resume", "session", "previous"],
            icon: ChevronRight,
            onSelect: (remainder) => {
                setContinueLast(true)
                setMessageInput(remainder)
            },
        },
        {
            id: "option-fresh",
            command: "fresh",
            label: "Fresh session",
            description: "Turn off --continue and start the next Claude Code run fresh.",
            group: "Options",
            keywords: ["new", "reset", "session"],
            icon: X,
            onSelect: (remainder) => {
                setContinueLast(false)
                setMessageInput(remainder)
            },
        },
        ...(["default", "low", "medium", "high", "max"] as const).map((value) => ({
            id: `option-effort-${value}`,
            command: `effort-${value}`,
            label: value === "default" ? "Default effort" : `${value} effort`,
            description:
                value === "default"
                    ? "Reset Claude Code effort to the runtime default."
                    : `Set Claude Code effort to ${value} for the next run.`,
            group: "Options" as const,
            keywords: ["effort", "thinking", "reasoning", value],
            icon: Activity,
            onSelect: (remainder: string) => {
                setEffort(value)
                setMessageInput(remainder)
            },
        })),
        ...getCommandPresets("claude").map((preset) => ({
            id: `slash-${preset.id}`,
            command: preset.id,
            label: preset.label,
            description: preset.helpText,
            group: "Quick Commands" as const,
            keywords: ["claude", "cli", "runtime", ...preset.label.split(/\s+/)],
            icon: Terminal,
            onSelect: (remainder: string) =>
                handleSelectCliCommand(
                    {
                        runtime: "claude",
                        preset,
                    },
                    remainder || preset.defaultArgs,
                    remainder,
                ),
        })),
        ...getCommandPresets("opencode").map((preset) => ({
            id: `slash-${preset.id}`,
            command: preset.id,
            label: preset.label,
            description: preset.helpText,
            group: "Quick Commands" as const,
            keywords: ["opencode", "cli", "runtime", ...preset.label.split(/\s+/)],
            icon: Terminal,
            onSelect: (remainder: string) =>
                handleSelectCliCommand(
                    {
                        runtime: "opencode",
                        preset,
                    },
                    remainder || preset.defaultArgs,
                    remainder,
                ),
        })),
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
        command.onSelect(slashRemainder)
        setSlashSelectedIndex(0)
    }

    const composerHelperText = commandMode
        ? "Command badge turns the composer into a terminal line. Clear it to return to chat."
        : codexMode
            ? "Codex badge sends this composer text as a real subagent delegation task."
            : missingCustomModel
                ? "Enter a full Claude Code model name before sending."
                : "Type / for modes, models, quick commands, and monitor navigation."
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
        codexMode
            ? {
                id: "codex",
                label: "Codex delegation",
                meta: "subagent",
                tone: "accent" as const,
                icon: Bot,
                onRemove: handleClearCodexMode,
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

            {consoleMode && (
                /* Rich Input Area - CodePilot Style */
                <div className="shrink-0 border-t border-slate-200 bg-[#f1f2ed] p-4">
                    <div className="overflow-hidden rounded-[22px] border border-slate-200 bg-[#e8ebe4] shadow-[0_18px_36px_rgba(15,23,42,0.05)]">
                        <div className="border-b border-slate-200 bg-[#edf0e8] px-4 py-3">
                            <div className="flex flex-wrap items-center gap-2">
                                <span
                                    className={cn(
                                        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
                                        runtimeInfo.source === "claude_code"
                                            ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                            : runtimeInfo.source === "anthropic_api"
                                                ? "border-amber-200 bg-amber-50 text-amber-700"
                                                : "border-slate-200 bg-slate-100 text-slate-600",
                                    )}
                                >
                                    {runtimeLoading ? "Checking runtime" : runtimeInfo.label}
                                </span>
                                <span
                                    className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-slate-200 bg-[#f7f8f4] px-2.5 py-1 text-xs text-slate-600"
                                    title={runtimeInfo.cwd || runtimeInfo.actualCwd || undefined}
                                >
                                    <FileCode className="h-3.5 w-3.5 text-slate-500" />
                                    <span className="truncate">{runtimeInfo.workspaceLabel}</span>
                                </span>
                                <span className="truncate text-[11px] text-slate-500">
                                    {runtimeLoading ? "Resolving Claude Code status..." : runtimeInfo.statusLabel}
                                </span>
                            </div>
                            <div className="mt-2 truncate font-mono text-[11px] text-slate-500">
                                {composerPreview}
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

                            {modelOption === "custom" && !utilityMode ? (
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
                                value={messageInput}
                                onChange={(e) => setMessageInput(e.target.value)}
                                placeholder={composerPlaceholder}
                                className="min-h-[78px] resize-none border-0 bg-transparent px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus-visible:ring-0"
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
                                            setMessageInput(slashRemainder)
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

                                        if (codexMode) {
                                            e.preventDefault()
                                            handleClearCodexMode()
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
                                                <div className="text-[11px] font-medium text-slate-800">Slash commands</div>
                                                <div className="mt-0.5 text-[11px] text-slate-500">
                                                    Modes, models, monitor jumps, Codex delegation, and quick CLI presets.
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
                                                    {(["Modes", "Actions", "Views", "Models", "Options", "Quick Commands"] as const).map((group) => {
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

                        <div className="border-t border-slate-200 bg-[#f1f3ee] px-3 py-3">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex flex-1 flex-wrap items-center gap-1.5">
                                    <Select value={mode} onValueChange={(value) => setMode(value as Mode)}>
                                        <SelectTrigger className="h-8 w-[104px] border-slate-200 bg-[#f7f8f4] text-xs text-slate-700">
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
                                        <SelectTrigger className="h-8 w-[148px] border-slate-200 bg-[#f7f8f4] text-xs text-slate-700">
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

                                    <Popover open={showAdvancedOptions} onOpenChange={setShowAdvancedOptions}>
                                        <PopoverTrigger asChild>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className={cn(
                                                    "relative h-8 w-8 shrink-0 rounded-md border border-transparent bg-transparent text-slate-500 hover:bg-[#e7e9e3] hover:text-slate-900",
                                                    (showAdvancedOptions || advancedOptionsCount > 0) &&
                                                        "border-slate-200 bg-[#e7e9e3] text-slate-900",
                                                )}
                                                title="Claude Code advanced options"
                                            >
                                                <Settings2 className="h-3.5 w-3.5" />
                                                {advancedOptionsCount > 0 ? (
                                                    <span className="absolute -right-1 -top-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-slate-700 px-1 text-[10px] text-white">
                                                        {advancedOptionsCount}
                                                    </span>
                                                ) : null}
                                            </Button>
                                        </PopoverTrigger>
                                        <PopoverContent
                                            align="start"
                                            side="top"
                                            sideOffset={10}
                                            className="w-[440px] max-w-[92vw] border-slate-200 bg-[#f7f8f4] p-0 text-slate-900 shadow-[0_18px_40px_rgba(15,23,42,0.10)]"
                                        >
                                            <div className="border-b border-slate-200 bg-[#f0f2ec] px-3 py-2.5">
                                                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                                                    Claude Code Options
                                                </div>
                                                <div className="mt-1 text-[12px] text-slate-600">
                                                    Print-mode flags that apply to the next Claude Code run.
                                                </div>
                                            </div>
                                            <ScrollArea className="max-h-[65vh]">
                                                <div className="grid gap-3 p-3 xl:grid-cols-2">
                                                    <label className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700">
                                                        <Checkbox
                                                            checked={continueLast}
                                                            onCheckedChange={(value) => setContinueLast(Boolean(value))}
                                                            className="border-slate-300 data-[state=checked]:border-slate-700 data-[state=checked]:bg-slate-700"
                                                        />
                                                        Continue most recent CLI session
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Resume session</span>
                                                        <Input
                                                            value={resumeSession}
                                                            onChange={(event) => setResumeSession(event.target.value)}
                                                            placeholder="Existing Claude session ID"
                                                            className="h-8 border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Session ID</span>
                                                        <Input
                                                            value={cliSessionId}
                                                            onChange={(event) => setCliSessionId(event.target.value)}
                                                            placeholder="Pin a UUID for this print-mode run"
                                                            className="h-8 border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Agent override</span>
                                                        <Input
                                                            value={agentOverride}
                                                            onChange={(event) => setAgentOverride(event.target.value)}
                                                            placeholder="reviewer / planner / custom agent"
                                                            className="h-8 border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Effort</span>
                                                        <Select value={effort} onValueChange={(value) => setEffort(value as EffortOption)}>
                                                            <SelectTrigger className="h-8 border-slate-200 bg-white text-xs text-slate-700">
                                                                <SelectValue />
                                                            </SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="default">Default</SelectItem>
                                                                <SelectItem value="low">Low</SelectItem>
                                                                <SelectItem value="medium">Medium</SelectItem>
                                                                <SelectItem value="high">High</SelectItem>
                                                                <SelectItem value="max">Max</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Tools</span>
                                                        <Input
                                                            value={toolsText}
                                                            onChange={(event) => setToolsText(event.target.value)}
                                                            placeholder="Comma-separated, e.g. Bash,Edit,Read"
                                                            className="h-8 border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Allowed tools</span>
                                                        <Input
                                                            value={allowedToolsText}
                                                            onChange={(event) => setAllowedToolsText(event.target.value)}
                                                            placeholder='Comma-separated, e.g. "Bash(git:*),Read"'
                                                            className="h-8 border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">Add directories</span>
                                                        <Textarea
                                                            value={addDirsText}
                                                            onChange={(event) => setAddDirsText(event.target.value)}
                                                            placeholder={"One directory per line\n../shared-worktree"}
                                                            className="min-h-[72px] border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1">
                                                        <span className="text-[11px] font-medium text-slate-600">MCP config</span>
                                                        <Textarea
                                                            value={mcpConfigText}
                                                            onChange={(event) => setMcpConfigText(event.target.value)}
                                                            placeholder={"One file path or JSON object per line\n./mcp.json"}
                                                            className="min-h-[72px] border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>

                                                    <label className="space-y-1 xl:col-span-2">
                                                        <span className="text-[11px] font-medium text-slate-600">Settings</span>
                                                        <Textarea
                                                            value={settingsText}
                                                            onChange={(event) => setSettingsText(event.target.value)}
                                                            placeholder='Path or JSON blob for --settings, e.g. {"theme":"slate"}'
                                                            className="min-h-[84px] border-slate-200 bg-white text-xs text-slate-700"
                                                        />
                                                    </label>
                                                </div>
                                            </ScrollArea>
                                        </PopoverContent>
                                    </Popover>

                                    <CliCommandRunner
                                        key={viewMode === "commands" && !utilityMode ? "command-popover-open" : "command-popover-closed"}
                                        runtimeInfo={runtimeInfo}
                                        activeCommand={activeCliCommand}
                                        activeArgs={commandMode ? messageInput : ""}
                                        defaultOpen={viewMode === "commands" && !utilityMode}
                                        showActiveBadge={false}
                                        onSelectCommand={handleSelectCliCommand}
                                        onClearCommand={handleClearCliCommand}
                                    />

                                    {!utilityMode ? (
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-8 w-8 rounded-md border border-transparent bg-transparent text-slate-500 hover:bg-[#e7e9e3] hover:text-slate-900"
                                            onClick={() => activateCodexMode(messageInput, messageInput)}
                                            disabled={isBusy}
                                            title="Switch the composer into Codex delegation mode"
                                        >
                                            <Bot className="h-3.5 w-3.5" />
                                        </Button>
                                    ) : null}
                                </div>

                                <Button
                                    size="icon"
                                    className="h-9 w-9 shrink-0 rounded-full bg-slate-700 text-white hover:bg-slate-600"
                                    onClick={handleComposerSubmit}
                                    disabled={
                                        commandMode
                                            ? runningCliCommand || commandRuntimeUnavailable
                                            : codexMode
                                                ? !messageInput.trim() || isBusy
                                                : !messageInput.trim() || isBusy || missingCustomModel
                                    }
                                    title={
                                        commandMode
                                            ? commandRuntimeUnavailable
                                                ? "Selected command runtime is unavailable"
                                                : runningCliCommand
                                                    ? "Running command"
                                                    : "Run command in composer"
                                            : codexMode
                                                ? "Delegate this task to Codex"
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
                                    ) : codexMode ? (
                                        isBusy ? (
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                        ) : (
                                            <Bot className="h-4 w-4" />
                                        )
                                    ) : (
                                        <Send className="h-4 w-4" />
                                    )}
                                </Button>
                            </div>

                            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
                                <span>{composerHelperText}</span>
                                {!commandMode && !codexMode ? (
                                    <span className="hidden md:inline">
                                        Claude Code aliases: {knownModelAliases.join(", ")}
                                    </span>
                                ) : null}
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
                        setPendingAction(null)
                    }}
                />
            )}
        </div>
    )
}
