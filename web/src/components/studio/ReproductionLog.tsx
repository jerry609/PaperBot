"use client"

import { memo, useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react"
import { useRouter } from "next/navigation"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useStudioStore, AgentAction } from "@/lib/store/studio-store"
import { useProjectContext } from "@/lib/store/project-context"
import { readSSE } from "@/lib/sse"
import { CodeBlock } from "@/components/ai-elements"
import { DiffModal } from "./DiffViewer"
import { WorkspaceSetupDialog } from "./WorkspaceSetupDialog"
import { ContextDialogPanel } from "./ContextDialogPanel"
import { AgentBoard } from "./AgentBoard"
import {
    StudioPermissionSelector,
    type StudioPermissionProfile,
} from "./StudioPermissionSelector"
import {
    buildCommandPreview,
    type ActiveCommand as CliActiveCommand,
    type CommandResult as CliCommandResult,
    getCommandPresets,
} from "./CliCommandRunner"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import { parseStudioSlashCommand } from "@/lib/studio-slash"
import {
    buildStudioApprovalContinuePrompt,
    parseStudioApprovalRequest,
} from "@/lib/studio-approval"
import {
    getStudioBridgeDelegationTaskId,
    getStudioBridgeWorkerRunId,
    parseStudioBridgeResult,
    type StudioBridgeResult,
} from "@/lib/studio-bridge-result"
import { collapseToolActivityActions } from "@/lib/studio-chat-activity"
import {
    resolveDetectedModelSelection,
    type StudioRuntimeInfo,
} from "@/lib/studio-runtime"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { buildSubagentActivityGroups } from "@/lib/agent-events/subagent-groups"
import { cn } from "@/lib/utils"
import {
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
    Paperclip,
    Copy,
    Check,
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

function buildChatThreadTitle(message: string, attachmentNames: string[] = []): string {
    const singleLine = message.replace(/\s+/g, " ").trim()
    if (!singleLine) {
        if (attachmentNames.length === 1) {
            const label = attachmentNames[0]?.split("/").pop() || attachmentNames[0]
            return label.length <= 56 ? `File: ${label}` : `File: ${label.slice(0, 53)}...`
        }
        if (attachmentNames.length > 1) {
            return `${attachmentNames.length} attached files`
        }
        return "New thread"
    }
    return singleLine.length <= 56 ? singleLine : `${singleLine.slice(0, 53)}...`
}

function normalizeThinkingMessage(value: unknown): string | null {
    if (typeof value !== "string") return null
    const normalized = value.replace(/\s+/g, " ").trim()
    return normalized || null
}

type StudioSessionStatusPayload = {
    subtype?: string
    session_id?: string
    mode?: string
    requested_mode?: string
    permission_profile?: string
    permission_mode?: string
    chat_transport?: string
    preferred_chat_transport?: string
    claude_agent_sdk_available?: boolean
    cwd?: string
}

function isStudioMode(value: unknown): value is Mode {
    return value === "Code" || value === "Plan" || value === "Ask"
}

function formatStudioChatTransportLabel(transport: unknown): string {
    if (transport === "claude_agent_sdk") return "Agent SDK"
    if (transport === "claude_cli_print") return "CLI print"
    if (transport === "anthropic_api") return "API fallback"
    return "managed transport"
}

function formatStudioChatSurfaceLabel(surface: unknown): string {
    if (surface === "managed_session") return "Managed session"
    return "Managed session"
}

function buildStudioSessionInitMessage(payload: StudioSessionStatusPayload): string | null {
    const transport = formatStudioChatTransportLabel(payload.chat_transport)
    const effectiveMode = isStudioMode(payload.mode) ? payload.mode : null
    const requestedMode = isStudioMode(payload.requested_mode) ? payload.requested_mode : null
    const permissionSuffix =
        payload.permission_profile === "full_access"
            ? " Full access enabled."
            : ""

    const prefix = effectiveMode ? `[${effectiveMode}] ` : ""
    const modeSuffix =
        effectiveMode && requestedMode && effectiveMode !== requestedMode
            ? ` ${requestedMode} requested; running in ${effectiveMode}.`
            : ""
    const sdkSuffix =
        payload.chat_transport === "claude_cli_print" && payload.claude_agent_sdk_available !== true
            ? " Agent SDK route not installed yet."
            : ""

    return `${prefix}Managed chat connected on ${transport}.${modeSuffix}${permissionSuffix}${sdkSuffix}`.trim()
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
    requiresRuntimeSupport?: boolean
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

function buildCodexWorkerSmokePrompt(): string {
    return [
        "Verify whether you can delegate a tiny read-only task to the `codex-worker` project agent in this repository.",
        "",
        "Requirements:",
        "1. Check the available Claude project agents first.",
        "2. If `codex-worker` exists, delegate a minimal read-only task such as reporting the current git branch.",
        "3. Tell me whether delegation succeeded, which agent name you used, and what result came back.",
        "4. If delegation is unavailable, explain exactly what is missing instead of guessing.",
    ].join("\n")
}

function buildOpenCodeWorkerSmokePrompt(): string {
    return [
        "Verify whether this repository is configured so Claude Code can delegate to an OpenCode-backed worker.",
        "",
        "Requirements:",
        "1. Inspect the available Claude project agents first.",
        "2. If there is an OpenCode bridge agent, delegate a tiny read-only task and report the result.",
        "3. If there is no OpenCode bridge agent, stop and state that it is not configured for Claude Code delegation yet.",
        "4. Tell me the exact agent name you checked.",
    ].join("\n")
}

type ComposerUploadedFile = {
    id: string
    name: string
    type: string
    size: number
    data: string
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function readFileAsDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
            if (typeof reader.result === "string") {
                resolve(reader.result)
                return
            }
            reject(new Error(`Could not read ${file.name}`))
        }
        reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${file.name}`))
        reader.readAsDataURL(file)
    })
}

async function fileToComposerUpload(file: File): Promise<ComposerUploadedFile> {
    const dataUrl = await readFileAsDataUrl(file)
    const data = dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl
    return {
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type || "application/octet-stream",
        size: file.size,
        data,
    }
}

function buildUploadSignature(file: Pick<ComposerUploadedFile, "name" | "size" | "type">): string {
    return `${file.name}:${file.size}:${file.type}`
}

function mergeUploadedFiles(
    current: ComposerUploadedFile[],
    incoming: ComposerUploadedFile[],
): ComposerUploadedFile[] {
    const next = [...current]
    const seen = new Set(current.map((file) => buildUploadSignature(file)))

    for (const file of incoming) {
        const signature = buildUploadSignature(file)
        if (seen.has(signature)) continue
        seen.add(signature)
        next.push(file)
    }

    return next
}

async function readResponseDetail(res: Response, fallback: string): Promise<string> {
    const text = await res.text()
    try {
        const payload = JSON.parse(text) as { detail?: string }
        return payload.detail || fallback
    } catch {
        return text || fallback
    }
}

function buildVisibleActions(actions: AgentAction[]): AgentAction[] {
    const collapsedActions = collapseToolActivityActions(actions)
    return collapsedActions.filter((action, index) => {
        if (action.type !== "thinking") return true
        if (!isGenericThinkingMessage(action.content)) return true
        return index === collapsedActions.length - 1
    })
}

function summarizeToolPayload(payload: unknown): string | null {
    if (!payload || typeof payload !== "object") {
        if (typeof payload === "string" && payload.trim()) {
            return payload.trim().replace(/\s+/g, " ").slice(0, 120)
        }
        return null
    }

    const record = payload as Record<string, unknown>
    const priorityKeys = [
        "summary",
        "status",
        "task_kind",
        "path",
        "file_path",
        "filename",
        "target_path",
        "target_file",
        "command",
        "query",
        "pattern",
        "description",
        "prompt",
        "task_title",
        "message",
        "instructions",
        "assignee",
        "subagent_type",
    ]
    for (const key of priorityKeys) {
        const value = record[key]
        if (typeof value === "string" && value.trim()) {
            return value.trim().replace(/\s+/g, " ").slice(0, 120)
        }
    }

    return null
}

function formatBridgeTaskKind(taskKind: StudioBridgeResult["taskKind"]): string {
    if (taskKind === "approval_required") return "Approval"
    if (taskKind === "code") return "Code"
    if (taskKind === "review") return "Review"
    if (taskKind === "research") return "Research"
    if (taskKind === "plan") return "Plan"
    if (taskKind === "ops") return "Ops"
    if (taskKind === "failure") return "Failure"
    return "Result"
}

function formatBridgeStatus(status: StudioBridgeResult["status"]): string {
    if (status === "approval_required") return "approval"
    if (status === "completed") return "completed"
    if (status === "partial") return "partial"
    if (status === "failed") return "failed"
    return status
}

function bridgeStatusClassName(status: StudioBridgeResult["status"]): string {
    if (status === "completed") return "border-emerald-200 bg-emerald-50 text-emerald-700"
    if (status === "partial") return "border-amber-200 bg-amber-50 text-amber-700"
    if (status === "approval_required") return "border-amber-200 bg-amber-50 text-amber-700"
    if (status === "failed") return "border-rose-200 bg-rose-50 text-rose-700"
    return "border-slate-200 bg-[#f8faf6] text-slate-500"
}

function bridgePayloadMetricBadges(bridgeResult: StudioBridgeResult): string[] {
    const payload = bridgeResult.payload
    const badges: string[] = []
    const metricKeys: Array<[keyof typeof payload, string]> = [
        ["files_changed", "files"],
        ["files_created", "created"],
        ["findings", "findings"],
        ["claims", "claims"],
        ["steps", "steps"],
        ["commands", "commands"],
        ["checks", "checks"],
        ["tests_run", "tests"],
    ]

    for (const [key, label] of metricKeys) {
        const value = payload[key]
        if (Array.isArray(value) && value.length > 0) {
            badges.push(`${value.length} ${label}`)
        }
    }

    if (bridgeResult.artifacts.length > 0) {
        badges.push(`${bridgeResult.artifacts.length} artifact${bridgeResult.artifacts.length === 1 ? "" : "s"}`)
    }

    return badges
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
                "inline-flex min-w-0 max-w-[min(100%,34rem)] items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs shadow-sm",
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

function MessageAttachmentPill({
    name,
    type,
    size,
}: {
    name: string
    type: string
    size: number
}) {
    return (
        <div className="inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-[10px] text-slate-600 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]">
            {type.startsWith("image/") ? (
                <Paperclip className="h-3 w-3 shrink-0 text-slate-400" />
            ) : (
                <FileText className="h-3 w-3 shrink-0 text-slate-400" />
            )}
            <span className="truncate font-medium text-slate-700">{name}</span>
            <span className="shrink-0 text-[10px] uppercase tracking-[0.12em] text-slate-400">
                {formatFileSize(size)}
            </span>
        </div>
    )
}

interface ActionItemProps {
    action: AgentAction
    onViewDiff: (action: AgentAction) => void
    onOpenMonitor?: (delegationTaskId?: string) => void
    onApproveApprovalRequest?: (action: AgentAction) => void
    approvalPending?: boolean
}

type MarkdownActionBlockTone = "default" | "error"

function MarkdownActionBlock({
    rawContent,
    renderContent,
    label,
    tone = "default",
}: {
    rawContent: string
    renderContent?: string
    label?: string
    tone?: MarkdownActionBlockTone
}) {
    const [copied, setCopied] = useState(false)

    const copy = useCallback(async () => {
        try {
            await navigator.clipboard.writeText(rawContent)
            setCopied(true)
            window.setTimeout(() => setCopied(false), 1200)
        } catch {
            setCopied(false)
        }
    }, [rawContent])

    return (
        <div
            className={cn(
                "max-w-[86%] overflow-hidden rounded-[16px] border shadow-[0_1px_0_rgba(255,255,255,0.6)_inset]",
                tone === "error"
                    ? "border-rose-200 bg-rose-50"
                    : "border-slate-200/90 bg-[#f7f8f4]",
            )}
        >
            <div
                className={cn(
                    "flex items-center justify-between gap-2 border-b px-2.5 py-1",
                    tone === "error" ? "border-rose-200/80 bg-rose-100/60" : "border-slate-200 bg-[#eef1ea]",
                )}
            >
                <span
                    className={cn(
                        "truncate text-[10px] font-medium uppercase tracking-[0.14em]",
                        tone === "error" ? "text-rose-700" : "text-slate-500",
                    )}
                >
                    {label ?? (tone === "error" ? "Error" : "Claude Code")}
                </span>
                <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className={cn(
                        "h-5.5 gap-1 rounded-full px-1.5 text-[9px]",
                        tone === "error"
                            ? "text-rose-700 hover:bg-rose-100 hover:text-rose-800"
                            : "text-slate-500 hover:bg-white hover:text-slate-700",
                    )}
                    onClick={copy}
                >
                    {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                    {copied ? "Copied" : "Copy"}
                </Button>
            </div>
            <div className="px-2.5 py-2">
                <div className="space-y-1.5 text-[11px] leading-5 text-slate-800">
                    <Markdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ className, ...props }) => (
                                <h1 className={cn("mb-1.5 text-[14px] font-semibold text-slate-900", className)} {...props} />
                            ),
                            h2: ({ className, ...props }) => (
                                <h2 className={cn("mb-1.5 text-[12px] font-semibold text-slate-900", className)} {...props} />
                            ),
                            h3: ({ className, ...props }) => (
                                <h3 className={cn("mb-1 text-[11px] font-semibold text-slate-900", className)} {...props} />
                            ),
                            p: ({ className, ...props }) => (
                                <p className={cn("my-0 whitespace-pre-wrap text-[11px] leading-5 text-slate-800", className)} {...props} />
                            ),
                            ul: ({ className, ...props }) => (
                                <ul className={cn("my-1.5 list-disc space-y-0.5 pl-4 text-[11px] text-slate-800", className)} {...props} />
                            ),
                            ol: ({ className, ...props }) => (
                                <ol className={cn("my-1.5 list-decimal space-y-0.5 pl-4 text-[11px] text-slate-800", className)} {...props} />
                            ),
                            li: ({ className, ...props }) => (
                                <li className={cn("leading-5", className)} {...props} />
                            ),
                            a: ({ className, ...props }) => (
                                <a
                                    {...props}
                                    className={cn("text-slate-900 underline underline-offset-2", className)}
                                    target="_blank"
                                    rel="noreferrer"
                                />
                            ),
                            strong: ({ className, ...props }) => (
                                <strong className={cn("font-semibold text-slate-900", className)} {...props} />
                            ),
                            blockquote: ({ className, ...props }) => (
                                <blockquote
                                    className={cn("my-1.5 border-l-2 border-slate-300 pl-2.5 text-[11px] text-slate-600", className)}
                                    {...props}
                                />
                            ),
                            pre: ({ className, ...props }) => (
                                <pre
                                    className={cn(
                                        "my-0 overflow-auto rounded-[12px] border border-slate-200 bg-slate-950 px-2.5 py-2 text-[10px] leading-[18px] text-slate-100",
                                        className,
                                    )}
                                    {...props}
                                />
                            ),
                            code: ({ className, ...props }) => {
                                const blockCode = typeof className === "string" && className.includes("language-")
                                return (
                                    <code
                                        className={cn(
                                            blockCode
                                                ? "bg-transparent p-0 font-mono text-inherit"
                                                : "rounded bg-white px-1 py-0.5 font-mono text-[10px] text-slate-800",
                                            className,
                                        )}
                                        {...props}
                                    />
                                )
                            },
                            table: ({ className, ...props }) => (
                                <table className={cn("my-1.5 w-full border-collapse text-[11px]", className)} {...props} />
                            ),
                            th: ({ className, ...props }) => (
                                <th
                                    className={cn(
                                        "border border-slate-200 bg-white px-1.5 py-1 text-left text-[10px] font-semibold text-slate-700",
                                        className,
                                    )}
                                    {...props}
                                />
                            ),
                            td: ({ className, ...props }) => (
                                <td className={cn("border border-slate-200 px-1.5 py-1 align-top text-[10px] text-slate-700", className)} {...props} />
                            ),
                        }}
                    >
                        {renderContent ?? rawContent}
                    </Markdown>
                </div>
            </div>
        </div>
    )
}

function ActionItem({
    action,
    onViewDiff,
    onOpenMonitor,
    onApproveApprovalRequest,
    approvalPending = false,
}: ActionItemProps) {
    const [expanded, setExpanded] = useState(false)
    const attachments = action.metadata?.attachments ?? []
    const bridgeResult = action.metadata?.bridgeResult ?? null
    const hasExpandableContent = Boolean(action.metadata?.params || action.metadata?.result || bridgeResult)
    const hasToolResult = action.metadata?.result !== undefined || bridgeResult !== null
    const commandOutput = action.metadata?.commandOutput
    const toolSummary =
        bridgeResult?.summary ??
        summarizeToolPayload(action.metadata?.params) ??
        summarizeToolPayload(action.metadata?.result)
    const stringifyPayload = (payload: unknown): string =>
        typeof payload === "string" ? payload : JSON.stringify(payload, null, 2) || ""

    if (action.type === "user") {
        const hasTextContent = action.content.trim().length > 0
        if (!hasTextContent && attachments.length === 0) {
            return null
        }

        return (
            <div className="flex justify-end pb-2">
                <div className="flex max-w-[86%] flex-col items-end gap-1">
                    {attachments.length > 0 ? (
                        <div className="flex flex-wrap justify-end gap-1">
                            {attachments.map((attachment) => (
                                <MessageAttachmentPill
                                    key={`${attachment.name}:${attachment.size}:${attachment.type}`}
                                    {...attachment}
                                />
                            ))}
                        </div>
                    ) : null}
                    {hasTextContent ? (
                        <div className="rounded-[16px] bg-slate-700 px-3 py-1.5 text-[11px] leading-[18px] text-white shadow-sm">
                            <p className="whitespace-pre-wrap">{action.content}</p>
                        </div>
                    ) : null}
                </div>
            </div>
        )
    }

    if (action.type === "thinking") {
        return (
            <div className="pb-1">
                <div className="inline-flex max-w-[86%] items-start gap-1.5 rounded-full border border-slate-200 bg-white/85 px-2.5 py-1 text-[10px] leading-4 text-slate-500 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]">
                    <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin text-slate-400" />
                    <span className="shrink-0 uppercase tracking-[0.12em] text-slate-400">thinking</span>
                    <span className="min-w-0 whitespace-pre-wrap text-slate-500">
                        {action.content}
                    </span>
                </div>
            </div>
        )
    }

    if (action.type === "activity_summary" && action.metadata?.activitySummary) {
        const summary = action.metadata.activitySummary
        const countBadges = [
            summary.counts.read ? `${summary.counts.read} reads` : null,
            summary.counts.search ? `${summary.counts.search} searches` : null,
            summary.counts.command ? `${summary.counts.command} commands` : null,
            summary.counts.write ? `${summary.counts.write} edits` : null,
            summary.counts.delegation ? `${summary.counts.delegation} delegations` : null,
            summary.counts.web ? `${summary.counts.web} web` : null,
        ].filter(Boolean) as string[]

        const sharedClassName =
            "block w-full max-w-[86%] rounded-[18px] border border-slate-200 bg-white/90 px-2 py-1.5 text-left shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]"

        const content = (
            <>
                <div className="flex items-center gap-1.5">
                    <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-[#f3f5ef]">
                        <Activity className="h-3 w-3 text-slate-500" />
                    </div>
                    <span className="min-w-0 flex-1 truncate text-[10px] font-medium text-slate-800">
                        {summary.label}
                    </span>
                    <span className="shrink-0 rounded-full border border-slate-200 bg-[#eef1ea] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                        {summary.totalTools} action{summary.totalTools === 1 ? "" : "s"}
                    </span>
                    <span
                        className={cn(
                            "shrink-0 rounded-full border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em]",
                            summary.status === "done"
                                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                : "border-slate-200 bg-[#f8faf6] text-slate-500",
                        )}
                    >
                        {summary.status}
                    </span>
                </div>

                {countBadges.length > 0 ? (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                        {countBadges.map((badge) => (
                            <span
                                key={badge}
                                className="rounded-full border border-slate-200 bg-[#f7f8f4] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500"
                            >
                                {badge}
                            </span>
                        ))}
                    </div>
                ) : null}

                {summary.recent.length > 0 ? (
                    <div className="mt-1.5 text-[9px] leading-4 text-slate-500">
                        Recent: {summary.recent.join(" · ")}
                    </div>
                ) : null}

                <div className="mt-1 text-[9px] text-slate-400">
                    Full tool activity stays in Monitor.
                </div>
            </>
        )

        return (
            <div className="pb-1.5">
                {onOpenMonitor ? (
                    <button
                        type="button"
                        onClick={() => onOpenMonitor(summary.delegationTaskId)}
                        className={`${sharedClassName} transition-colors hover:bg-white`}
                    >
                        {content}
                    </button>
                ) : (
                    <div className={sharedClassName}>{content}</div>
                )}
            </div>
        )
    }

    if (action.type === "approval_request" && action.metadata?.approvalRequest) {
        const approval = action.metadata.approvalRequest
        const canApprove = Boolean(approval.cliSessionId) && !approvalPending
        const approvalBridgeResult = approval.bridgeResult ?? null
        const approvalMonitorTaskId =
            getStudioBridgeDelegationTaskId(approvalBridgeResult) ??
            approval.toolId ??
            null
        const approvalWorkerRunId = getStudioBridgeWorkerRunId(approvalBridgeResult)

        return (
            <div className="pb-1.5">
                <div className="max-w-[86%] rounded-[18px] border border-amber-200 bg-amber-50/90 px-2.5 py-2 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]">
                    <div className="flex items-start gap-2">
                        <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-amber-200 bg-white">
                            <AlertCircle className="h-3 w-3 text-amber-700" />
                        </div>
                        <div className="min-w-0 flex-1">
                            <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-amber-800">
                                Approval required
                            </div>
                            {approvalBridgeResult ? (
                                <div className="mt-0.5 flex flex-wrap items-center gap-1">
                                    <span className="rounded-full border border-amber-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-amber-800">
                                        {formatBridgeTaskKind(approvalBridgeResult.taskKind)}
                                    </span>
                                    <span className="rounded-full border border-amber-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-amber-800">
                                        {approvalBridgeResult.executor}
                                    </span>
                                </div>
                            ) : null}
                            <p className="mt-1 whitespace-pre-wrap text-[11px] leading-[18px] text-amber-950">
                                {approval.message}
                            </p>
                            {approval.command ? (
                                <div className="mt-1.5 rounded-[12px] border border-amber-200 bg-white px-2 py-1.5">
                                    <code className="break-all text-[10px] text-slate-800">
                                        {approval.command}
                                    </code>
                                </div>
                            ) : null}
                            <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                                <Button
                                    type="button"
                                    size="sm"
                                    className="h-6 rounded-full bg-amber-700 px-2.5 text-[10px] text-white hover:bg-amber-800"
                                    onClick={() => onApproveApprovalRequest?.(action)}
                                    disabled={!canApprove}
                                >
                                    {approvalPending ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : null}
                                    Approve & Continue
                                </Button>
                                {onOpenMonitor && approvalMonitorTaskId ? (
                                    <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        className="h-6 rounded-full border-amber-200 bg-white px-2.5 text-[10px] text-amber-900 hover:bg-amber-100"
                                        onClick={() => onOpenMonitor(approvalMonitorTaskId)}
                                    >
                                        Open in Monitor
                                    </Button>
                                ) : null}
                                <span className="text-[9px] leading-4 text-amber-800/80">
                                    {approval.cliSessionId
                                        ? "Resumes the parent Claude session in full access mode."
                                        : "Missing Claude session id for resume."}
                                </span>
                                {approval.workerAgentId ? (
                                    <span className="rounded-full border border-amber-200 bg-white px-2 py-0.5 text-[9px] uppercase tracking-[0.12em] text-amber-800">
                                        worker {approval.workerAgentId}
                                    </span>
                                ) : null}
                                {approvalWorkerRunId ? (
                                    <span className="rounded-full border border-amber-200 bg-white px-2 py-0.5 text-[9px] uppercase tracking-[0.12em] text-amber-800">
                                        run {approvalWorkerRunId.slice(0, 18)}
                                    </span>
                                ) : null}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    if (action.type === "function_call" && action.metadata?.functionName) {
        const monitorTaskId =
            getStudioBridgeDelegationTaskId(bridgeResult) ??
            action.metadata?.toolId ??
            null
        const workerRunId = getStudioBridgeWorkerRunId(bridgeResult)
        return (
            <div className="pb-1">
                <div className="max-w-[86%] rounded-[18px] border border-slate-200 bg-white/90 px-2 py-1.5 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]">
                    <div className="flex items-center gap-1.5">
                        <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-[#f3f5ef]">
                            <Wrench className="h-3 w-3 text-slate-500" />
                        </div>
                        <code className="shrink-0 rounded-full bg-[#eef1ea] px-1.5 py-0.5 text-[9px] font-mono text-slate-700">
                            {action.metadata.functionName}()
                        </code>
                        {bridgeResult ? (
                            <span className="shrink-0 rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                                {formatBridgeTaskKind(bridgeResult.taskKind)}
                            </span>
                        ) : null}
                        {toolSummary ? (
                            <span className="min-w-0 flex-1 truncate text-[10px] text-slate-500">
                                {toolSummary}
                            </span>
                        ) : (
                            <span className="min-w-0 flex-1 text-[10px] text-slate-400">
                                {hasToolResult ? "done" : "running"}
                            </span>
                        )}
                        <span
                            className={cn(
                                "shrink-0 rounded-full border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em]",
                                hasToolResult
                                    ? bridgeResult
                                        ? bridgeStatusClassName(bridgeResult.status)
                                        : "border-emerald-200 bg-emerald-50 text-emerald-700"
                                    : "border-slate-200 bg-[#f8faf6] text-slate-500",
                            )}
                        >
                            {hasToolResult ? (bridgeResult ? formatBridgeStatus(bridgeResult.status) : "done") : "running"}
                        </span>
                        {hasExpandableContent ? (
                            <button
                                type="button"
                                onClick={() => setExpanded((current) => !current)}
                                className="rounded-full p-0.5 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
                                title={expanded ? "Collapse tool output" : "Expand tool output"}
                            >
                                {expanded ? (
                                    <ChevronDown className="h-3.5 w-3.5" />
                                ) : (
                                    <ChevronRight className="h-3.5 w-3.5" />
                                )}
                            </button>
                        ) : null}
                    </div>
                    {bridgeResult ? (
                        <div className="mt-1.5 rounded-[12px] border border-slate-200 bg-[#f8faf5] px-2 py-1.5">
                            <div className="text-[10px] leading-[18px] text-slate-800">
                                {bridgeResult.summary}
                            </div>
                            <div className="mt-1.5 flex flex-wrap items-center gap-1">
                                <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                                    {bridgeResult.executor}
                                </span>
                                {workerRunId ? (
                                    <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                                        run {workerRunId.slice(0, 18)}
                                    </span>
                                ) : null}
                                {bridgePayloadMetricBadges(bridgeResult).map((badge) => (
                                    <span
                                        key={badge}
                                        className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500"
                                    >
                                        {badge}
                                    </span>
                                ))}
                            </div>
                            {bridgeResult.artifacts.length > 0 ? (
                                <div className="mt-1.5 flex flex-wrap gap-1">
                                    {bridgeResult.artifacts.slice(0, 4).map((artifact) => (
                                        <span
                                            key={`${artifact.kind}:${artifact.label}:${artifact.path ?? artifact.value ?? ""}`}
                                            className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] text-slate-600"
                                        >
                                            {artifact.label}
                                        </span>
                                    ))}
                                </div>
                            ) : null}
                            {onOpenMonitor && monitorTaskId ? (
                                <div className="mt-2">
                                    <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        className="h-6 rounded-full border-slate-200 bg-white px-2.5 text-[10px] text-slate-700 hover:bg-slate-50"
                                        onClick={() => onOpenMonitor(monitorTaskId)}
                                    >
                                        Open in Monitor
                                    </Button>
                                </div>
                            ) : null}
                        </div>
                    ) : null}
                    {expanded ? (
                        <div className="mt-1 space-y-1">
                            {Boolean(action.metadata.params) ? (
                                <CodeBlock title="Args" code={stringifyPayload(action.metadata.params)} />
                            ) : null}
                            {bridgeResult ? (
                                <CodeBlock title="Bridge Result" code={JSON.stringify(bridgeResult.raw, null, 2)} />
                            ) : null}
                            {Boolean(action.metadata.result) ? (
                                <CodeBlock title="Result" code={stringifyPayload(action.metadata.result)} />
                            ) : null}
                        </div>
                    ) : null}
                </div>
            </div>
        )
    }

    if (action.type === "file_change" && action.metadata?.filename) {
        return (
            <div className="pb-1">
                <button
                    type="button"
                    onClick={() => onViewDiff(action)}
                    className="flex max-w-[86%] items-center gap-1.5 rounded-[16px] border border-slate-200 bg-white/90 px-2 py-1 text-left text-[9px] text-slate-600 shadow-[0_1px_0_rgba(255,255,255,0.75)_inset] transition-colors hover:bg-white"
                >
                    <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-[#f3f5ef]">
                        <FileCode className="h-3 w-3 text-slate-500" />
                    </div>
                    <span className="truncate font-mono text-[10px] text-slate-700">{action.metadata.filename}</span>
                    <span className="shrink-0 rounded-full bg-[#eef1ea] px-1.5 py-0.5 text-[9px]">
                        <span className="text-emerald-700">+{action.metadata.linesAdded || 0}</span>
                        <span className="mx-0.5 text-slate-400">/</span>
                        <span className="text-rose-700">-{action.metadata.linesDeleted || 0}</span>
                    </span>
                </button>
            </div>
        )
    }

    if (action.type === "error") {
        const renderContent =
            commandOutput?.kind === "stdout" || commandOutput?.kind === "stderr"
                ? `\`\`\`\n${action.content}\n\`\`\``
                : action.content
        return (
            <div className="pb-1.5">
                <MarkdownActionBlock
                    rawContent={action.content}
                    renderContent={renderContent}
                    label={commandOutput?.title ?? "Error"}
                    tone="error"
                />
            </div>
        )
    }

    if (action.type === "complete") {
        return (
            <div className="pb-1.5">
                <span className="inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[9px] font-medium uppercase tracking-[0.12em] text-emerald-700">
                    done
                </span>
            </div>
        )
    }

    if (action.type === "text") {
        const renderContent =
            commandOutput?.kind === "stdout" || commandOutput?.kind === "stderr"
                ? `\`\`\`\n${action.content}\n\`\`\``
                : action.content
        return (
            <div className="pb-1.5">
                <MarkdownActionBlock
                    rawContent={action.content}
                    renderContent={renderContent}
                    label={commandOutput?.title ?? "Claude Code"}
                />
            </div>
        )
    }

    return (
        <div className="pb-1.5">
            <MarkdownActionBlock rawContent={action.content} label="Claude Code" />
        </div>
    )
}

const MemoizedActionItem = memo(ActionItem)

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
    const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
    const toolCalls = useAgentEventStore((state) => state.toolCalls)
    const openWorkerRun = useAgentEventStore((state) => state.openWorkerRun)
    const activeFileData = activeFile ? files[activeFile] : null

    const selectedPaper = useMemo(() =>
        selectedPaperId ? papers.find(p => p.id === selectedPaperId) ?? null : null,
        [papers, selectedPaperId]
    )

    const [status, setStatus] = useState<StepStatus>("idle")
    const [mode, setMode] = useState<Mode>("Code")
    const [modelOption, setModelOption] = useState("sonnet")
    const [customModel, setCustomModel] = useState("")
    const [permissionProfile, setPermissionProfile] = useState<StudioPermissionProfile>("default")
    const [lastError, setLastError] = useState<string | null>(null)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)
    const [saving, setSaving] = useState(false)
    const [messageInput, setMessageInput] = useState("")
    const [uploadedFiles, setUploadedFiles] = useState<ComposerUploadedFile[]>([])
    const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null)
    const composerFileInputRef = useRef<HTMLInputElement | null>(null)
    const slashPaletteListRef = useRef<HTMLDivElement | null>(null)
    const [composerCursor, setComposerCursor] = useState(0)
    const [activeCliCommand, setActiveCliCommand] = useState<CliActiveCommand | null>(null)
    const [chatDraftBeforeUtility, setChatDraftBeforeUtility] = useState("")
    const [runningCliCommand, setRunningCliCommand] = useState(false)
    const [runningApprovalActionId, setRunningApprovalActionId] = useState<string | null>(null)
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
    const modelSelectionInitializedRef = useRef(false)
    const modelSelectionDirtyRef = useRef(false)
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

    const applyModelSelection = useCallback(
        (
            nextModelOption: string,
            nextCustomModel = "",
            { markDirty = true }: { markDirty?: boolean } = {},
        ) => {
            if (markDirty) {
                modelSelectionDirtyRef.current = true
            }
            setModelOption(nextModelOption)
            setCustomModel(nextCustomModel)
        },
        [],
    )

    useEffect(() => {
        if (runtimeLoading || modelSelectionInitializedRef.current || modelSelectionDirtyRef.current) return

        const detectedSelection = resolveDetectedModelSelection(
            runtimeInfo.detectedDefaultModel,
            knownModelAliases,
        )

        if (detectedSelection) {
            applyModelSelection(detectedSelection.modelOption, detectedSelection.customModel, {
                markDirty: false,
            })
        } else if (!knownModelAliases.includes(modelOption)) {
            applyModelSelection(knownModelAliases[0] ?? "sonnet", "", {
                markDirty: false,
            })
        }

        modelSelectionInitializedRef.current = true
    }, [
        applyModelSelection,
        knownModelAliases,
        modelOption,
        runtimeInfo.detectedDefaultModel,
        runtimeLoading,
    ])

    useEffect(() => {
        if (modelOption === "custom") return
        if (!knownModelAliases.includes(modelOption)) {
            setModelOption(knownModelAliases[0] ?? "sonnet")
        }
    }, [knownModelAliases, modelOption])

    useEffect(() => {
        if (runtimeLoading) return
        if (runtimeInfo.codeModeEnabled === false && mode === "Code") {
            setMode("Plan")
        }
    }, [mode, runtimeInfo.codeModeEnabled, runtimeLoading])

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
    const visibleActions = useMemo(
        () => (visibleTask ? buildVisibleActions(visibleTask.actions) : []),
        [visibleTask],
    )
    const workerRunIdByDelegationTaskId = useMemo(() => {
        const groups = buildSubagentActivityGroups(codexDelegations, toolCalls)
        return new Map(
            groups
                .filter((group) => group.taskId.trim().length > 0)
                .map((group) => [group.taskId, group.workerRunId]),
        )
    }, [codexDelegations, toolCalls])
    const projectDir = selectedPaper?.outputDir || lastGenCodeResult?.outputDir || null
    const isBusy = status === "running"
    const runtimeLabel = runtimeLoading
        ? "Studio runtime"
        : `${runtimeInfo.label} · ${runtimeInfo.statusLabel}`
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
            ? "Message managed fallback..."
            : "Message Claude Code shell..."
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

    const prepareProjectDirForChat = useCallback(async (directory: string) => {
        const res = await fetch("/api/runbook/project-dir/prepare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_dir: directory,
                create_if_missing: true,
            }),
        })

        if (!res.ok) {
            throw new Error(await readResponseDetail(res, `Directory is not available (${res.status})`))
        }

        const data = await res.json() as { project_dir?: string }
        return data.project_dir || directory
    }, [])

    const runChatWithDir = async (targetDir: string) => {
        // Chat with specified directory - called after workspace setup
        const message = messageInput.trim()
        if (!message && uploadedFiles.length === 0) return
        if (!requestedModel) {
            setLastError("Select a Claude Code model alias or enter a full custom model name.")
            return
        }
        const pendingUploadedFiles = [...uploadedFiles]
        await handleSendMessageWithDir(message, targetDir, pendingUploadedFiles)
    }

    const handleSendMessage = async () => {
        if ((!messageInput.trim() && uploadedFiles.length === 0) || isBusy) return
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
        const pendingUploadedFiles = [...uploadedFiles]
        await handleSendMessageWithDir(message, projectDir || undefined, pendingUploadedFiles)
    }

    const streamChatTurn = async ({
        taskId,
        userMessageContent,
        effectiveMessage,
        outgoingMode,
        preparedProjectDir,
        selectedUploadedFiles = [],
        existingHistory = [],
        permissionProfileOverride = permissionProfile,
        resumeSessionOverride,
        cliSessionIdOverride,
    }: {
        taskId: string
        userMessageContent: string
        effectiveMessage: string
        outgoingMode: Mode
        preparedProjectDir?: string
        selectedUploadedFiles?: ComposerUploadedFile[]
        existingHistory?: Array<{ role: "user" | "assistant"; content: string }>
        permissionProfileOverride?: StudioPermissionProfile
        resumeSessionOverride?: string
        cliSessionIdOverride?: string
    }) => {
        let assistantResponse = ""
        let assistantHistoryCommitted = false
        let negotiatedMode: Mode = outgoingMode
        let currentCliSessionId = cliSessionIdOverride?.trim() || resumeSessionOverride?.trim() || ""
        const chatProjectDir = outgoingMode === "Code" ? preparedProjectDir : undefined
        const initialThinking = `[${outgoingMode}] Sending to ${runtimeLabel}...`

        addAction(
            taskId,
            {
                type: "user",
                content: userMessageContent.trim(),
                metadata:
                    selectedUploadedFiles.length > 0
                        ? {
                            attachments: selectedUploadedFiles.map((file) => ({
                                name: file.name,
                                type: file.type,
                                size: file.size,
                            })),
                        }
                        : undefined,
            },
        )
        appendTaskHistory(taskId, { role: "user", content: effectiveMessage })
        upsertThinkingAction(taskId, initialThinking)
        updateTaskStatus(taskId, "running")

        try {
            const res = await fetch("/api/studio/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: effectiveMessage,
                    mode: outgoingMode,
                    model: requestedModel,
                    permission_profile: permissionProfileOverride,
                    paper: selectedPaper ? {
                        title: selectedPaper.title,
                        abstract: selectedPaper.abstract,
                        method_section: selectedPaper.methodSection,
                    } : undefined,
                    history: existingHistory,
                    uploaded_files: selectedUploadedFiles,
                    session_id: taskId,
                    project_dir: chatProjectDir,
                    context_pack_id: contextPack?.context_pack_id,
                    continue_last: continueLast,
                    resume_session: resumeSessionOverride?.trim() || undefined,
                    cli_session_id: cliSessionIdOverride?.trim() || undefined,
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
                if (evt?.type === "status") {
                    const data = (evt.data ?? {}) as StudioSessionStatusPayload

                    if (data.subtype === "init") {
                        if (isStudioMode(data.mode)) {
                            negotiatedMode = data.mode
                            if (data.mode !== mode) {
                                setMode(data.mode)
                            }
                        }
                        if (data.permission_profile === "default" || data.permission_profile === "full_access") {
                            setPermissionProfile(data.permission_profile)
                        }
                        const initMessage = buildStudioSessionInitMessage(data)
                        if (initMessage) {
                            pushThinking(initMessage)
                        }
                        continue
                    }

                    if (data.subtype === "mode_changed") {
                        if (isStudioMode(data.mode)) {
                            negotiatedMode = data.mode
                            if (data.mode !== mode) {
                                setMode(data.mode)
                            }
                        }
                        const reason =
                            normalizeThinkingMessage((data as Record<string, unknown>).reason) ??
                            buildStudioSessionInitMessage(data)
                        if (reason) {
                            pushThinking(reason)
                        }
                        continue
                    }
                } else if (evt?.type === "progress") {
                    const data = (evt.data ?? {}) as Record<string, unknown>
                    const cliEvent = data.cli_event as string | undefined

                    if (cliEvent === "session_init") {
                        const nextCliSessionId =
                            typeof data.cli_session_id === "string" ? data.cli_session_id.trim() : ""
                        if (nextCliSessionId) {
                            currentCliSessionId = nextCliSessionId
                        }
                    } else if (cliEvent === "text") {
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
                                toolId: data.tool_id as string | undefined,
                                functionName: data.tool_name as string,
                                params: data.tool_input as Record<string, unknown>,
                            },
                        })
                    } else if (cliEvent === "tool_result") {
                        lastActionIsText = false
                        lastStreamActionType = "tool"
                        lastThinkingContent = null
                        const functionName = data.tool_name as string
                        const rawResult = data.content as string
                        const bridgeResult =
                            parseStudioBridgeResult((data as Record<string, unknown>).bridge_result) ??
                            parseStudioBridgeResult(rawResult)
                        const attached = attachResultToLatestFunctionCall(
                            taskId,
                            functionName,
                            rawResult,
                            data.tool_id as string | undefined,
                            bridgeResult ? { bridgeResult } : undefined,
                        )
                        if (!attached) {
                            addAction(taskId, {
                                type: "function_call",
                                content: `${functionName}()`,
                                metadata: {
                                    toolId: data.tool_id as string | undefined,
                                    functionName,
                                    result: rawResult,
                                    bridgeResult: bridgeResult ?? undefined,
                                },
                            })
                        }
                    } else if (cliEvent === "bridge_result") {
                        lastActionIsText = false
                        lastStreamActionType = "tool"
                        lastThinkingContent = null
                        const functionName = data.tool_name as string
                        const bridgeResult = parseStudioBridgeResult((data as Record<string, unknown>).bridge_result)
                        if (!bridgeResult) {
                            continue
                        }
                        const attached = attachResultToLatestFunctionCall(
                            taskId,
                            functionName,
                            undefined,
                            data.tool_id as string | undefined,
                            { bridgeResult },
                        )
                        if (!attached) {
                            addAction(taskId, {
                                type: "function_call",
                                content: `${functionName}()`,
                                metadata: {
                                    toolId: data.tool_id as string | undefined,
                                    functionName,
                                    bridgeResult,
                                },
                            })
                        }
                    } else if (cliEvent === "approval_required") {
                        lastActionIsText = false
                        lastStreamActionType = "other"
                        lastThinkingContent = null

                        const rawMessage = typeof data.message === "string" ? data.message : ""
                        const approvalBridgeResult =
                            parseStudioBridgeResult((data as Record<string, unknown>).bridge_result) ??
                            parseStudioBridgeResult(rawMessage)
                        const parsedApproval = parseStudioApprovalRequest(rawMessage)
                        const approvalMessage =
                            approvalBridgeResult?.summary ||
                            parsedApproval?.message ||
                            rawMessage.trim() ||
                            "This action requires approval before Claude can continue."
                        const explicitCommand =
                            typeof data.command === "string" && data.command.trim().length > 0
                                ? data.command.trim()
                                : parsedApproval?.command ?? undefined
                        const explicitWorkerAgentId =
                            typeof data.worker_agent_id === "string" && data.worker_agent_id.trim().length > 0
                                ? data.worker_agent_id.trim()
                                : parsedApproval?.workerAgentId ?? undefined
                        const resumeCliSessionId =
                            typeof data.cli_session_id === "string" && data.cli_session_id.trim().length > 0
                                ? data.cli_session_id.trim()
                                : currentCliSessionId || undefined

                        addAction(taskId, {
                            type: "approval_request",
                            content: approvalMessage,
                            metadata: {
                                approvalRequest: {
                                    message: approvalMessage,
                                    command: explicitCommand,
                                    cliSessionId: resumeCliSessionId,
                                    workerAgentId: explicitWorkerAgentId,
                                    toolId: typeof data.tool_id === "string" ? data.tool_id : undefined,
                                    toolName: typeof data.tool_name === "string" ? data.tool_name : undefined,
                                    bridgeResult: approvalBridgeResult ?? parsedApproval?.bridgeResult ?? null,
                                },
                            },
                        })
                    } else if (cliEvent === "thinking") {
                        pushThinking((data.text as string) || "Thinking...")
                    } else if (data.keepalive) {
                        continue
                    } else if (data.message) {
                        pushThinking(data.message as string)
                    } else if (data.delta) {
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
                            ? `[${negotiatedMode}] Completed in ${data.num_turns} turns`
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

    const handleSendMessageWithDir = async (
        message: string,
        targetDir?: string,
        selectedUploadedFiles: ComposerUploadedFile[] = [],
    ) => {
        const outgoingMode = mode

        if (outgoingMode === "Code" && runtimeInfo.codeModeEnabled === false) {
            setMode("Plan")
            setLastError("Code mode is disabled in this Studio runtime. Studio switched back to Plan.")
            return
        }

        let preparedProjectDir = targetDir
        if (outgoingMode === "Code" && targetDir) {
            try {
                preparedProjectDir = await prepareProjectDirForChat(targetDir)
                if (selectedPaperId) {
                    updatePaper(selectedPaperId, { outputDir: preparedProjectDir })
                }
            } catch (error) {
                setLastError(error instanceof Error ? error.message : "Failed to prepare project directory")
                return
            }
        }

        setStatus("running")
        setLastError(null)
        onViewModeChange("log")
        setMessageInput("")
        setUploadedFiles([])

        const effectiveMessage = message.trim() || "Please inspect the attached files."
        const threadTitle = buildChatThreadTitle(
            message,
            selectedUploadedFiles.map((file) => file.name),
        )
        const existingHistory = activeChatTask?.history ?? []
        const taskId =
            activeChatTask?.id ??
            addTask(threadTitle)

        if (activeChatTask && activeChatTask.history.length === 0 && activeChatTask.actions.length === 0) {
            renameTask(taskId, threadTitle)
        }

        await streamChatTurn({
            taskId,
            userMessageContent: message.trim(),
            effectiveMessage,
            outgoingMode,
            preparedProjectDir,
            selectedUploadedFiles,
            existingHistory,
            permissionProfileOverride: permissionProfile,
            resumeSessionOverride: resumeSession.trim() || undefined,
            cliSessionIdOverride: cliSessionId.trim() || undefined,
        })
    }

    const handleApproveApprovalRequest = async (action: AgentAction) => {
        const approval = action.metadata?.approvalRequest
        if (!activeChatTask || !approval?.cliSessionId) {
            setLastError("Missing Claude session id for approval resume.")
            return
        }
        if (status === "running") return

        const approvalMode: Mode =
            mode === "Code" && runtimeInfo.codeModeEnabled === false ? "Plan" : mode
        if (approvalMode !== mode) {
            setMode("Plan")
        }

        setRunningApprovalActionId(action.id)
        setStatus("running")
        setLastError(null)
        setPermissionProfile("full_access")
        onViewModeChange("log")

        try {
            await streamChatTurn({
                taskId: activeChatTask.id,
                userMessageContent: approval.command
                    ? `Approved and continued: ${approval.command}`
                    : "Approved and continued the pending task.",
                effectiveMessage: buildStudioApprovalContinuePrompt({
                    command: approval.command,
                    workerAgentId: approval.workerAgentId,
                    bridgeResult: approval.bridgeResult,
                }),
                outgoingMode: approvalMode,
                preparedProjectDir: projectDir ?? undefined,
                selectedUploadedFiles: [],
                existingHistory: [],
                permissionProfileOverride: "full_access",
                resumeSessionOverride: approval.cliSessionId,
            })
        } finally {
            setRunningApprovalActionId(null)
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

    const beginCommandTimelineRun = useCallback((inputContent: string, title?: string) => {
        setStatus("running")
        setLastError(null)
        onViewModeChange("log")

        const nextTitle = title ?? buildChatThreadTitle(inputContent)
        const taskId = activeChatTask?.id ?? addTask(nextTitle)

        if (activeChatTask && activeChatTask.history.length === 0 && activeChatTask.actions.length === 0) {
            renameTask(taskId, nextTitle)
        }

        addAction(taskId, {
            type: "user",
            content: inputContent,
        })
        updateTaskStatus(taskId, "running")
        setActiveTask(taskId)
        return taskId
    }, [
        activeChatTask,
        addAction,
        addTask,
        onViewModeChange,
        renameTask,
        setActiveTask,
        updateTaskStatus,
    ])

    const finishCommandTimelineRun = useCallback((
        taskId: string,
        result: {
            ok: boolean
            outputs?: Array<Omit<AgentAction, "id" | "timestamp">>
            error?: string | null
        },
    ) => {
        const error = result.error?.trim() ?? ""
        const outputs =
            result.outputs?.filter((output) => output.type === "complete" || output.content.trim().length > 0) ?? []

        if (outputs.length > 0) {
            outputs.forEach((output) => addAction(taskId, output))
        } else {
            if (error) {
                addAction(taskId, { type: "error", content: error })
            } else {
                addAction(taskId, { type: "complete", content: "Completed" })
            }
        }

        if (!result.ok || error) {
            updateTaskStatus(taskId, "error")
            setLastError(error || "Command failed")
            setStatus("error")
            return
        }

        updateTaskStatus(taskId, "completed")
        setStatus("success")
    }, [addAction, updateTaskStatus])

    const showSyntheticCommandOutput = (
        inputContent: string,
        title: string,
        content: string,
        metadata?: AgentAction["metadata"],
    ) => {
        const taskId = beginCommandTimelineRun(inputContent, title)
        finishCommandTimelineRun(taskId, {
            ok: true,
            outputs: [
                {
                    type: "text",
                    content,
                    metadata,
                },
            ],
        })
    }

    const clearActiveConversation = () => {
        setActiveTask(null)
        setMessageInput("")
        setUploadedFiles([])
        setChatDraftBeforeUtility("")
        setActiveCliCommand(null)
        setLastError(null)
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

        const taskId = beginCommandTimelineRun(preview, buildChatThreadTitle(preview))

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
            const outputs: Array<Omit<AgentAction, "id" | "timestamp">> = []

            if (payload?.stdout?.trim()) {
                outputs.push({
                    type: "text",
                    content: payload.stdout,
                    metadata: {
                        commandOutput: {
                            kind: "stdout",
                            title: "STDOUT",
                        },
                    },
                })
            }

            if (payload?.stderr?.trim()) {
                outputs.push({
                    type: payload?.ok ? "text" : "error",
                    content: payload.stderr,
                    metadata: {
                        commandOutput: {
                            kind: "stderr",
                            title: "STDERR",
                        },
                    },
                })
            }

            finishCommandTimelineRun(taskId, {
                ok: payload?.ok === true,
                outputs,
                error: payload?.ok ? null : payload?.stderr || `Command failed (${payload?.returncode ?? 500})`,
            })
        } catch (e) {
            const message = e instanceof Error ? e.message : "Failed to run Studio command"
            finishCommandTimelineRun(taskId, {
                ok: false,
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
                "/help",
                "Slash help",
                buildSlashHelpMarkdown(),
                {
                    commandOutput: {
                        kind: "help",
                        title: "Slash help",
                    },
                },
            )
            return true
        }

        if (parsed.kind === "status") {
            setMessageInput("")
            showSyntheticCommandOutput(
                "/status",
                "Slash status",
                buildStatusMarkdown(),
                {
                    commandOutput: {
                        kind: "status",
                        title: "Slash status",
                    },
                },
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
            applyModelSelection(parsed.modelOption, parsed.customModel)
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

    const replaceActiveSlashToken = useCallback((replacement = "") => {
        if (!activeSlashMatch) return replacement
        const before = messageInput.slice(0, activeSlashMatch.start)
        const after = messageInput.slice(activeSlashMatch.end)
        return mergeComposerText(before, replacement, after)
    }, [activeSlashMatch, messageInput])

    const syncComposerCursor = (target: HTMLTextAreaElement) => {
        setComposerCursor(target.selectionStart ?? target.value.length)
    }

    const focusComposerAt = useCallback((nextCursor: number) => {
        requestAnimationFrame(() => {
            if (!composerTextareaRef.current) return
            composerTextareaRef.current.focus()
            composerTextareaRef.current.setSelectionRange(nextCursor, nextCursor)
            setComposerCursor(nextCursor)
        })
    }, [])

    const focusComposerToEnd = useCallback(() => {
        const nextCursor = composerTextareaRef.current?.value.length ?? messageInput.length
        focusComposerAt(nextCursor)
    }, [focusComposerAt, messageInput.length])

    const handleComposerFileSelection = useCallback(
        async (event: ChangeEvent<HTMLInputElement>) => {
            const selectedFiles = Array.from(event.target.files ?? [])
            event.target.value = ""
            if (selectedFiles.length === 0) return

            try {
                const nextUploads = await Promise.all(selectedFiles.map((file) => fileToComposerUpload(file)))
                setUploadedFiles((current) => mergeUploadedFiles(current, nextUploads))
                setLastError(null)
                focusComposerToEnd()
            } catch (error) {
                setLastError(error instanceof Error ? error.message : "Failed to upload file")
            }
        },
        [focusComposerToEnd],
    )

    const handleOpenComposerUpload = useCallback(() => {
        if (commandMode) return
        composerFileInputRef.current?.click()
    }, [commandMode])

    const handleInsertSlashCommand = () => {
        const baseValue = commandMode ? chatDraftBeforeUtility : messageInput
        const cursor = commandMode
            ? (chatDraftBeforeUtility.length || 0)
            : (composerTextareaRef.current?.selectionStart ?? messageInput.length)
        const nextValue = `${baseValue.slice(0, cursor)}/${baseValue.slice(cursor)}`

        if (commandMode) {
            setActiveCliCommand(null)
            setChatDraftBeforeUtility("")
        }

        setMessageInput(nextValue)
        setLastError(null)
        focusComposerAt(cursor + 1)
    }

    function openAgentBoardWorkspace(delegationTaskId?: string) {
        const workerRunId = delegationTaskId
            ? workerRunIdByDelegationTaskId.get(delegationTaskId) ?? null
            : null
        if (workerRunId) {
            openWorkerRun(workerRunId)
        }
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
    const supportedSlashCommandSet = useMemo(() => {
        const values = runtimeInfo.supportedSlashCommands.map((item) => item.trim().toLowerCase()).filter(Boolean)
        return values.length > 0 ? new Set(values) : null
    }, [runtimeInfo.supportedSlashCommands])
    const supportedPermissionProfiles = useMemo(
        () =>
            runtimeInfo.supportedPermissionProfiles
                .map((item) => item.trim().toLowerCase())
                .filter((item) => item === "default" || item === "full_access"),
        [runtimeInfo.supportedPermissionProfiles],
    )
    const supportedRuntimeCommandSet = useMemo(() => {
        const values = runtimeInfo.runtimeCommands.map((item) => item.trim().toLowerCase()).filter(Boolean)
        return values.length > 0 ? new Set(values) : null
    }, [runtimeInfo.runtimeCommands])
    const supportedSlashCommands = useMemo(() => {
        const commands = runtimeInfo.supportedSlashCommands.filter((item) => item.trim().length > 0)
        return commands.length > 0
            ? commands
            : ["help", "status", "new", "clear", "plan", "model", "agents", "mcp", "auth", "doctor"]
    }, [runtimeInfo.supportedSlashCommands])
    const buildSlashHelpMarkdown = useCallback(() => {
        const commandLines = supportedSlashCommands.map((command) => `- \`/${command}\``)
        return [
            "## Supported Studio slash commands",
            "",
            ...commandLines,
            "",
            "Claude Code chat turns stream directly into this thread.",
            "Runtime utilities still run through the Claude CLI path and mirror into Monitor.",
        ].join("\n")
    }, [supportedSlashCommands])
    const buildStatusMarkdown = useCallback(() => {
        const claudeCliStatus = runtimeLoading
            ? "Checking"
            : runtimeInfo.source === "claude_code"
                ? "Available"
                : "Unavailable"
        const projectAgentStatus = runtimeInfo.source !== "claude_code"
            ? "Unknown"
            : runtimeInfo.claudeAgentsError
                ? "Probe failed"
                : runtimeInfo.projectAgentCount > 0
                    ? `${runtimeInfo.projectAgentCount} configured`
                    : "None detected"
        const codexBridgeStatus = runtimeInfo.source !== "claude_code"
            ? "Unknown"
            : runtimeInfo.claudeAgentsError
                ? "Unknown (agent probe failed)"
                : runtimeInfo.codexWorkerAvailable
                    ? `Ready (${runtimeInfo.codexWorkerName ?? "configured"})`
                    : runtimeInfo.projectAgentCount > 0
                        ? "Not configured"
                        : "No bridge agent found"
        const opencodeCliStatus = runtimeInfo.opencodeAvailable
            ? runtimeInfo.opencodeVersion
                ? `Available (${runtimeInfo.opencodeVersion})`
                : "Available"
            : "Unavailable"
        const opencodeBridgeStatus = runtimeInfo.source !== "claude_code"
            ? "Unknown"
            : runtimeInfo.claudeAgentsError
                ? "Unknown (agent probe failed)"
                : runtimeInfo.opencodeWorkerAvailable
                    ? `Ready (${runtimeInfo.opencodeWorkerName ?? "configured"})`
                    : runtimeInfo.projectAgentCount > 0
                        ? "Not configured"
                        : "No bridge agent found"
        const fields = [
            { label: "Runtime", value: runtimeLoading ? "Checking runtime" : runtimeLabel },
            { label: "Claude CLI", value: claudeCliStatus },
            { label: "Chat surface", value: formatStudioChatSurfaceLabel(runtimeInfo.chatSurface) },
            { label: "Transport", value: formatStudioChatTransportLabel(runtimeInfo.chatTransport) },
            { label: "Preferred route", value: formatStudioChatTransportLabel(runtimeInfo.preferredChatTransport) },
            { label: "CLI version", value: runtimeInfo.version || "Unavailable" },
            { label: "Agent SDK", value: runtimeInfo.claudeAgentSdkAvailable ? "Installed" : "Not installed" },
            { label: "Project agents", value: projectAgentStatus },
            { label: "Codex bridge", value: codexBridgeStatus },
            { label: "OpenCode CLI", value: opencodeCliStatus },
            { label: "OpenCode bridge", value: opencodeBridgeStatus },
            {
                label: "Code mode",
                value:
                    runtimeInfo.codeModeEnabled === true
                        ? "Enabled"
                        : runtimeInfo.codeModeEnabled === false
                            ? "Disabled"
                            : "Unknown",
            },
            { label: "Mode", value: mode },
            { label: "Permission", value: permissionProfile },
            { label: "Model", value: requestedModel || "Pending" },
            {
                label: "Detected default",
                value:
                    runtimeInfo.detectedDefaultModel
                        ? runtimeInfo.detectedDefaultModelSource
                            ? `${runtimeInfo.detectedDefaultModel} (${runtimeInfo.detectedDefaultModelSource})`
                            : runtimeInfo.detectedDefaultModel
                        : "Unavailable",
            },
            { label: "Workspace", value: projectDir ?? "Not set" },
            { label: "Uploaded files", value: String(uploadedFiles.length) },
            { label: "Paper", value: selectedPaper?.title ?? "None" },
            { label: "Session", value: activeChatTask?.name ?? "New thread" },
            ...(runtimeInfo.claudeAgentsError
                ? [{ label: "Agent probe", value: runtimeInfo.claudeAgentsError }]
                : []),
            ...(runtimeInfo.error ? [{ label: "Runtime error", value: runtimeInfo.error }] : []),
        ]

        return [
            "## Studio status",
            "",
            ...fields.map(({ label, value }) => `- **${label}:** ${value}`),
        ].join("\n")
    }, [
        activeChatTask?.name,
        mode,
        permissionProfile,
        projectDir,
        requestedModel,
        runtimeInfo.detectedDefaultModel,
        runtimeInfo.detectedDefaultModelSource,
        runtimeInfo.chatSurface,
        runtimeInfo.chatTransport,
        runtimeInfo.claudeAgentsError,
        runtimeInfo.claudeAgentSdkAvailable,
        runtimeInfo.codeModeEnabled,
        runtimeInfo.codexWorkerAvailable,
        runtimeInfo.codexWorkerName,
        runtimeInfo.error,
        runtimeInfo.opencodeAvailable,
        runtimeInfo.opencodeVersion,
        runtimeInfo.opencodeWorkerAvailable,
        runtimeInfo.opencodeWorkerName,
        runtimeInfo.preferredChatTransport,
        runtimeInfo.projectAgentCount,
        runtimeInfo.source,
        runtimeInfo.version,
        runtimeLabel,
        runtimeLoading,
        selectedPaper?.title,
        uploadedFiles.length,
    ])

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
            id: "slash-probe-codex",
            command: "probe-codex",
            label: "Insert Codex smoke prompt",
            description: "Insert a normal chat prompt that asks Claude Code to verify delegation through `codex-worker`.",
            group: "Claude Code",
            requiresRuntimeSupport: false,
            keywords: ["codex", "worker", "subagent", "delegate", "smoke", "test"],
            icon: Code,
            onSelect: () => {
                setMessageInput(replaceActiveSlashToken(buildCodexWorkerSmokePrompt()))
                setLastError(null)
            },
        },
        {
            id: "slash-probe-opencode",
            command: "probe-opencode",
            label: "Insert OpenCode smoke prompt",
            description: "Insert a normal chat prompt that asks Claude Code whether an OpenCode bridge agent is configured.",
            group: "Claude Code",
            requiresRuntimeSupport: false,
            keywords: ["opencode", "worker", "bridge", "delegate", "smoke", "test"],
            icon: Terminal,
            onSelect: () => {
                setMessageInput(replaceActiveSlashToken(buildOpenCodeWorkerSmokePrompt()))
                setLastError(null)
            },
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

    const availableSlashCommands = slashCommands.filter((item) => {
        if (item.requiresRuntimeSupport === false) {
            return true
        }
        if (item.group === "Runtime") {
            return supportedRuntimeCommandSet ? supportedRuntimeCommandSet.has(item.command) : true
        }
        return supportedSlashCommandSet ? supportedSlashCommandSet.has(item.command) : true
    })

    const filteredSlashCommands = availableSlashCommands.filter((item) => {
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

    useEffect(() => {
        if (!slashPaletteActive) return
        const list = slashPaletteListRef.current
        if (!list) return
        const activeItem = list.querySelector<HTMLElement>(`[data-slash-index="${slashSelectedIndex}"]`)
        activeItem?.scrollIntoView({ block: "nearest" })
    }, [slashPaletteActive, slashSelectedIndex])

    const handleApplySlashCommand = useCallback((command: SlashCommandItem) => {
        command.onSelect(replaceActiveSlashToken())
        setSlashSelectedIndex(0)
        focusComposerToEnd()
    }, [focusComposerToEnd, replaceActiveSlashToken])

    const composerHelperText = commandMode
        ? "Claude Code command selected. Enter runs it, and clear returns the composer to chat."
        : runtimeInfo.codeModeEnabled === false && mode === "Plan"
            ? "Code mode is disabled in this runtime. Studio will keep chat turns in Plan mode."
        : uploadedFiles.length > 0
            ? `${uploadedFiles.length} uploaded file${uploadedFiles.length === 1 ? "" : "s"} ready.`
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
        ...uploadedFiles.map((file) => ({
            id: `file:${file.id}`,
            label: file.name,
            meta: formatFileSize(file.size),
            tone: "neutral" as const,
            icon: Paperclip,
            onRemove: () =>
                setUploadedFiles((current) => current.filter((item) => item.id !== file.id)),
        })),
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
    const canAttachFiles = !commandMode
    const composerHeaderTitle = commandMode
        ? buildCommandPreview(activeCliCommand!.runtime, activeCliCommand!.preset, messageInput) || "Claude Code command"
        : slashPaletteActive
            ? "Insert slash command"
            : activeChatTask?.name || "New thread"
    const composerHeaderBadge = commandMode ? "Command" : slashPaletteActive ? "Slash" : "Chat"
    const composerHeaderRightLabel = slashPaletteActive ? `/${slashToken || ""}` : "/"

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
                        <div className="px-3 py-3">
                            {!visibleTask || visibleActions.length === 0 ? (
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
                                    {visibleActions.map((action) => (
                                        <MemoizedActionItem
                                            key={action.id}
                                            action={action}
                                            onViewDiff={setDiffAction}
                                            onOpenMonitor={onOpenBoardWorkspace}
                                            onApproveApprovalRequest={handleApproveApprovalRequest}
                                            approvalPending={runningApprovalActionId === action.id}
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
                                <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
                                    <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                                        {composerHeaderBadge}
                                    </span>
                                    <div className="min-w-0 flex-1 truncate text-[12px] font-medium text-slate-800">
                                        {composerHeaderTitle}
                                    </div>
                                </div>
                                <span className="shrink-0 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-500">
                                    {composerHeaderRightLabel}
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
                                        onChange={(event) =>
                                            applyModelSelection("custom", event.target.value)
                                        }
                                        placeholder="claude-sonnet-4-6"
                                        className="h-9 border-slate-200 bg-[#f7f8f4] text-xs text-slate-700"
                                        title="Full Claude Code model name"
                                    />
                                </div>
                            ) : null}

                            <div className="relative">
                                <input
                                    ref={composerFileInputRef}
                                    type="file"
                                    multiple
                                    className="hidden"
                                    onChange={(event) => {
                                        void handleComposerFileSelection(event)
                                    }}
                                />

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
                                    className={cn(
                                        "min-h-[88px] resize-none border-0 bg-transparent px-4 py-3 text-[14px] leading-7 text-slate-800 placeholder:text-slate-400 focus-visible:ring-0",
                                        slashPaletteActive ? "pb-40" : "pb-3",
                                    )}
                                    onKeyDown={(e) => {
                                        if (slashPaletteActive) {
                                            if (e.key === "ArrowDown" && filteredSlashCommands.length > 0) {
                                                e.preventDefault()
                                                setSlashSelectedIndex((current) =>
                                                    current >= filteredSlashCommands.length - 1 ? 0 : current + 1,
                                                )
                                                return
                                            }

                                            if (e.key === "ArrowUp" && filteredSlashCommands.length > 0) {
                                                e.preventDefault()
                                                setSlashSelectedIndex((current) =>
                                                    current <= 0 ? filteredSlashCommands.length - 1 : current - 1,
                                                )
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
                                    <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex px-4">
                                        <div className="pointer-events-auto max-w-[560px] flex-1 overflow-hidden rounded-2xl border border-slate-200 bg-[#f8faf5] shadow-[0_18px_40px_rgba(15,23,42,0.10)]">
                                            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3 border-b border-slate-200 bg-[#f0f2ec] px-3 py-2.5">
                                                <div className="min-w-0">
                                                    <div className="text-[11px] font-medium text-slate-800">Claude Code commands</div>
                                                    <div className="mt-0.5 truncate text-[10px] text-slate-500">
                                                        Slash opens Claude-style chat commands plus safe runtime utilities.
                                                    </div>
                                                </div>
                                                <span className="max-w-[10rem] shrink-0 truncate rounded-full border border-slate-200 bg-white px-2 py-0.5 font-mono text-[10px] text-slate-500">
                                                    /{slashToken || ""}
                                                </span>
                                            </div>

                                            <div
                                                ref={slashPaletteListRef}
                                                className="max-h-60 overflow-y-auto overscroll-contain"
                                            >
                                                {filteredSlashCommands.length === 0 ? (
                                                    <div className="px-3 py-4 text-sm text-slate-500">
                                                        No matching slash command.
                                                    </div>
                                                ) : (
                                                    <div className="space-y-1.5 p-2">
                                                        {(["Claude Code", "Session", "Runtime"] as const).map((group) => {
                                                            const groupItems = filteredSlashCommands.filter((item) => item.group === group)
                                                            if (groupItems.length === 0) return null

                                                            return (
                                                                <div key={group}>
                                                                    <div className="px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-400">
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
                                                                                    data-slash-index={globalIndex}
                                                                                    data-selected={selected ? "true" : "false"}
                                                                                    className={cn(
                                                                                        "flex w-full items-center gap-2.5 rounded-xl border px-2.5 py-2 text-left transition-colors",
                                                                                        selected
                                                                                            ? "border-slate-300 bg-[#edf0e7] shadow-[inset_0_0_0_1px_rgba(148,163,184,0.15)]"
                                                                                            : "border-transparent bg-transparent hover:border-slate-200 hover:bg-[#eef1ea]",
                                                                                    )}
                                                                                    onMouseEnter={() => setSlashSelectedIndex(globalIndex)}
                                                                                    onMouseDown={(event) => {
                                                                                        event.preventDefault()
                                                                                        handleApplySlashCommand(item)
                                                                                    }}
                                                                                >
                                                                                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white">
                                                                                        <ItemIcon className="h-3 w-3 text-slate-500" />
                                                                                    </div>
                                                                                    <div className="min-w-0 flex-1">
                                                                                        <div className="flex items-center gap-2">
                                                                                            <span className="font-mono text-[10px] text-slate-900">
                                                                                                /{item.command}
                                                                                            </span>
                                                                                            <span className="truncate text-[10px] text-slate-500">
                                                                                                {item.label}
                                                                                            </span>
                                                                                        </div>
                                                                                        <div className="mt-0.5 line-clamp-2 text-[10px] leading-4 text-slate-500">
                                                                                            {item.description}
                                                                                        </div>
                                                                                    </div>
                                                                                    {selected ? (
                                                                                        <span className="shrink-0 rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
                                                                                            enter
                                                                                        </span>
                                                                                    ) : null}
                                                                                </button>
                                                                            )
                                                                        })}
                                                                    </div>
                                                                </div>
                                                            )
                                                        })}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="flex items-center justify-between gap-3 border-t border-slate-200 bg-[#f2f4ef] px-3 py-2 text-[9px] uppercase tracking-[0.12em] text-slate-400">
                                                <span>↑↓ navigate</span>
                                                <span>Enter/Tab select</span>
                                                <span>Esc close</span>
                                            </div>
                                        </div>
                                    </div>
                                ) : null}
                            </div>
                        </div>

                        <div className="border-t border-slate-200 bg-[#f3f4ef] px-3 py-3">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex flex-1 flex-wrap items-center gap-1.5">
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <Button
                                                type="button"
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 rounded-full border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                                                onClick={handleInsertSlashCommand}
                                            >
                                                <span className="font-mono text-[13px] leading-none">/</span>
                                            </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>Insert slash command</TooltipContent>
                                    </Tooltip>

                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <Button
                                                type="button"
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 rounded-full border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                                                onClick={handleOpenComposerUpload}
                                                disabled={!canAttachFiles}
                                            >
                                                <Paperclip className="h-3.5 w-3.5" />
                                            </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                            {commandMode
                                                ? "Return to chat mode before uploading files"
                                                : "Upload files"}
                                        </TooltipContent>
                                    </Tooltip>

                                    <StudioPermissionSelector
                                        permissionProfile={permissionProfile}
                                        onPermissionChange={setPermissionProfile}
                                        supportedProfiles={supportedPermissionProfiles}
                                    />

                                    <Select value={mode} onValueChange={(value) => setMode(value as Mode)}>
                                        <SelectTrigger className="h-8 w-[104px] rounded-full border-slate-200 bg-white text-xs text-slate-700">
                                            <Code className="mr-1 h-3.5 w-3.5" />
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Code" disabled={runtimeInfo.codeModeEnabled === false}>
                                                {runtimeInfo.codeModeEnabled === false ? "Code (disabled)" : "Code"}
                                            </SelectItem>
                                            <SelectItem value="Plan">Plan</SelectItem>
                                            <SelectItem value="Ask">Ask</SelectItem>
                                        </SelectContent>
                                    </Select>

                                    <Select
                                        value={modelOption}
                                        onValueChange={(value) =>
                                            applyModelSelection(
                                                value,
                                                value === "custom" ? customModel : "",
                                            )
                                        }
                                    >
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
                                            : ((!messageInput.trim() && uploadedFiles.length === 0) || isBusy || missingCustomModel)
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
