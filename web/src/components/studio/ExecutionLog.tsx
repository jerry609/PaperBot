"use client"

import { useState } from "react"
import { useStudioStore, AgentAction } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Bot, FileCode, Wrench, Plug, AlertCircle, CheckCircle2, Search, Terminal, ChevronDown, ChevronRight, Clock, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"
import { DiffModal } from "./DiffViewer"

const actionIcons: Record<string, React.ElementType> = {
    thinking: Sparkles,
    file_change: FileCode,
    function_call: Wrench,
    mcp_call: Plug,
    error: AlertCircle,
    complete: CheckCircle2,
    text: Bot,
    search_codebase: Search,
    edit_file: FileCode,
    run_command: Terminal,
}

const actionColors: Record<string, { bg: string; text: string; border: string }> = {
    thinking: { bg: "bg-purple-50 dark:bg-purple-950/30", text: "text-purple-600 dark:text-purple-400", border: "border-purple-200 dark:border-purple-800" },
    file_change: { bg: "bg-blue-50 dark:bg-blue-950/30", text: "text-blue-600 dark:text-blue-400", border: "border-blue-200 dark:border-blue-800" },
    function_call: { bg: "bg-orange-50 dark:bg-orange-950/30", text: "text-orange-600 dark:text-orange-400", border: "border-orange-200 dark:border-orange-800" },
    mcp_call: { bg: "bg-emerald-50 dark:bg-emerald-950/30", text: "text-emerald-600 dark:text-emerald-400", border: "border-emerald-200 dark:border-emerald-800" },
    error: { bg: "bg-red-50 dark:bg-red-950/30", text: "text-red-600 dark:text-red-400", border: "border-red-200 dark:border-red-800" },
    complete: { bg: "bg-green-50 dark:bg-green-950/30", text: "text-green-600 dark:text-green-400", border: "border-green-200 dark:border-green-800" },
    text: { bg: "bg-muted/50", text: "text-foreground", border: "border-border" },
    search_codebase: { bg: "bg-yellow-50 dark:bg-yellow-950/30", text: "text-yellow-600 dark:text-yellow-400", border: "border-yellow-200 dark:border-yellow-800" },
    edit_file: { bg: "bg-blue-50 dark:bg-blue-950/30", text: "text-blue-600 dark:text-blue-400", border: "border-blue-200 dark:border-blue-800" },
    run_command: { bg: "bg-cyan-50 dark:bg-cyan-950/30", text: "text-cyan-600 dark:text-cyan-400", border: "border-cyan-200 dark:border-cyan-800" },
}

function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

interface ActionItemProps {
    action: AgentAction;
    onViewDiff: (action: AgentAction) => void;
    isLast: boolean;
}

function ActionItem({ action, onViewDiff, isLast }: ActionItemProps) {
    const [expanded, setExpanded] = useState(false)
    const iconKey = action.metadata?.functionName || action.type
    const Icon = actionIcons[iconKey] || actionIcons[action.type] || Bot
    const colors = actionColors[iconKey] || actionColors[action.type] || actionColors.text

    const hasExpandableContent = Boolean(action.metadata?.params || action.metadata?.result || action.metadata?.mcpResult)

    return (
        <div className="relative flex gap-3">
            {/* Timeline line */}
            {!isLast && (
                <div className="absolute left-3 top-8 bottom-0 w-px bg-border" />
            )}

            {/* Icon */}
            <div className={cn(
                "relative z-10 w-6 h-6 flex items-center justify-center shrink-0 rounded-md border",
                colors.bg, colors.border
            )}>
                <Icon className={cn("h-3 w-3", colors.text)} />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0 pb-4">
                <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                        {action.type === 'file_change' && action.metadata?.filename ? (
                            <div className="space-y-1">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <button
                                        onClick={() => onViewDiff(action)}
                                        className={cn("font-mono text-sm hover:underline", colors.text)}
                                    >
                                        {action.metadata.filename}
                                    </button>
                                    <span className="text-xs px-1.5 py-0.5 rounded bg-muted">
                                        <span className="text-green-600 dark:text-green-400">+{action.metadata.linesAdded || 0}</span>
                                        <span className="text-muted-foreground mx-1">/</span>
                                        <span className="text-red-600 dark:text-red-400">-{action.metadata.linesDeleted || 0}</span>
                                    </span>
                                </div>
                                {action.content && (
                                    <p className="text-xs text-muted-foreground">{action.content}</p>
                                )}
                            </div>
                        ) : action.type === 'function_call' && action.metadata?.functionName ? (
                            <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                    <code className={cn("text-xs font-mono px-1.5 py-0.5 rounded", colors.bg, colors.text)}>
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
                                    <div className="mt-2 space-y-2">
                                        {Boolean(action.metadata.params) && (
                                            <div className="rounded-md border bg-muted/30 p-2">
                                                <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Args</div>
                                                <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap break-all text-muted-foreground">
                                                    {JSON.stringify(action.metadata.params, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                        {Boolean(action.metadata.result) && (
                                            <div className="rounded-md border border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-950/20 p-2">
                                                <div className="text-[10px] font-medium text-green-600 dark:text-green-400 uppercase tracking-wider mb-1">Result</div>
                                                <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap break-all text-muted-foreground">
                                                    {typeof action.metadata.result === 'string'
                                                        ? action.metadata.result
                                                        : JSON.stringify(action.metadata.result, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : action.type === 'mcp_call' && action.metadata?.mcpTool ? (
                            <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                    <span className="text-[10px] text-muted-foreground uppercase">{action.metadata.mcpServer}</span>
                                    <code className={cn("text-xs font-mono px-1.5 py-0.5 rounded", colors.bg, colors.text)}>
                                        {action.metadata.mcpTool}
                                    </code>
                                    {Boolean(action.metadata.mcpResult) && (
                                        <button
                                            onClick={() => setExpanded(!expanded)}
                                            className="text-muted-foreground hover:text-foreground transition-colors"
                                        >
                                            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                        </button>
                                    )}
                                </div>
                                {expanded && Boolean(action.metadata.mcpResult) && (
                                    <div className="mt-2 rounded-md border bg-muted/30 p-2">
                                        <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap break-all text-muted-foreground">
                                            {typeof action.metadata.mcpResult === 'string'
                                                ? action.metadata.mcpResult
                                                : JSON.stringify(action.metadata.mcpResult, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        ) : action.type === 'error' ? (
                            <div className={cn("text-sm rounded-md border p-2", colors.bg, colors.border)}>
                                <span className={colors.text}>{action.content}</span>
                            </div>
                        ) : action.type === 'complete' ? (
                            <div className="flex items-center gap-2">
                                <span className={cn("text-sm font-medium", colors.text)}>Task completed</span>
                            </div>
                        ) : (
                            <p className="text-sm text-foreground/90 whitespace-pre-wrap leading-relaxed">{action.content}</p>
                        )}
                    </div>

                    {/* Timestamp */}
                    <div className="flex items-center gap-1 text-[10px] text-muted-foreground shrink-0">
                        <Clock className="h-2.5 w-2.5" />
                        {formatTime(action.timestamp)}
                    </div>
                </div>
            </div>
        </div>
    )
}

export function ExecutionLog() {
    const { tasks, activeTaskId } = useStudioStore()
    const activeTask = tasks.find(t => t.id === activeTaskId)
    const [diffAction, setDiffAction] = useState<AgentAction | null>(null)

    const handleViewDiff = (action: AgentAction) => {
        setDiffAction(action)
    }

    const handleCloseDiff = () => {
        setDiffAction(null)
    }

    const statusStyles = {
        running: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 animate-pulse",
        completed: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
        pending: "bg-gray-100 text-gray-700 dark:bg-gray-900/40 dark:text-gray-300",
        error: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    }

    return (
        <div className="h-full flex flex-col">
            {/* Header */}
            <div className="p-3 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center justify-between">
                <div className="flex items-center gap-2 min-w-0">
                    <Terminal className="h-4 w-4 text-muted-foreground shrink-0" />
                    <h3 className="text-sm font-medium truncate">
                        {activeTask ? activeTask.name : 'Execution Log'}
                    </h3>
                </div>
                {activeTask && (
                    <span className={cn(
                        "text-[10px] font-medium px-2 py-0.5 rounded-full uppercase tracking-wider",
                        statusStyles[activeTask.status]
                    )}>
                        {activeTask.status}
                    </span>
                )}
            </div>

            {/* Content */}
            <ScrollArea className="flex-1">
                <div className="p-4">
                    {!activeTask || activeTask.actions.length === 0 ? (
                        <div className="flex flex-col items-center justify-center text-muted-foreground text-sm py-16 space-y-3">
                            <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center">
                                <Sparkles className="h-8 w-8 opacity-30" />
                            </div>
                            <div className="text-center space-y-1">
                                <p className="font-medium">Ready to assist</p>
                                <p className="text-xs max-w-[200px]">Enter a prompt below to start a new task</p>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-0">
                            {activeTask.actions.map((action, index) => (
                                <ActionItem
                                    key={action.id}
                                    action={action}
                                    onViewDiff={handleViewDiff}
                                    isLast={index === activeTask.actions.length - 1}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </ScrollArea>

            {/* Diff Modal */}
            <DiffModal
                isOpen={!!diffAction}
                oldValue={diffAction?.metadata?.oldContent || '// Original file content'}
                newValue={diffAction?.metadata?.newContent || diffAction?.metadata?.diff || '// Modified file content'}
                filename={diffAction?.metadata?.filename}
                onClose={handleCloseDiff}
                onApply={() => {
                    handleCloseDiff()
                }}
                onReject={handleCloseDiff}
            />
        </div>
    )
}
