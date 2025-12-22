"use client"

import { useStudioStore, AgentAction } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Bot, FileCode, Wrench, Plug, AlertCircle, CheckCircle2, Search, Terminal } from "lucide-react"
import { cn } from "@/lib/utils"

const actionIcons: Record<string, React.ElementType> = {
    thinking: Bot,
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

const actionColors: Record<string, string> = {
    thinking: "text-purple-500",
    file_change: "text-blue-500",
    function_call: "text-orange-500",
    mcp_call: "text-green-500",
    error: "text-red-500",
    complete: "text-green-500",
    text: "text-foreground",
    search_codebase: "text-yellow-500",
    edit_file: "text-blue-500",
    run_command: "text-cyan-500",
}

function ActionItem({ action }: { action: AgentAction }) {
    const iconKey = action.metadata?.functionName || action.type
    const Icon = actionIcons[iconKey] || actionIcons[action.type] || Bot
    const color = actionColors[iconKey] || actionColors[action.type] || "text-foreground"
    const { setSelectedFileForDiff } = useStudioStore()

    return (
        <div className="flex gap-3 py-2 border-b border-border/50 last:border-0">
            <div className={cn("w-6 h-6 flex items-center justify-center shrink-0 mt-0.5 rounded-full bg-muted", color)}>
                <Icon className="h-3.5 w-3.5" />
            </div>
            <div className="flex-1 min-w-0">
                {action.type === 'file_change' && action.metadata?.filename ? (
                    <div>
                        <button
                            onClick={() => setSelectedFileForDiff(action.metadata?.filename || null)}
                            className="text-blue-500 hover:underline font-medium text-sm"
                        >
                            📁 {action.metadata.filename}
                        </button>
                        <span className="text-xs text-muted-foreground ml-2">
                            <span className="text-green-500">+{action.metadata.linesAdded || 0}</span>
                            {" / "}
                            <span className="text-red-500">-{action.metadata.linesDeleted || 0}</span>
                        </span>
                        {action.content && (
                            <p className="text-sm text-muted-foreground mt-1">{action.content}</p>
                        )}
                    </div>
                ) : action.type === 'function_call' && action.metadata?.functionName ? (
                    <div>
                        <div className="flex items-center gap-2">
                            <code className="text-sm font-mono bg-muted px-1.5 py-0.5 rounded">
                                {action.metadata.functionName}()
                            </code>
                            <span className="text-xs text-muted-foreground">called</span>
                        </div>
                        {action.metadata.params && (
                            <pre className="text-xs text-muted-foreground mt-1 font-mono bg-muted/50 p-2 rounded overflow-x-auto">
                                {JSON.stringify(action.metadata.params, null, 2)}
                            </pre>
                        )}
                        {action.metadata.result && (
                            <div className="mt-2">
                                <span className="text-xs text-green-600 font-medium">Result:</span>
                                <pre className="text-xs text-muted-foreground mt-1 font-mono bg-muted/50 p-2 rounded overflow-x-auto">
                                    {typeof action.metadata.result === 'string'
                                        ? action.metadata.result
                                        : JSON.stringify(action.metadata.result, null, 2)}
                                </pre>
                            </div>
                        )}
                    </div>
                ) : action.type === 'mcp_call' && action.metadata?.mcpTool ? (
                    <div>
                        <span className="text-xs text-muted-foreground">{action.metadata.mcpServer}: </span>
                        <code className="text-sm font-mono bg-muted px-1.5 py-0.5 rounded">
                            {action.metadata.mcpTool}
                        </code>
                        {action.metadata.mcpResult && (
                            <pre className="text-xs text-muted-foreground mt-1 font-mono bg-muted/50 p-2 rounded overflow-x-auto">
                                {typeof action.metadata.mcpResult === 'string'
                                    ? action.metadata.mcpResult
                                    : JSON.stringify(action.metadata.mcpResult, null, 2)}
                            </pre>
                        )}
                    </div>
                ) : (
                    <p className="text-sm whitespace-pre-wrap">{action.content}</p>
                )}
            </div>
        </div>
    )
}

export function ExecutionLog() {
    const { tasks, activeTaskId } = useStudioStore()
    const activeTask = tasks.find(t => t.id === activeTaskId)

    return (
        <div className="h-full flex flex-col">
            <div className="p-3 border-b bg-background flex items-center justify-between">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    {activeTask ? activeTask.name : 'Execution Log'}
                </h3>
                {activeTask && (
                    <span className={cn(
                        "text-xs px-2 py-0.5 rounded-full",
                        activeTask.status === 'running' && "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
                        activeTask.status === 'completed' && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
                        activeTask.status === 'error' && "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
                    )}>
                        {activeTask.status}
                    </span>
                )}
            </div>

            <ScrollArea className="flex-1">
                <div className="p-4 space-y-1">
                    {!activeTask || activeTask.actions.length === 0 ? (
                        <div className="text-center text-muted-foreground text-sm py-12 space-y-2">
                            <Bot className="h-10 w-10 mx-auto opacity-30" />
                            <p>No actions yet</p>
                            <p className="text-xs">Start by entering a prompt below</p>
                        </div>
                    ) : (
                        activeTask.actions.map(action => (
                            <ActionItem key={action.id} action={action} />
                        ))
                    )}
                </div>
            </ScrollArea>
        </div>
    )
}
