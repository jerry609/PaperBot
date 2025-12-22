"use client"

import { useState, useRef, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { PanelsTopLeft, Settings, List, Play, Plug, Code2, ChevronLeft, ChevronRight, Activity, Box, Clock } from "lucide-react"
import { TasksPanel } from "@/components/studio/TasksPanel"
import { ExecutionLog } from "@/components/studio/ExecutionLog"
import { PromptInput } from "@/components/studio/PromptInput"
import { MCPSettings } from "@/components/studio/MCPSettings"
import { DiffViewer } from "@/components/studio/DiffViewer"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore } from "@/lib/store/studio-store"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"
import { useChat } from "@ai-sdk/react"
import { cn } from "@/lib/utils"

// Sandbox status component
function SandboxStatus() {
    const [status, setStatus] = useState<{
        e2b: { status: string };
        docker: { status: string };
        queue: { redis_connected: boolean; pending: number; running: number };
    } | null>(null)

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const res = await fetch('/api/sandbox/status')
                if (res.ok) {
                    setStatus(await res.json())
                }
            } catch {
                // Ignore
            }
        }
        fetchStatus()
        const interval = setInterval(fetchStatus, 10000)
        return () => clearInterval(interval)
    }, [])

    const getStatusColor = (s: string) => {
        if (s === 'healthy' || s === 'available') return 'bg-green-500'
        if (s === 'not_configured' || s === 'unavailable') return 'bg-yellow-500'
        return 'bg-red-500'
    }

    return (
        <ScrollArea className="h-full">
            <div className="p-3 space-y-4">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                    System Status
                </div>

                {!status ? (
                    <div className="space-y-2">
                        <div className="h-12 bg-muted/50 rounded animate-pulse" />
                        <div className="h-12 bg-muted/50 rounded animate-pulse" />
                        <div className="h-12 bg-muted/50 rounded animate-pulse" />
                    </div>
                ) : (
                    <div className="space-y-2">
                        <div className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                            <div className={cn("w-2 h-2 rounded-full", getStatusColor(status.e2b?.status || 'unknown'))} />
                            <Box className="h-4 w-4 text-muted-foreground" />
                            <div className="flex-1">
                                <div className="text-sm font-medium">E2B Sandbox</div>
                                <div className="text-xs text-muted-foreground capitalize">{status.e2b?.status || 'Unknown'}</div>
                            </div>
                        </div>

                        <div className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                            <div className={cn("w-2 h-2 rounded-full", getStatusColor(status.docker?.status || 'unknown'))} />
                            <Activity className="h-4 w-4 text-muted-foreground" />
                            <div className="flex-1">
                                <div className="text-sm font-medium">Docker</div>
                                <div className="text-xs text-muted-foreground capitalize">{status.docker?.status || 'Unknown'}</div>
                            </div>
                        </div>

                        <div className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                            <div className={cn("w-2 h-2 rounded-full", status.queue?.redis_connected ? 'bg-green-500' : 'bg-yellow-500')} />
                            <Clock className="h-4 w-4 text-muted-foreground" />
                            <div className="flex-1">
                                <div className="text-sm font-medium">Job Queue</div>
                                <div className="text-xs text-muted-foreground">
                                    {status.queue?.redis_connected ? `${status.queue.pending || 0} pending, ${status.queue.running || 0} running` : 'Disconnected'}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div className="pt-4 border-t">
                    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                        Quick Actions
                    </div>
                    <div className="space-y-2">
                        <Button variant="outline" size="sm" className="w-full justify-start text-xs">
                            <Play className="h-3.5 w-3.5 mr-2" />
                            Submit Paper2Code Job
                        </Button>
                        <Button variant="outline" size="sm" className="w-full justify-start text-xs">
                            <Activity className="h-3.5 w-3.5 mr-2" />
                            View Job Queue
                        </Button>
                    </div>
                </div>
            </div>
        </ScrollArea>
    )
}

function StudioContent() {
    const { addTask, addAction, updateTaskStatus, tasks, activeTaskId, selectedFileForDiff, setSelectedFileForDiff } = useStudioStore()
    const [leftCollapsed, setLeftCollapsed] = useState(false)
    const [rightCollapsed, setRightCollapsed] = useState(true)
    const currentTaskIdRef = useRef<string | null>(null)

    // Sample diff data (would come from actual file changes in real implementation)
    const [diffData, setDiffData] = useState<{ oldValue: string; newValue: string } | null>(null)

    const { messages, status, sendMessage, setMessages } = useChat({
        api: '/api/chat',
        onResponse: () => {
            // When we get a response, mark task as running
            if (currentTaskIdRef.current) {
                updateTaskStatus(currentTaskIdRef.current, 'running')
            }
        },
        onFinish: () => {
            // Mark task as completed
            if (currentTaskIdRef.current) {
                updateTaskStatus(currentTaskIdRef.current, 'completed')
                addAction(currentTaskIdRef.current, {
                    type: 'complete',
                    content: 'Task completed'
                })
            }
        },
        onError: (error) => {
            if (currentTaskIdRef.current) {
                updateTaskStatus(currentTaskIdRef.current, 'error')
                addAction(currentTaskIdRef.current, {
                    type: 'error',
                    content: error.message
                })
            }
        }
    })

    // Handle file selection for diff
    useEffect(() => {
        if (selectedFileForDiff) {
            setRightCollapsed(false)
            // In real implementation, fetch the actual diff
            setDiffData({
                oldValue: '// Original code\nfunction hello() {\n  console.log("Hello");\n}',
                newValue: '// Modified code\nfunction hello() {\n  console.log("Hello, World!");\n}\n\n// New function\nfunction goodbye() {\n  console.log("Goodbye!");\n}'
            })
        }
    }, [selectedFileForDiff])

    // Process messages and extract tool calls for display
    useEffect(() => {
        if (!currentTaskIdRef.current || messages.length === 0) return

        const lastMessage = messages[messages.length - 1]
        if (lastMessage.role === 'assistant' && lastMessage.parts) {
            for (const part of lastMessage.parts) {
                if (part.type === 'tool-invocation') {
                    const toolCall = part as { type: 'tool-invocation'; toolInvocation: { toolName: string; args: Record<string, unknown>; result?: unknown } }

                    // Add function call action
                    addAction(currentTaskIdRef.current, {
                        type: 'function_call',
                        content: `Called ${toolCall.toolInvocation.toolName}`,
                        metadata: {
                            functionName: toolCall.toolInvocation.toolName,
                            params: toolCall.toolInvocation.args,
                            result: toolCall.toolInvocation.result
                        }
                    })

                    // If it's a file operation, also add file_change action
                    if (['write_file', 'edit_file'].includes(toolCall.toolInvocation.toolName)) {
                        const result = toolCall.toolInvocation.result as { data?: { path?: string; linesAdded?: number; linesDeleted?: number; linesWritten?: number } }
                        if (result?.data) {
                            addAction(currentTaskIdRef.current, {
                                type: 'file_change',
                                content: `Modified file`,
                                metadata: {
                                    filename: result.data.path,
                                    linesAdded: result.data.linesAdded || result.data.linesWritten || 0,
                                    linesDeleted: result.data.linesDeleted || 0,
                                }
                            })
                        }
                    }
                } else if (part.type === 'text') {
                    const textPart = part as { type: 'text'; text: string }
                    // Add thinking/text action
                    if (textPart.text.trim()) {
                        addAction(currentTaskIdRef.current, {
                            type: 'thinking',
                            content: textPart.text
                        })
                    }
                }
            }
        }
    }, [messages, addAction])

    const handlePromptSubmit = async (prompt: string, model: string) => {
        // Create a new task
        const taskId = addTask(prompt.slice(0, 50) + (prompt.length > 50 ? '...' : ''))
        currentTaskIdRef.current = taskId

        // Add initial thinking action
        addAction(taskId, {
            type: 'thinking',
            content: `Processing: "${prompt}"`
        })

        // Clear previous messages and send new one
        setMessages([])

        // Send message with model selection
        sendMessage({
            content: prompt,
            data: { model }
        })
    }

    const isProcessing = status === 'streaming' || status === 'submitted'

    return (
        <div className="flex h-[calc(100vh-theme(spacing.16))] flex-col">
            {/* Top Bar */}
            <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-2.5 px-4 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-3">
                    <h2 className="text-base font-semibold flex items-center gap-2">
                        <PanelsTopLeft className="h-4 w-4 text-primary" /> DeepCode Studio
                    </h2>
                    <Badge variant="secondary" className="bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300 text-[10px]">
                        v3 Beta
                    </Badge>
                </div>
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => setLeftCollapsed(!leftCollapsed)}
                    >
                        {leftCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}
                        <span className="ml-1">Panel</span>
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => setRightCollapsed(!rightCollapsed)}
                    >
                        <Code2 className="h-3.5 w-3.5" />
                        <span className="ml-1">Preview</span>
                        {!rightCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}
                    </Button>
                    <div className="w-px h-4 bg-border mx-1" />
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                        <Settings className="h-3.5 w-3.5" />
                    </Button>
                </div>
            </div>

            {/* Main Workspace with Resizable Panels */}
            <ResizablePanelGroup direction="horizontal" className="flex-1">
                {/* Left Sidebar */}
                {!leftCollapsed && (
                    <>
                        <ResizablePanel defaultSize={18} minSize={15} maxSize={30}>
                            <div className="h-full bg-muted/5 flex flex-col">
                                <Tabs defaultValue="tasks" className="h-full flex flex-col">
                                    <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0 h-9">
                                        <TabsTrigger
                                            value="tasks"
                                            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-3"
                                        >
                                            <List className="h-3 w-3 mr-1.5" />
                                            Tasks
                                        </TabsTrigger>
                                        <TabsTrigger
                                            value="mcp"
                                            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-3"
                                        >
                                            <Plug className="h-3 w-3 mr-1.5" />
                                            MCP
                                        </TabsTrigger>
                                        <TabsTrigger
                                            value="sandbox"
                                            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-3"
                                        >
                                            <Play className="h-3 w-3 mr-1.5" />
                                            Sandbox
                                        </TabsTrigger>
                                    </TabsList>
                                    <TabsContent value="tasks" className="flex-1 m-0 overflow-hidden">
                                        <TasksPanel />
                                    </TabsContent>
                                    <TabsContent value="mcp" className="flex-1 m-0 overflow-hidden">
                                        <MCPSettings />
                                    </TabsContent>
                                    <TabsContent value="sandbox" className="flex-1 m-0 overflow-hidden">
                                        <SandboxStatus />
                                    </TabsContent>
                                </Tabs>
                            </div>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                    </>
                )}

                {/* Center Panel: Execution Log + Prompt */}
                <ResizablePanel defaultSize={rightCollapsed ? 82 : 52} minSize={35}>
                    <div className="h-full flex flex-col bg-background">
                        <div className="flex-1 overflow-hidden">
                            <ExecutionLog />
                        </div>
                        <PromptInput
                            onSubmit={handlePromptSubmit}
                            isLoading={isProcessing}
                        />
                    </div>
                </ResizablePanel>

                {/* Right Panel: Code Preview / Diff */}
                {!rightCollapsed && (
                    <>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={30} minSize={20} maxSize={50}>
                            <div className="h-full bg-muted/5 flex flex-col">
                                {selectedFileForDiff && diffData ? (
                                    <DiffViewer
                                        oldValue={diffData.oldValue}
                                        newValue={diffData.newValue}
                                        filename={selectedFileForDiff}
                                        onClose={() => {
                                            setSelectedFileForDiff(null)
                                            setDiffData(null)
                                        }}
                                        onApply={() => {
                                            // Apply changes logic
                                            setSelectedFileForDiff(null)
                                            setDiffData(null)
                                        }}
                                        onReject={() => {
                                            setSelectedFileForDiff(null)
                                            setDiffData(null)
                                        }}
                                    />
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-muted-foreground text-sm p-6">
                                        <Code2 className="h-12 w-12 mb-4 opacity-20" />
                                        <p className="font-medium">Code Preview</p>
                                        <p className="text-xs mt-1 text-center">
                                            Click on a file change in the execution log to view the diff
                                        </p>
                                    </div>
                                )}
                            </div>
                        </ResizablePanel>
                    </>
                )}
            </ResizablePanelGroup>
        </div>
    )
}

// Wrap with MCPProvider
export default function DeepCodeStudioPage() {
    return (
        <MCPProvider>
            <StudioContent />
        </MCPProvider>
    )
}
