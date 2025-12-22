"use client"

import { useState, useRef, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { PanelsTopLeft, Settings, List, Play, Plug, Sparkles, Activity, Box, Clock, LayoutPanelLeft, LayoutPanelTop, PanelRight, PanelLeftClose, PanelLeftOpen } from "lucide-react"
import { TasksPanel } from "@/components/studio/TasksPanel"
import { ExecutionLog } from "@/components/studio/ExecutionLog"
import { PromptInput } from "@/components/studio/PromptInput"
import { MCPSettings } from "@/components/studio/MCPSettings"
import { DeepCodeEditor } from "@/components/studio/DeepCodeEditor"
import { RunbookPanel } from "@/components/studio/RunbookPanel"
import { BlueprintPanel } from "@/components/studio/BlueprintPanel"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore } from "@/lib/store/studio-store"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"
import { useChat } from "@ai-sdk/react"
import { DefaultChatTransport } from "ai"
import { cn } from "@/lib/utils"
import { usePanelRef } from "react-resizable-panels"

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
    const { addTask, addAction, updateTaskStatus } = useStudioStore()
    const currentTaskIdRef = useRef<string | null>(null)
    const workspacePanelRef = usePanelRef()
    const opsPanelRef = usePanelRef()
    const runbookPanelRef = usePanelRef()
    const timelinePanelRef = usePanelRef()
    const blueprintPanelRef = usePanelRef()

    const [collapsed, setCollapsed] = useState({
        workspace: false,
        ops: false,
        runbook: false,
        timeline: false,
        blueprint: false,
    })

    const { messages, status, sendMessage, setMessages } = useChat({
        transport: new DefaultChatTransport({ api: '/api/chat' }),
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

    // Update task status when chat status changes
    useEffect(() => {
        if (currentTaskIdRef.current && status === 'streaming') {
            updateTaskStatus(currentTaskIdRef.current, 'running')
        }
    }, [status, updateTaskStatus])

    // Process messages and extract tool calls for display
    useEffect(() => {
        if (!currentTaskIdRef.current || messages.length === 0) return

        const lastMessage = messages[messages.length - 1]
        if (lastMessage.role === 'assistant' && lastMessage.parts) {
            for (const part of lastMessage.parts) {
                // Check if this is a tool part (type starts with 'tool-')
                if (part.type.startsWith('tool-')) {
                    const toolPart = part as {
                        type: string;
                        toolCallId: string;
                        state: string;
                        input?: unknown;
                        output?: unknown;
                        errorText?: string;
                    }
                    const toolName = part.type.replace('tool-', '')

                    // Add function call action
                    addAction(currentTaskIdRef.current, {
                        type: 'function_call',
                        content: `Called ${toolName}`,
                        metadata: {
                            functionName: toolName,
                            params: toolPart.input as Record<string, unknown>,
                            result: toolPart.output
                        }
                    })

                    // If it's a file operation, also add file_change action
                    if (['write_file', 'edit_file'].includes(toolName)) {
                        const result = toolPart.output as { data?: { path?: string; linesAdded?: number; linesDeleted?: number; linesWritten?: number } } | undefined
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
            text: prompt,
        }, {
            body: { model }
        })
    }

    const isProcessing = status === 'streaming' || status === 'submitted'
    const setCollapsedKey = (key: keyof typeof collapsed, value: boolean) =>
        setCollapsed((prev) => (prev[key] === value ? prev : { ...prev, [key]: value }))

    const togglePanel = (key: keyof typeof collapsed, ref: ReturnType<typeof usePanelRef>) => {
        try {
            if (collapsed[key]) ref.current?.expand()
            else ref.current?.collapse()
        } catch {
            // Ignore
        }
    }

    return (
        <div className="flex h-[calc(100vh_-_theme(spacing.16))] min-h-0 flex-col">
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
                    <div className="hidden xl:flex items-center gap-1 mr-1">
                        <Button
                            variant={collapsed.workspace ? "outline" : "secondary"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => togglePanel("workspace", workspacePanelRef)}
                            title="Toggle Workspace"
                        >
                            <LayoutPanelLeft className="h-3.5 w-3.5 mr-1.5" />
                            Workspace
                        </Button>
                        <Button
                            variant={collapsed.runbook ? "outline" : "secondary"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => togglePanel("runbook", runbookPanelRef)}
                            title="Toggle Runbook"
                        >
                            <LayoutPanelTop className="h-3.5 w-3.5 mr-1.5" />
                            Runbook
                        </Button>
                        <Button
                            variant={collapsed.timeline ? "outline" : "secondary"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => togglePanel("timeline", timelinePanelRef)}
                            title="Toggle Timeline"
                        >
                            <LayoutPanelTop className="h-3.5 w-3.5 mr-1.5" />
                            Timeline
                        </Button>
                        <Button
                            variant={collapsed.blueprint ? "outline" : "secondary"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => togglePanel("blueprint", blueprintPanelRef)}
                            title="Toggle Blueprint"
                        >
                            <PanelRight className="h-3.5 w-3.5 mr-1.5" />
                            Blueprint
                        </Button>
                        <Button
                            variant={collapsed.ops ? "outline" : "ghost"}
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => togglePanel("ops", opsPanelRef)}
                            title="Toggle Ops Column"
                        >
                            {collapsed.ops ? <PanelLeftOpen className="h-3.5 w-3.5" /> : <PanelLeftClose className="h-3.5 w-3.5" />}
                        </Button>
                    </div>
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                        <Settings className="h-3.5 w-3.5" />
                    </Button>
                </div>
            </div>

            {/* Desktop: resizable 3 columns (Workspace / Ops / Blueprint) */}
            <div className="hidden xl:flex flex-1 min-h-0">
                <ResizablePanelGroup orientation="horizontal" className="flex-1 min-h-0">
                    <ResizablePanel
                        id="workspace"
                        panelRef={workspacePanelRef}
                        collapsible
                        collapsedSize={0}
                        defaultSize="48"
                        minSize="30"
                        onResize={({ inPixels }) => setCollapsedKey("workspace", inPixels < 2)}
                    >
                        <div className="h-full min-w-0 min-h-0 bg-background">
                            <DeepCodeEditor />
                        </div>
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    <ResizablePanel
                        id="ops"
                        panelRef={opsPanelRef}
                        collapsible
                        collapsedSize={0}
                        defaultSize="32"
                        minSize="26"
                        onResize={({ inPixels }) => setCollapsedKey("ops", inPixels < 2)}
                    >
                        <ResizablePanelGroup orientation="vertical" className="h-full min-h-0">
                            <ResizablePanel
                                id="runbook"
                                panelRef={runbookPanelRef}
                                collapsible
                                collapsedSize={0}
                                defaultSize="40"
                                minSize="20"
                                onResize={({ inPixels }) => setCollapsedKey("runbook", inPixels < 2)}
                            >
                                <RunbookPanel />
                            </ResizablePanel>
                            <ResizableHandle withHandle />
                            <ResizablePanel
                                id="timeline"
                                panelRef={timelinePanelRef}
                                collapsible
                                collapsedSize={0}
                                defaultSize="60"
                                minSize="25"
                                onResize={({ inPixels }) => setCollapsedKey("timeline", inPixels < 2)}
                            >
                                <div className="h-full flex flex-col bg-background min-w-0 min-h-0">
                                    <div className="flex-1 min-h-0 overflow-hidden">
                                        <Tabs defaultValue="timeline" className="h-full flex flex-col min-h-0">
                                            <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0 h-9 shrink-0">
                                                <TabsTrigger
                                                    value="timeline"
                                                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-2.5"
                                                >
                                                    <Sparkles className="h-3 w-3 mr-1.5" />
                                                    Timeline
                                                </TabsTrigger>
                                                <TabsTrigger
                                                    value="tasks"
                                                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-2.5"
                                                >
                                                    <List className="h-3 w-3 mr-1.5" />
                                                    Tasks
                                                </TabsTrigger>
                                                <TabsTrigger
                                                    value="mcp"
                                                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-2.5"
                                                >
                                                    <Plug className="h-3 w-3 mr-1.5" />
                                                    MCP
                                                </TabsTrigger>
                                                <TabsTrigger
                                                    value="sandbox"
                                                    className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none text-xs h-9 px-2.5"
                                                >
                                                    <Play className="h-3 w-3 mr-1.5" />
                                                    Sandbox
                                                </TabsTrigger>
                                            </TabsList>
                                            <TabsContent value="timeline" className="flex-1 min-h-0 m-0 overflow-hidden">
                                                <div className="h-full flex flex-col min-h-0">
                                                    <div className="flex-1 min-h-0 overflow-hidden">
                                                        <ExecutionLog />
                                                    </div>
                                                    <PromptInput
                                                        onSubmit={handlePromptSubmit}
                                                        isLoading={isProcessing}
                                                    />
                                                </div>
                                            </TabsContent>
                                            <TabsContent value="tasks" className="flex-1 min-h-0 m-0 overflow-hidden">
                                                <TasksPanel />
                                            </TabsContent>
                                            <TabsContent value="mcp" className="flex-1 min-h-0 m-0 overflow-hidden">
                                                <MCPSettings />
                                            </TabsContent>
                                            <TabsContent value="sandbox" className="flex-1 min-h-0 m-0 overflow-hidden">
                                                <SandboxStatus />
                                            </TabsContent>
                                        </Tabs>
                                    </div>
                                </div>
                            </ResizablePanel>
                        </ResizablePanelGroup>
                    </ResizablePanel>

                    <ResizableHandle withHandle />
                    <ResizablePanel
                        id="blueprint"
                        panelRef={blueprintPanelRef}
                        collapsible
                        collapsedSize={0}
                        defaultSize="20"
                        minSize="16"
                        maxSize="40"
                        onResize={({ inPixels }) => setCollapsedKey("blueprint", inPixels < 2)}
                    >
                        <BlueprintPanel />
                    </ResizablePanel>
                </ResizablePanelGroup>
            </div>

            {/* Small screens: switch via tabs to avoid visual crowding */}
            <div className="flex xl:hidden flex-1 min-h-0">
                <Tabs defaultValue="workspace" className="h-full w-full flex flex-col min-h-0">
                    <TabsList className="w-full justify-start rounded-none border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-0 h-10 shrink-0">
                        <TabsTrigger value="workspace" className="rounded-none text-xs h-10 px-3 data-[state=active]:bg-transparent">
                            Workspace
                        </TabsTrigger>
                        <TabsTrigger value="runbook" className="rounded-none text-xs h-10 px-3 data-[state=active]:bg-transparent">
                            Runbook
                        </TabsTrigger>
                        <TabsTrigger value="timeline" className="rounded-none text-xs h-10 px-3 data-[state=active]:bg-transparent">
                            Timeline
                        </TabsTrigger>
                        <TabsTrigger value="blueprint" className="rounded-none text-xs h-10 px-3 data-[state=active]:bg-transparent">
                            Blueprint
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="workspace" className="flex-1 min-h-0 m-0 overflow-hidden">
                        <DeepCodeEditor />
                    </TabsContent>
                    <TabsContent value="runbook" className="flex-1 min-h-0 m-0 overflow-hidden">
                        <RunbookPanel />
                    </TabsContent>
                    <TabsContent value="timeline" className="flex-1 min-h-0 m-0 overflow-hidden">
                        <div className="h-full flex flex-col min-h-0">
                            <div className="flex-1 min-h-0 overflow-hidden">
                                <ExecutionLog />
                            </div>
                            <PromptInput
                                onSubmit={handlePromptSubmit}
                                isLoading={isProcessing}
                            />
                        </div>
                    </TabsContent>
                    <TabsContent value="blueprint" className="flex-1 min-h-0 m-0 overflow-hidden">
                        <BlueprintPanel />
                    </TabsContent>
                </Tabs>
            </div>
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
