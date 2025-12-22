"use client"

import { useState, useRef, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { PanelsTopLeft, Settings, List, Play, Plug } from "lucide-react"
import { TasksPanel } from "@/components/studio/TasksPanel"
import { ExecutionLog } from "@/components/studio/ExecutionLog"
import { PromptInput } from "@/components/studio/PromptInput"
import { MCPSettings } from "@/components/studio/MCPSettings"
import { useStudioStore } from "@/lib/store/studio-store"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useChat } from "@ai-sdk/react"

export default function DeepCodeStudioPage() {
    const { addTask, addAction, updateTaskStatus, tasks, activeTaskId } = useStudioStore()
    const [leftPanelWidth, setLeftPanelWidth] = useState(240)
    const currentTaskIdRef = useRef<string | null>(null)

    const { messages, status, sendMessage, setMessages } = useChat({
        api: '/api/chat',
        onResponse: (response) => {
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
            <div className="border-b bg-background p-3 px-4 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-4">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                        <PanelsTopLeft className="h-5 w-5" /> DeepCode Studio
                    </h2>
                    <Badge variant="secondary" className="bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
                        v3 Beta
                    </Badge>
                </div>
                <Button variant="ghost" size="icon">
                    <Settings className="h-4 w-4" />
                </Button>
            </div>

            {/* Main Workspace */}
            <div className="flex-1 overflow-hidden flex">
                {/* Left Sidebar */}
                <div
                    className="border-r bg-muted/10 shrink-0 flex flex-col"
                    style={{ width: leftPanelWidth }}
                >
                    <Tabs defaultValue="tasks" className="h-full flex flex-col">
                        <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0">
                            <TabsTrigger
                                value="tasks"
                                className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent text-xs"
                            >
                                <List className="h-3.5 w-3.5 mr-1" />
                                Tasks
                            </TabsTrigger>
                            <TabsTrigger
                                value="mcp"
                                className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent text-xs"
                            >
                                <Plug className="h-3.5 w-3.5 mr-1" />
                                MCP
                            </TabsTrigger>
                            <TabsTrigger
                                value="sandbox"
                                className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent text-xs"
                            >
                                <Play className="h-3.5 w-3.5 mr-1" />
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
                            <div className="p-4 text-center text-muted-foreground text-sm">
                                <div className="py-8 space-y-2">
                                    <Play className="h-8 w-8 mx-auto opacity-30" />
                                    <p>Sandbox Management</p>
                                    <p className="text-xs">Coming soon (Phase 5)</p>
                                </div>
                            </div>
                        </TabsContent>
                    </Tabs>
                </div>

                {/* Resize Handle */}
                <div
                    className="w-1 bg-border hover:bg-primary/50 cursor-col-resize transition-colors shrink-0"
                    onMouseDown={(e) => {
                        e.preventDefault()
                        const startX = e.clientX
                        const startWidth = leftPanelWidth

                        const onMouseMove = (moveEvent: MouseEvent) => {
                            const newWidth = Math.max(180, Math.min(400, startWidth + moveEvent.clientX - startX))
                            setLeftPanelWidth(newWidth)
                        }

                        const onMouseUp = () => {
                            document.removeEventListener('mousemove', onMouseMove)
                            document.removeEventListener('mouseup', onMouseUp)
                        }

                        document.addEventListener('mousemove', onMouseMove)
                        document.addEventListener('mouseup', onMouseUp)
                    }}
                />

                {/* Center Panel: Execution Log + Prompt */}
                <div className="flex-1 flex flex-col min-w-0">
                    <div className="flex-1 overflow-hidden">
                        <ExecutionLog />
                    </div>
                    <PromptInput
                        onSubmit={handlePromptSubmit}
                        isLoading={isProcessing}
                    />
                </div>
            </div>
        </div>
    )
}
