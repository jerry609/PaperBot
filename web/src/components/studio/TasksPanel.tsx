"use client"

import { useStudioStore, Task } from "@/lib/store/studio-store"
import { cn } from "@/lib/utils"
import { Circle, CheckCircle2, Loader2, AlertCircle, Clock, Sparkles } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

const statusConfig = {
    running: { icon: Loader2, color: "text-blue-500", bg: "bg-blue-50 dark:bg-blue-950/30", animate: true },
    completed: { icon: CheckCircle2, color: "text-green-500", bg: "bg-green-50 dark:bg-green-950/30", animate: false },
    pending: { icon: Circle, color: "text-muted-foreground", bg: "bg-muted/30", animate: false },
    error: { icon: AlertCircle, color: "text-red-500", bg: "bg-red-50 dark:bg-red-950/30", animate: false },
}

function formatRelativeTime(date: Date): string {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)

    if (seconds < 60) return 'Just now'
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    return date.toLocaleDateString()
}

interface TaskItemProps {
    task: Task
    isActive: boolean
    onClick: () => void
}

function TaskItem({ task, isActive, onClick }: TaskItemProps) {
    const config = statusConfig[task.status]
    const Icon = config.icon

    return (
        <button
            onClick={onClick}
            className={cn(
                "w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all",
                "hover:bg-muted/50",
                isActive && "bg-muted ring-1 ring-border"
            )}
        >
            <div className={cn(
                "mt-0.5 w-5 h-5 flex items-center justify-center rounded-md shrink-0",
                config.bg
            )}>
                <Icon className={cn(
                    "h-3 w-3",
                    config.color,
                    config.animate && "animate-spin"
                )} />
            </div>
            <div className="flex-1 min-w-0">
                <p className={cn(
                    "text-sm font-medium truncate",
                    isActive ? "text-foreground" : "text-foreground/80"
                )}>
                    {task.name}
                </p>
                <div className="flex items-center gap-2 mt-0.5">
                    <span className={cn(
                        "text-[10px] font-medium uppercase tracking-wider",
                        config.color
                    )}>
                        {task.status}
                    </span>
                    <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="h-2.5 w-2.5" />
                        {formatRelativeTime(task.createdAt)}
                    </span>
                </div>
                {task.actions.length > 0 && (
                    <p className="text-[10px] text-muted-foreground mt-1">
                        {task.actions.length} action{task.actions.length !== 1 ? 's' : ''}
                    </p>
                )}
            </div>
        </button>
    )
}

export function TasksPanel() {
    const { tasks, activeTaskId, setActiveTask } = useStudioStore()

    const sortedTasks = [...tasks].sort((a, b) =>
        b.createdAt.getTime() - a.createdAt.getTime()
    )

    return (
        <ScrollArea className="h-full">
            <div className="p-2 space-y-1">
                {tasks.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-muted-foreground py-12 px-4">
                        <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mb-3">
                            <Sparkles className="h-6 w-6 opacity-30" />
                        </div>
                        <p className="text-sm font-medium">No tasks yet</p>
                        <p className="text-xs text-center mt-1">
                            Tasks will appear here when you start a conversation
                        </p>
                    </div>
                ) : (
                    sortedTasks.map(task => (
                        <TaskItem
                            key={task.id}
                            task={task}
                            isActive={activeTaskId === task.id}
                            onClick={() => setActiveTask(task.id)}
                        />
                    ))
                )}
            </div>
        </ScrollArea>
    )
}
