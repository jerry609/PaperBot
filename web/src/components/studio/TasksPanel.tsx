"use client"

import { useStudioStore, Task } from "@/lib/store/studio-store"
import { cn } from "@/lib/utils"
import { Circle, CheckCircle2, Loader2, AlertCircle } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

const statusConfig = {
    running: { icon: Loader2, color: "text-blue-500", animate: true },
    completed: { icon: CheckCircle2, color: "text-green-500", animate: false },
    pending: { icon: Circle, color: "text-muted-foreground", animate: false },
    error: { icon: AlertCircle, color: "text-red-500", animate: false },
}

export function TasksPanel() {
    const { tasks, activeTaskId, setActiveTask } = useStudioStore()

    return (
        <div className="h-full flex flex-col bg-muted/10 border-r">
            <div className="p-3 border-b">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Tasks
                </h3>
            </div>

            <ScrollArea className="flex-1">
                <div className="p-2 space-y-1">
                    {tasks.length === 0 ? (
                        <div className="text-center text-muted-foreground text-xs py-8">
                            No tasks yet
                        </div>
                    ) : (
                        tasks.map(task => {
                            const config = statusConfig[task.status]
                            const Icon = config.icon
                            return (
                                <div
                                    key={task.id}
                                    onClick={() => setActiveTask(task.id)}
                                    className={cn(
                                        "flex items-center gap-2 px-2 py-2 rounded-md text-sm cursor-pointer hover:bg-muted/50 transition-colors",
                                        activeTaskId === task.id && "bg-muted"
                                    )}
                                >
                                    <Icon className={cn("h-4 w-4", config.color, config.animate && "animate-spin")} />
                                    <span className="truncate flex-1">{task.name}</span>
                                </div>
                            )
                        })
                    )}
                </div>
            </ScrollArea>
        </div>
    )
}
