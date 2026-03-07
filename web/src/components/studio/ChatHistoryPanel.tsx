"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { useStudioStore } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Plus, MessageSquare } from "lucide-react"
import { cn } from "@/lib/utils"

export function ChatHistoryPanel() {
  const { tasks, activeTaskId, setActiveTask, selectedPaperId } = useStudioStore()

  const paperTasks = useMemo(() => {
    return tasks
      .filter(task => task.paperId === selectedPaperId)
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
  }, [tasks, selectedPaperId])

  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 60_000)
    return () => clearInterval(id)
  }, [])

  const relativeTime = useCallback((date: Date) => {
    const diff = Date.now() - new Date(date).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 60) return `${mins}m`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h`
    const days = Math.floor(hours / 24)
    if (days < 7) return `${days}d`
    const weeks = Math.floor(days / 7)
    return `${weeks}w`
  }, [])

  return (
    <div className="h-full flex flex-col border-r">
      <div className="px-3 py-2.5 border-b flex items-center justify-between">
        <span className="text-sm font-medium">Threads</span>
        <Button variant="ghost" size="icon" className="h-6 w-6" title="New thread">
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-1.5 space-y-0.5">
          {paperTasks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <MessageSquare className="h-5 w-5 mb-2 opacity-30" />
              <p className="text-xs">No conversations yet</p>
            </div>
          ) : (
            paperTasks.map(task => (
              <button
                key={task.id}
                onClick={() => setActiveTask(task.id)}
                className={cn(
                  "w-full text-left px-3 py-2 rounded-md transition-colors text-xs",
                  task.id === activeTaskId
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-muted"
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{task.name}</span>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {relativeTime(task.createdAt)}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
