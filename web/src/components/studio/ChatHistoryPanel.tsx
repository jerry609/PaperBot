"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { useStudioStore, type Task } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Plus, MessageSquare } from "lucide-react"
import { cn } from "@/lib/utils"

function compactText(value: string): string {
  return value.replace(/\s+/g, " ").trim()
}

function buildThreadPreview(task: Task): string {
  const lastAssistant = [...task.history].reverse().find((entry) => entry.role === "assistant")?.content
  if (lastAssistant) return compactText(lastAssistant)

  const lastTextAction = [...task.actions]
    .reverse()
    .find((action) => action.type === "text" || action.type === "user")?.content
  if (lastTextAction) return compactText(lastTextAction)

  if (task.status === "running") return "Waiting for Claude Code..."
  return "New thread"
}

function taskStatusMeta(status: Task["status"]): { label: string; dotClassName: string } {
  if (status === "running") {
    return { label: "Live", dotClassName: "bg-amber-500" }
  }
  if (status === "completed") {
    return { label: "Done", dotClassName: "bg-emerald-500" }
  }
  if (status === "error") {
    return { label: "Error", dotClassName: "bg-rose-500" }
  }
  return { label: "Idle", dotClassName: "bg-slate-300" }
}

export function ChatHistoryPanel() {
  const { tasks, activeTaskId, setActiveTask, selectedPaperId } = useStudioStore()

  const paperTasks = useMemo(() => {
    return tasks
      .filter(task => task.paperId === selectedPaperId && task.kind === "chat")
      .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
  }, [tasks, selectedPaperId])

  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 60_000)
    return () => clearInterval(id)
  }, [])

  const relativeTime = useCallback((date: Date) => {
    const diff = Date.now() - new Date(date).getTime()
    if (diff < 60000) return "now"
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
    <div className="flex h-full flex-col border-r border-slate-200 bg-[#f8f9f6]">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Threads</p>
          <p className="mt-1 text-[11px] text-slate-500">{paperTasks.length} saved</p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 rounded-md border border-transparent px-2 text-[11px] text-slate-600 hover:bg-white hover:text-slate-900"
          title="New thread"
          onClick={() => setActiveTask(null)}
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-2 p-2">
          {paperTasks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-slate-500">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-200 bg-white">
                <MessageSquare className="h-4 w-4 opacity-40" />
              </div>
              <p className="mt-3 text-[12px] font-medium text-slate-800">No threads yet</p>
              <p className="mt-1 max-w-[180px] text-center text-[11px] leading-5 text-slate-500">
                Start a Claude Code conversation to create the first thread.
              </p>
            </div>
          ) : (
            paperTasks.map((task) => {
              const status = taskStatusMeta(task.status)
              const preview = buildThreadPreview(task)
              const messageCount = task.history.length

              return (
                <button
                  key={task.id}
                  onClick={() => setActiveTask(task.id)}
                  className={cn(
                    "w-full rounded-2xl border px-3 py-3 text-left transition-colors",
                    task.id === activeTaskId
                      ? "border-slate-300 bg-[#edf0e8] shadow-sm"
                      : "border-transparent bg-transparent hover:border-slate-200 hover:bg-white",
                  )}
                >
                  <div className="flex items-start gap-2">
                    <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", status.dotClassName)} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-[12px] font-medium text-slate-900">{task.name}</span>
                        <span className="shrink-0 text-[10px] text-slate-400">
                          {relativeTime(task.updatedAt)}
                        </span>
                      </div>

                      <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-slate-500">
                        {preview}
                      </p>

                      <div className="mt-2 flex items-center justify-between gap-2 text-[10px] text-slate-400">
                        <span>{messageCount > 0 ? `${messageCount} msg${messageCount === 1 ? "" : "s"}` : "Draft"}</span>
                        <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 uppercase tracking-[0.12em] text-slate-500">
                          {status.label}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              )
            })
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
