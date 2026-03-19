"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { useStudioStore, type Task } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { MessageSquare, Plus, Search } from "lucide-react"
import { cn } from "@/lib/utils"

function compactText(value: string): string {
  return value.replace(/\s+/g, " ").trim()
}

function buildThreadPreview(task: Task): string {
  const lastAssistant = [...task.history].reverse().find((entry) => entry.role === "assistant")?.content
  if (lastAssistant) return compactText(lastAssistant)

  const lastUser = [...task.history].reverse().find((entry) => entry.role === "user")?.content
  if (lastUser) return compactText(lastUser)

  const lastTextAction = [...task.actions]
    .reverse()
    .find((action) => action.type === "text" || action.type === "user")?.content
  if (lastTextAction) return compactText(lastTextAction)

  if (task.status === "running") return "Waiting for Claude Code..."
  return "New thread"
}

function taskStatusMeta(status: Task["status"]): { label: string; dotClassName: string } {
  if (status === "running") {
    return { label: "Working", dotClassName: "bg-amber-500" }
  }
  if (status === "completed") {
    return { label: "Done", dotClassName: "bg-emerald-500" }
  }
  if (status === "error") {
    return { label: "Error", dotClassName: "bg-rose-500" }
  }
  return { label: "Draft", dotClassName: "bg-slate-300" }
}

export function ChatHistoryPanel() {
  const { tasks, activeTaskId, setActiveTask, selectedPaperId } = useStudioStore()
  const [searchQuery, setSearchQuery] = useState("")

  const paperTasks = useMemo(() => {
    return tasks
      .filter(task => task.paperId === selectedPaperId && task.kind === "chat")
      .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
  }, [tasks, selectedPaperId])

  const filteredTasks = useMemo(() => {
    const query = compactText(searchQuery).toLowerCase()
    if (!query) return paperTasks

    return paperTasks.filter((task) => {
      const preview = buildThreadPreview(task)
      return `${task.name} ${preview}`.toLowerCase().includes(query)
    })
  }, [paperTasks, searchQuery])

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
    <div className="flex h-full flex-col bg-[#f5f6f2]">
      <div className="border-b border-slate-200 bg-[#f4f5f1] px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 flex items-center gap-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">Threads</p>
            <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-500">
              {paperTasks.length}
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 rounded-full border border-slate-200 bg-white text-slate-600 hover:text-slate-900"
            title="New thread"
            onClick={() => setActiveTask(null)}
          >
            <Plus className="h-3 w-3" />
          </Button>
        </div>

        <label className="mt-1.5 flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-2.5 py-1 shadow-[0_1px_0_rgba(255,255,255,0.8)_inset]">
          <Search className="h-3 w-3 shrink-0 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search threads"
            className="h-4 w-full border-0 bg-transparent p-0 text-[10px] text-slate-700 placeholder:text-slate-400 focus:outline-none"
          />
        </label>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-0.5 p-2">
          {paperTasks.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-[18px] border border-dashed border-slate-200 bg-white/60 px-4 py-8 text-slate-500">
              <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-200 bg-[#f7f8f4]">
                <MessageSquare className="h-4 w-4 opacity-40" />
              </div>
              <p className="mt-2.5 text-[12px] font-medium text-slate-800">No threads yet</p>
              <p className="mt-1 max-w-[180px] text-center text-[10px] leading-4 text-slate-500">
                Start a Claude Code conversation.
              </p>
            </div>
          ) : filteredTasks.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-[18px] border border-dashed border-slate-200 bg-white/60 px-4 py-8 text-slate-500">
              <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-200 bg-[#f7f8f4]">
                <Search className="h-4 w-4 opacity-40" />
              </div>
              <p className="mt-2.5 text-[12px] font-medium text-slate-800">No matching threads</p>
              <p className="mt-1 max-w-[180px] text-center text-[10px] leading-4 text-slate-500">
                Try a title or reply snippet.
              </p>
            </div>
          ) : (
            filteredTasks.map((task) => {
              const status = taskStatusMeta(task.status)
              const preview = buildThreadPreview(task)
              const messageCount = task.history.length

              return (
                <button
                  key={task.id}
                  onClick={() => setActiveTask(task.id)}
                  className={cn(
                    "group relative w-full rounded-[14px] px-3 py-1.5 text-left transition-[background-color,box-shadow]",
                    task.id === activeTaskId
                      ? "bg-[#eef1ea] shadow-[inset_0_0_0_1px_rgba(148,163,184,0.14),0_1px_0_rgba(255,255,255,0.9)]"
                      : "bg-transparent hover:bg-white/75",
                  )}
                >
                  {task.id === activeTaskId ? (
                    <span className="absolute left-0 top-2.5 bottom-2.5 w-[2px] rounded-full bg-slate-400" />
                  ) : null}
                  <div className="flex items-start gap-2">
                    <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full transition-opacity", status.dotClassName, task.id === activeTaskId ? "opacity-100" : "opacity-80 group-hover:opacity-100")} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-[10px] font-medium text-slate-900">{task.name}</span>
                        <span className="shrink-0 text-[9px] uppercase tracking-[0.12em] text-slate-400">
                          {relativeTime(task.updatedAt)}
                        </span>
                      </div>

                      <p className="mt-0.5 line-clamp-1 text-[9px] leading-4 text-slate-500">
                        {preview}
                      </p>

                      <div className="mt-1 flex items-center gap-1 text-[9px] text-slate-400">
                        <span className={cn(
                          "rounded-full border px-1.5 py-0.5 uppercase tracking-[0.12em]",
                          task.id === activeTaskId
                            ? "border-slate-300 bg-white text-slate-600"
                            : "border-slate-200 bg-[#f7f8f4] text-slate-500",
                        )}>
                          {status.label}
                        </span>
                        <span>{messageCount > 0 ? `${messageCount} msg` : "Draft"}</span>
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
