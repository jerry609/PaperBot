"use client"

import { useState } from "react"
import { Columns3, LayoutGrid, GitBranch } from "lucide-react"
import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { SplitPanels } from "@/components/layout/SplitPanels"
import { TasksPanel } from "@/components/agent-dashboard/TasksPanel"
import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { FileListPanel } from "@/components/agent-dashboard/FileListPanel"
import { KanbanBoard } from "@/components/agent-dashboard/KanbanBoard"
import { AgentDagPanel } from "@/components/agent-dashboard/AgentDagPanel"
import { useStudioStore } from "@/lib/store/studio-store"

export default function AgentDashboardPage() {
  useAgentEvents()

  const [viewMode, setViewMode] = useState<"panels" | "kanban" | "dag">("panels")

  const studioTasks = useStudioStore((s) => s.agentTasks)
  const eventKanbanTasks = useAgentEventStore((s) => s.kanbanTasks)
  const kanbanTasks = studioTasks.length > 0 ? studioTasks : eventKanbanTasks

  return (
    <div className="h-screen min-h-0 flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 shrink-0">
        <h1 className="text-sm font-semibold">Agent Dashboard</h1>
        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={() => setViewMode("panels")}
            className={`p-1.5 rounded ${viewMode === "panels" ? "bg-muted" : "hover:bg-muted/50"}`}
            title="Panels view"
            aria-label="Switch to panels view"
          >
            <Columns3 className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode("kanban")}
            className={`p-1.5 rounded ${viewMode === "kanban" ? "bg-muted" : "hover:bg-muted/50"}`}
            title="Kanban view"
            aria-label="Switch to kanban view"
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode("dag")}
            className={`p-1.5 rounded ${viewMode === "dag" ? "bg-muted" : "hover:bg-muted/50"}`}
            title="DAG view"
            aria-label="Switch to DAG view"
          >
            <GitBranch className="h-4 w-4" />
          </button>
        </div>
      </header>
      <div className="flex-1 min-h-0">
        {viewMode === "panels" && (
          <SplitPanels
            storageKey="agent-dashboard"
            rail={<TasksPanel />}
            list={<ActivityFeed />}
            detail={<FileListPanel />}
          />
        )}
        {viewMode === "kanban" && (
          <KanbanBoard tasks={kanbanTasks} />
        )}
        {viewMode === "dag" && (
          <AgentDagPanel />
        )}
      </div>
    </div>
  )
}
