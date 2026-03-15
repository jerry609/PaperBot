"use client"

import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { SplitPanels } from "@/components/layout/SplitPanels"
import { TasksPanel } from "@/components/agent-dashboard/TasksPanel"
import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { FileListPanel } from "@/components/agent-dashboard/FileListPanel"

export default function AgentDashboardPage() {
  useAgentEvents()

  return (
    <div className="h-screen min-h-0 flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 shrink-0">
        <h1 className="text-sm font-semibold">Agent Dashboard</h1>
      </header>
      <div className="flex-1 min-h-0">
        <SplitPanels
          storageKey="agent-dashboard"
          rail={<TasksPanel />}
          list={<ActivityFeed />}
          detail={<FileListPanel />}
        />
      </div>
    </div>
  )
}
