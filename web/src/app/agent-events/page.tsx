"use client"

import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { AgentStatusPanel } from "@/components/agent-events/AgentStatusPanel"
import { ToolCallTimeline } from "@/components/agent-events/ToolCallTimeline"

export default function AgentEventsPage() {
  // Mount the SSE hook exactly once at the page root.
  // Child components read from the Zustand store directly.
  useAgentEvents()

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-6 py-3">
        <h1 className="text-lg font-bold text-gray-100">Agent Events (Debug)</h1>
        <p className="text-xs text-gray-500 mt-0.5">
          Real-time view of agent lifecycle and tool call events from /api/events/stream
        </p>
      </header>

      <main className="flex-1 flex flex-col gap-0 overflow-hidden">
        {/* Agent status panel — compact strip at top */}
        <div className="border-b border-gray-800 bg-gray-900/50">
          <AgentStatusPanel />
        </div>

        {/* Two-column: activity feed (left) + tool call timeline (right) */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-0 overflow-hidden">
          <div className="border-r border-gray-800 overflow-hidden flex flex-col min-h-0">
            <ActivityFeed />
          </div>
          <div className="overflow-hidden flex flex-col min-h-0">
            <ToolCallTimeline />
          </div>
        </div>
      </main>
    </div>
  )
}
