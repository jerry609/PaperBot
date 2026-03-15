"use client"

import { useAgentEventStore } from "@/lib/agent-events/store"
import { AgentStatusPanel } from "@/components/agent-events/AgentStatusPanel"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export function TasksPanel() {
  const feed = useAgentEventStore((s) => s.feed)
  const selectedRunId = useAgentEventStore((s) => s.selectedRunId)
  const setSelectedRunId = useAgentEventStore((s) => s.setSelectedRunId)

  // Derive unique run_ids from feed, most recent first, limit 20
  const runs: { run_id: string; agent_name: string }[] = []
  const seen = new Set<string>()
  for (const item of feed) {
    const run_id = String(item.raw.run_id ?? "")
    if (run_id && !seen.has(run_id)) {
      seen.add(run_id)
      runs.push({ run_id, agent_name: item.agent_name })
      if (runs.length >= 20) break
    }
  }

  function handleRunClick(id: string) {
    // Toggle: clicking the same run deselects it
    setSelectedRunId(selectedRunId === id ? null : id)
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Agents section */}
      <div className="px-3 pt-3 pb-1">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Agents</p>
      </div>
      <AgentStatusPanel compact />

      <div className="mx-3 my-2 border-t border-gray-700" />

      {/* Runs section */}
      <div className="px-3 pb-1">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Runs</p>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {runs.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-xs text-gray-500">
            No runs yet
          </div>
        ) : (
          <ul className="px-2 pb-2 space-y-1">
            {runs.map(({ run_id, agent_name }) => (
              <li key={run_id}>
                <button
                  className={cn(
                    "w-full text-left px-2 py-2 rounded-md text-xs transition-colors",
                    "hover:bg-accent hover:text-accent-foreground",
                    selectedRunId === run_id
                      ? "bg-accent text-accent-foreground"
                      : "text-gray-300",
                  )}
                  onClick={() => handleRunClick(run_id)}
                >
                  <span className="font-mono block truncate">{run_id.slice(0, 8)}</span>
                  <span className="text-gray-500 block truncate">{agent_name}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}
