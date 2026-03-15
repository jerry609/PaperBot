"use client"

import { useAgentEventStore } from "@/lib/agent-events/store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FilePlus2, FileEdit, ChevronRight } from "lucide-react"
import { InlineDiffPanel } from "./InlineDiffPanel"
import type { FileTouchedEntry } from "@/lib/agent-events/types"
import { cn } from "@/lib/utils"

export function FileListPanel() {
  const filesTouched = useAgentEventStore((s) => s.filesTouched)
  const selectedRunId = useAgentEventStore((s) => s.selectedRunId)
  const selectedFile = useAgentEventStore((s) => s.selectedFile)
  const setSelectedFile = useAgentEventStore((s) => s.setSelectedFile)

  // If a file is selected, show the diff panel
  if (selectedFile) {
    return <InlineDiffPanel entry={selectedFile} onBack={() => setSelectedFile(null)} />
  }

  // Build filtered file list
  const entries: FileTouchedEntry[] = selectedRunId
    ? (filesTouched[selectedRunId] ?? [])
    : Object.values(filesTouched).flat()

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="px-3 py-2 border-b border-gray-700 shrink-0">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Files {selectedRunId ? `(run ${selectedRunId.slice(0, 8)})` : "(all runs)"}
        </p>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {entries.length === 0 ? (
          <div className="flex items-center justify-center py-12 text-xs text-gray-500">
            No file changes yet
          </div>
        ) : (
          <ul className="px-2 py-2 space-y-1">
            {entries.map((entry) => (
              <li key={`${entry.run_id}-${entry.path}`}>
                <button
                  className={cn(
                    "w-full flex items-center gap-2 px-2 py-2 rounded-md text-xs transition-colors",
                    "hover:bg-accent hover:text-accent-foreground text-gray-300",
                  )}
                  onClick={() => setSelectedFile(entry)}
                >
                  {entry.status === "created" ? (
                    <FilePlus2 size={14} className="text-green-400 shrink-0" />
                  ) : (
                    <FileEdit size={14} className="text-amber-400 shrink-0" />
                  )}
                  <span className="font-mono flex-1 truncate text-left" title={entry.path}>
                    {entry.path}
                  </span>
                  {entry.linesAdded !== undefined && (
                    <span className="text-green-400 shrink-0">+{entry.linesAdded}</span>
                  )}
                  <ChevronRight size={12} className="text-gray-600 shrink-0" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}
