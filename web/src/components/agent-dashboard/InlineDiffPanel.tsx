"use client"

import { DiffViewer } from "@/components/studio/DiffViewer"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { FileTouchedEntry } from "@/lib/agent-events/types"

interface InlineDiffPanelProps {
  entry: FileTouchedEntry
  onBack: () => void
}

export function InlineDiffPanel({ entry, onBack }: InlineDiffPanelProps) {
  const hasContent =
    entry.oldContent !== undefined || entry.newContent !== undefined || entry.diff !== undefined

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-700 shrink-0">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={onBack}
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <span className="font-mono text-xs text-gray-300 truncate" title={entry.path}>
          {entry.path}
        </span>
      </div>

      {/* Diff content */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {!hasContent ? (
          <div className="flex items-center justify-center h-full text-xs text-gray-500 px-4 text-center">
            Diff not available for this change
          </div>
        ) : (
          <DiffViewer
            oldValue={entry.oldContent ?? ""}
            newValue={entry.newContent ?? entry.diff ?? ""}
            filename={entry.path}
          />
        )}
      </div>
    </div>
  )
}
