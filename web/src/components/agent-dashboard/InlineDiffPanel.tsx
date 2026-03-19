"use client"

import { DiffViewer } from "@/components/studio/DiffViewer"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { FileTouchedEntry } from "@/lib/agent-events/types"

interface InlineDiffPanelProps {
  entry: FileTouchedEntry
  onBack: () => void
}

export function InlineDiffPanel({ entry, onBack }: Readonly<InlineDiffPanelProps>) {
  const hasContent =
    entry.oldContent !== undefined || entry.newContent !== undefined || entry.diff !== undefined

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header bar */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-200 px-3 py-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={onBack}
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <span className="truncate font-mono text-xs text-zinc-600" title={entry.path}>
          {entry.path}
        </span>
      </div>

      {/* Diff content */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {!hasContent ? (
          <div className="flex h-full items-center justify-center px-4 text-center text-xs text-zinc-500">
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
