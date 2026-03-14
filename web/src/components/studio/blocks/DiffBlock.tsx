"use client"

import { useState } from "react"
import { FileCode, ChevronDown, ChevronRight } from "lucide-react"
import type { AgentTaskLog } from "@/lib/store/studio-store"

export function DiffBlock({
  log,
  onFileClick,
}: {
  log: AgentTaskLog
  onFileClick?: (path: string) => void
}) {
  const [expanded, setExpanded] = useState(false)

  const filePath = (log.details?.file_path as string) ?? ""
  const linesAdded = (log.details?.lines_added as number) ?? 0
  const contentPreview = (log.details?.content_preview as string) ?? ""
  const lines = contentPreview ? contentPreview.split("\n") : []

  return (
    <div className="py-1.5 px-3">
      {/* Header */}
      <button
        className="flex items-center gap-2 w-full text-left group"
        onClick={() => (contentPreview ? setExpanded(!expanded) : onFileClick?.(filePath))}
      >
        {contentPreview ? (
          expanded ? (
            <ChevronDown className="h-3 w-3 text-zinc-400 shrink-0" />
          ) : (
            <ChevronRight className="h-3 w-3 text-zinc-400 shrink-0" />
          )
        ) : (
          <FileCode className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
        )}
        <FileCode className="h-3.5 w-3.5 text-amber-500 shrink-0" />
        <span className="text-xs font-mono text-zinc-600 truncate group-hover:text-indigo-600 transition-colors">
          {filePath}
        </span>
        {linesAdded > 0 && (
          <span className="text-[10px] font-mono text-emerald-600 shrink-0">+{linesAdded}</span>
        )}
      </button>

      {/* Expandable diff content — Vibe Kanban style */}
      {expanded && lines.length > 0 && (
        <div className="mt-1.5 ml-5 rounded-lg border border-zinc-200 overflow-hidden max-h-[300px] overflow-y-auto">
          {lines.slice(0, 100).map((line, i) => (
            <div
              key={i}
              className="flex border-l-2 border-l-emerald-400 border-b border-b-zinc-100 last:border-b-0 bg-white"
            >
              <span className="text-[10px] font-mono text-zinc-300 w-8 text-right pr-2 py-0.5 shrink-0 select-none">
                {i + 1}
              </span>
              <span className="text-[11px] font-mono text-zinc-700 px-2 py-0.5 whitespace-pre-wrap break-all">
                <span className="text-emerald-500 select-none">+ </span>
                {line}
              </span>
            </div>
          ))}
          {lines.length > 100 && (
            <div className="text-[10px] text-zinc-400 text-center py-1.5 bg-zinc-50 border-t border-zinc-100">
              ... {lines.length - 100} more lines
            </div>
          )}
        </div>
      )}
    </div>
  )
}
