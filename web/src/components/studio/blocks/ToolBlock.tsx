"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight, Terminal } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AgentTaskLog } from "@/lib/store/studio-store"

export function ToolBlock({ log }: { log: AgentTaskLog }) {
  const [expanded, setExpanded] = useState(false)
  const isError = log.level === "error" || log.message.toLowerCase().includes("error:")
  const obsPreview = (log.details?.observation_preview as string) ?? ""

  // Strip [step N] prefix for cleaner display
  const command = log.message.replace(/^\[step \d+\]\s*/, "")

  return (
    <div className="py-1.5 px-3 space-y-1">
      <button
        className="flex items-center gap-2 w-full text-left"
        onClick={() => obsPreview && setExpanded(!expanded)}
      >
        {obsPreview ? (
          expanded ? (
            <ChevronDown className="h-3 w-3 text-zinc-400 shrink-0" />
          ) : (
            <ChevronRight className="h-3 w-3 text-zinc-400 shrink-0" />
          )
        ) : (
          <Terminal className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
        )}
        <Terminal className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
        <code className="text-xs font-mono text-zinc-300 truncate">{command}</code>
        <span
          className={cn(
            "h-2 w-2 rounded-full shrink-0 ml-auto",
            isError ? "bg-red-500" : "bg-emerald-500",
          )}
        />
      </button>
      {expanded && obsPreview && (
        <div className="ml-6 bg-zinc-900 rounded px-2.5 py-1.5 max-h-24 overflow-y-auto">
          <pre className="text-[11px] font-mono text-zinc-400 whitespace-pre-wrap break-all leading-relaxed">
            {obsPreview}
          </pre>
        </div>
      )}
    </div>
  )
}
