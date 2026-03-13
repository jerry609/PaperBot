"use client"

import { Brain } from "lucide-react"
import type { AgentTaskLog } from "@/lib/store/studio-store"

export function ThinkBlock({ log }: { log: AgentTaskLog }) {
  // Strip leading [step N] prefix if present
  const text = log.message.replace(/^\[step \d+\]\s*/, "")

  return (
    <div className="flex items-start gap-2.5 py-2 px-3">
      <Brain className="h-3.5 w-3.5 text-indigo-400 mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-xs text-indigo-700 italic leading-relaxed whitespace-pre-wrap break-words">
          &ldquo;{text}&rdquo;
        </p>
      </div>
    </div>
  )
}
