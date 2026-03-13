"use client"

import { Info } from "lucide-react"
import type { AgentTaskLog } from "@/lib/store/studio-store"

export function InfoBlock({ log }: { log: AgentTaskLog }) {
  return (
    <div className="flex items-start gap-2 py-1.5 px-3">
      <Info className="h-3.5 w-3.5 text-zinc-400 mt-0.5 shrink-0" />
      <span className="text-xs text-zinc-400">{log.message}</span>
    </div>
  )
}
