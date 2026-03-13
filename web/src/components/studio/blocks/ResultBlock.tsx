"use client"

import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AgentTaskLog } from "@/lib/store/studio-store"

export function ResultBlock({ log }: { log: AgentTaskLog }) {
  const isSuccess = log.level === "success" || log.event === "human_approved"
  const isWarning = log.level === "warning" || log.event === "human_requested_changes"
  const Icon = isSuccess ? CheckCircle2 : isWarning ? AlertTriangle : XCircle
  const borderColor = isSuccess
    ? "border-emerald-200 bg-emerald-50/50"
    : isWarning
      ? "border-amber-200 bg-amber-50/50"
      : "border-red-200 bg-red-50/50"
  const iconColor = isSuccess
    ? "text-emerald-600"
    : isWarning
      ? "text-amber-600"
      : "text-red-600"
  const textColor = isSuccess
    ? "text-emerald-800"
    : isWarning
      ? "text-amber-800"
      : "text-red-800"

  const summary = (log.details?.summary as string) ?? ""
  // Strip [step N] prefix
  const message = log.message.replace(/^\[step \d+\]\s*/, "")

  return (
    <div className="py-1.5 px-3">
      <div className={cn("rounded-lg border px-3 py-2.5 flex items-start gap-2.5", borderColor)}>
        <Icon className={cn("h-4 w-4 mt-0.5 shrink-0", iconColor)} />
        <div className="min-w-0 flex-1 space-y-0.5">
          <p className={cn("text-xs font-medium leading-relaxed", textColor)}>
            {message}
          </p>
          {summary && (
            <p className="text-[11px] text-zinc-500 whitespace-pre-wrap break-words">
              {summary}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
