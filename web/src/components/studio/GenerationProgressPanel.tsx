"use client"

import { CheckCircle2, Circle, Loader2, AlertCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import type { StageObservationsEvent, StageProgressEvent } from "@/lib/types/p2c"
import { cn } from "@/lib/utils"

const STAGES: Array<{ key: string; label: string }> = [
  { key: "literature_distill", label: "Literature Distill" },
  { key: "blueprint_extract", label: "Blueprint Extract" },
  { key: "environment_extract", label: "Environment Extract" },
  { key: "spec_extract", label: "Spec & Hyperparams" },
  { key: "roadmap_planning", label: "Roadmap Planning" },
  { key: "success_criteria", label: "Success Criteria" },
]

interface Props {
  stages: StageProgressEvent[]
  liveObservations?: StageObservationsEvent[]
  error?: string | null
}

export function GenerationProgressPanel({ stages, liveObservations = [], error }: Props) {
  const lastStage = stages.length > 0 ? stages[stages.length - 1] : null
  const currentStage = lastStage?.stage || null
  const currentIndex = currentStage
    ? STAGES.findIndex((stage) => stage.key === currentStage)
    : -1
  const overallProgress = lastStage ? Math.round(lastStage.progress * 100) : 0

  const stageLabel = (stageKey: string) =>
    STAGES.find((stage) => stage.key === stageKey)?.label || stageKey

  return (
    <div className="p-6 space-y-4">
      <div className="space-y-1">
        <h3 className="text-base font-semibold">Generating Reproduction Context...</h3>
        {lastStage?.message && (
          <p className="text-xs text-muted-foreground">{lastStage.message}</p>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          <AlertCircle className="h-4 w-4 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <div className="space-y-3">
        {STAGES.map((stage, index) => {
          const isComplete = currentIndex > index || (currentIndex === index && overallProgress >= 100)
          const isActive = currentIndex === index && !isComplete
          const Icon = isComplete ? CheckCircle2 : isActive ? Loader2 : Circle

          return (
            <div key={stage.key} className="flex items-center gap-3 text-sm">
              <Icon
                className={cn(
                  "h-4 w-4",
                  isComplete && "text-green-600",
                  isActive && "text-primary animate-spin",
                  !isComplete && !isActive && "text-muted-foreground"
                )}
              />
              <span className={cn(isComplete && "text-green-700")}>{stage.label}</span>
            </div>
          )
        })}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Overall progress</span>
          <span>{overallProgress}%</span>
        </div>
        <Progress value={overallProgress} />
      </div>

      <div className="space-y-2">
        <div className="text-sm font-medium">Observation initialization</div>
        {liveObservations.length === 0 ? (
          <p className="text-xs text-muted-foreground">Waiting for observations…</p>
        ) : (
          <div className="space-y-3">
            {liveObservations.map((stage) => (
              <div key={stage.stage} className="space-y-2">
                <div className="text-xs text-muted-foreground">{stageLabel(stage.stage)}</div>
                <div className="space-y-1">
                  {stage.observations.map((obs) => (
                    <div key={obs.id} className="flex items-center gap-2 text-xs">
                      <Badge variant="outline" className="text-[10px]">{obs.type}</Badge>
                      <span className="flex-1 truncate">{obs.title}</span>
                      <span className="text-muted-foreground">{Math.round(obs.confidence * 100)}%</span>
                    </div>
                  ))}
                  {stage.observations.length === 0 && (
                    <div className="text-xs text-muted-foreground">No observations yet.</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
