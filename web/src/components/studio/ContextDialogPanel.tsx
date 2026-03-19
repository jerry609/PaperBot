"use client"

import { useMemo, useState, type ElementType, type ReactNode } from "react"
import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Loader2,
  Package,
  Play,
  Sparkles,
  Wrench,
} from "lucide-react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ContextPackPanel } from "./ContextPackPanel"
import { cn } from "@/lib/utils"
import type { StudioSkillInfo } from "@/lib/studio-runtime"
import type {
  ContextPackSession,
  GenerationStatus,
  ReproContextPack,
  StageObservationsEvent,
  StageProgressEvent,
} from "@/lib/types/p2c"

const STAGE_LABELS: Record<string, string> = {
  literature_distill: "Literature Distill",
  blueprint_extract: "Blueprint Extract",
  environment_extract: "Environment Extract",
  spec_extract: "Spec & Hyperparams",
  roadmap_planning: "Roadmap Planning",
  success_criteria: "Success Criteria",
}

const STAGE_ORDER = Object.keys(STAGE_LABELS)

type PaperLite = {
  id: string
  title: string
  abstract: string
}

type TimelineItem =
  | { kind: "progress"; key: string; event: StageProgressEvent }
  | { kind: "observations"; key: string; event: StageObservationsEvent }
  | { kind: "summary"; key: string; stage: string; count: number }

interface Props {
  selectedPaper: PaperLite | null
  generationStatus: GenerationStatus
  generationProgress: StageProgressEvent[]
  liveObservations: StageObservationsEvent[]
  contextPack: ReproContextPack | null
  contextPackLoading: boolean
  contextPackError: string | null
  skills: StudioSkillInfo[]
  onGenerate: (paper: PaperLite) => void
  onInsertSkill?: (slashCommand: string) => void
  onSessionCreated?: (session: ContextPackSession) => void
  onDeployToBoard?: () => void
}

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] || stage
}

function stageRank(stage: string): number {
  const index = STAGE_ORDER.indexOf(stage)
  return index === -1 ? 99 : index
}

function Actor({
  icon: Icon,
  tone = "assistant",
}: {
  icon: ElementType
  tone?: "assistant" | "tool" | "success" | "error" | "muted"
}) {
  return (
    <Avatar
      className={cn(
        "mt-0.5 h-7 w-7 border",
        tone === "assistant" && "border-primary/20 bg-primary/10 text-primary",
        tone === "tool" && "border-amber-200 bg-amber-50 text-amber-700",
        tone === "success" && "border-emerald-200 bg-emerald-50 text-emerald-700",
        tone === "error" && "border-red-200 bg-red-50 text-red-600",
        tone === "muted" && "border-border bg-muted text-muted-foreground",
      )}
    >
      <AvatarFallback className="bg-transparent">
        <Icon className="h-3.5 w-3.5" />
      </AvatarFallback>
    </Avatar>
  )
}

function MessageShell({
  icon,
  actor,
  badge,
  tone = "assistant",
  children,
}: {
  icon: ElementType
  actor: string
  badge?: string
  tone?: "assistant" | "tool" | "success" | "error" | "muted"
  children: ReactNode
}) {
  return (
    <div className="flex items-start gap-3">
      <Actor icon={icon} tone={tone} />
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2 text-[11px] text-muted-foreground">
          <span className="font-medium">{actor}</span>
          {badge ? (
            <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
              {badge}
            </Badge>
          ) : null}
        </div>
        {children}
      </div>
    </div>
  )
}

function TextCard({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border bg-card px-3.5 py-3 text-sm leading-6 text-foreground/90">
      {children}
    </div>
  )
}

function ProgressFormCard({ event }: { event: StageProgressEvent }) {
  const pct = Math.round(event.progress * 100)
  const isDone = pct >= 100
  return (
    <MessageShell
      icon={isDone ? CheckCircle2 : Activity}
      actor="Extractor"
      badge={stageLabel(event.stage)}
      tone={isDone ? "success" : "tool"}
    >
      <div className="overflow-hidden rounded-xl border bg-muted/20">
        <div className="border-b bg-background/70 px-3 py-2 text-sm font-medium">
          {isDone ? "Stage completed" : "Stage running"}
        </div>
        <div className="px-3 py-2 text-sm text-foreground/90">
          {event.message || "Running extraction..."}
        </div>
        <div className="px-3 pb-3">
          <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
            <span>Progress</span>
            <span>{pct}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-background/70">
            <div
              className={cn("h-full rounded-full transition-all", isDone ? "bg-emerald-500" : "bg-primary")}
              style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
            />
          </div>
        </div>
      </div>
    </MessageShell>
  )
}

function ObservationFormCard({ event }: { event: StageObservationsEvent }) {
  return (
    <MessageShell icon={Wrench} actor="Tool Result" badge={stageLabel(event.stage)} tone="assistant">
      <div className="overflow-hidden rounded-xl border bg-muted/20">
        <div className="border-b bg-background/70 px-3 py-2 text-sm font-medium">
          Extracted findings ({event.observations.length})
        </div>
        {event.observations.length === 0 ? (
          <div className="px-3 py-2 text-sm text-muted-foreground">No observations emitted yet.</div>
        ) : (
          <div className="divide-y">
            {event.observations.slice(0, 6).map((obs) => (
              <div key={obs.id} className="flex items-center gap-2 px-3 py-2 text-sm">
                <span className="text-muted-foreground">•</span>
                <Badge variant="outline" className="text-[10px]">
                  {obs.type}
                </Badge>
                <span className="min-w-0 flex-1 truncate text-foreground/90">{obs.title}</span>
                <span className="text-xs text-muted-foreground">{Math.round(obs.confidence * 100)}%</span>
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
            ))}
            {event.observations.length > 6 ? (
              <div className="px-3 py-2 text-xs text-muted-foreground">
                +{event.observations.length - 6} more observations
              </div>
            ) : null}
          </div>
        )}
      </div>
    </MessageShell>
  )
}

function StageSummaryText({ stage, count }: { stage: string; count: number }) {
  return (
    <MessageShell icon={Bot} actor="DeepCode" badge="Summary" tone="muted">
      <TextCard>
        Completed {stageLabel(stage)} and extracted {count} observation{count === 1 ? "" : "s"}.
        Next stage will continue refining reproducibility signals.
      </TextCard>
    </MessageShell>
  )
}

export function ContextDialogPanel({
  selectedPaper,
  generationStatus,
  generationProgress,
  liveObservations,
  contextPack,
  contextPackLoading,
  contextPackError,
  skills,
  onGenerate,
  onInsertSkill,
  onSessionCreated,
  onDeployToBoard,
}: Props) {
  const [showFullPack, setShowFullPack] = useState(false)

  // When restoring from cache/refresh, generationProgress and liveObservations
  // may be empty even though contextPack is available.  Reconstruct synthetic
  // timeline entries from the pack so the progress section isn't blank.
  const { effectiveProgress, effectiveObservations } = useMemo(() => {
    if (generationProgress.length > 0 || liveObservations.length > 0 || !contextPack) {
      return { effectiveProgress: generationProgress, effectiveObservations: liveObservations }
    }

    // Group observations by stage, preserving STAGE_ORDER
    const obsByStage = new Map<string, StageObservationsEvent["observations"]>()
    for (const obs of contextPack.observations) {
      const list = obsByStage.get(obs.stage) ?? []
      list.push({ id: obs.id, type: obs.type, title: obs.title, confidence: obs.confidence })
      obsByStage.set(obs.stage, list)
    }

    const syntheticProgress: StageProgressEvent[] = []
    const syntheticObservations: StageObservationsEvent[] = []

    for (const stage of STAGE_ORDER) {
      const stageObs = obsByStage.get(stage)
      if (!stageObs || stageObs.length === 0) continue
      syntheticProgress.push({ stage, progress: 1, message: "Stage completed" })
      syntheticObservations.push({ stage, observations: stageObs })
    }

    // Include any stages not in STAGE_ORDER
    for (const [stage, stageObs] of obsByStage) {
      if (STAGE_ORDER.includes(stage)) continue
      syntheticProgress.push({ stage, progress: 1, message: "Stage completed" })
      syntheticObservations.push({ stage, observations: stageObs })
    }

    return { effectiveProgress: syntheticProgress, effectiveObservations: syntheticObservations }
  }, [generationProgress, liveObservations, contextPack])

  const progressTimeline = useMemo(() => {
    const reduced: StageProgressEvent[] = []
    for (const event of effectiveProgress) {
      const prev = reduced[reduced.length - 1]
      const messageChanged = (event.message || "") !== (prev?.message || "")
      const stageChanged = prev?.stage !== event.stage
      const progressMoved = !prev || Math.abs(event.progress - prev.progress) >= 0.12
      if (!prev || stageChanged || messageChanged || progressMoved) {
        reduced.push(event)
      }
    }
    return reduced
  }, [effectiveProgress])

  const sortedLiveObservations = useMemo(
    () => [...effectiveObservations].sort((a, b) => stageRank(a.stage) - stageRank(b.stage)),
    [effectiveObservations],
  )

  const timelineItems = useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = []
    const obsByStage = new Map(sortedLiveObservations.map((entry) => [entry.stage, entry]))
    const lastProgressIndexByStage = new Map<string, number>()
    const renderedObservationStages = new Set<string>()

    progressTimeline.forEach((event, index) => {
      lastProgressIndexByStage.set(event.stage, index)
    })

    progressTimeline.forEach((event, index) => {
      items.push({ kind: "progress", key: `progress-${event.stage}-${index}`, event })
      const observationEvent = obsByStage.get(event.stage)
      if (observationEvent && lastProgressIndexByStage.get(event.stage) === index) {
        items.push({ kind: "observations", key: `obs-inline-${event.stage}`, event: observationEvent })
        items.push({
          kind: "summary",
          key: `summary-inline-${event.stage}`,
          stage: event.stage,
          count: observationEvent.observations.length,
        })
        renderedObservationStages.add(event.stage)
      }
    })

    sortedLiveObservations.forEach((event) => {
      if (!renderedObservationStages.has(event.stage)) {
        items.push({ kind: "observations", key: `obs-late-${event.stage}`, event })
        items.push({
          kind: "summary",
          key: `summary-late-${event.stage}`,
          stage: event.stage,
          count: event.observations.length,
        })
      }
    })

    return items
  }, [progressTimeline, sortedLiveObservations])

  const hasTimelineActivity =
    timelineItems.length > 0 || contextPackLoading || Boolean(contextPack) || Boolean(contextPackError)
  const recommendedSkills = useMemo(() => {
    if (skills.length === 0) return []
    const targetKinds = contextPack ? ["context_pack", "paper"] : ["paper"]
    const prioritized = skills.filter((skill) =>
      skill.recommendedFor.some((entry) => targetKinds.includes(entry)),
    )
    return prioritized.length > 0 ? prioritized : skills
  }, [contextPack, skills])

  return (
    <div className="h-full overflow-auto bg-gradient-to-b from-muted/25 via-background to-background">
      <div className="mx-auto flex w-full max-w-[860px] flex-col gap-4 px-3 py-4 md:px-4 md:py-5">
        <MessageShell icon={Bot} actor="DeepCode" badge="Skill Context" tone="muted">
          <TextCard>
            Progress and skill context are now presented in a mixed format:
            narrative text plus structured form cards.
          </TextCard>
        </MessageShell>

        {!hasTimelineActivity ? (
          <MessageShell icon={Sparkles} actor="DeepCode" badge="Action" tone="assistant">
            <TextCard>
              <div className="space-y-3">
                <p>
                  Build skill context for the selected paper. I will stream progress as form cards and add
                  concise narrative summaries between steps.
                </p>
                {selectedPaper ? (
                  <Button
                    size="sm"
                    onClick={() => onGenerate(selectedPaper)}
                    disabled={generationStatus === "generating"}
                  >
                    {generationStatus === "generating" ? (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="mr-1.5 h-4 w-4" />
                    )}
                    Build Skill Context
                  </Button>
                ) : (
                  <p>Select a paper to begin.</p>
                )}
              </div>
            </TextCard>
          </MessageShell>
        ) : null}

        {timelineItems.map((item) => {
          if (item.kind === "progress") return <ProgressFormCard key={item.key} event={item.event} />
          if (item.kind === "observations") return <ObservationFormCard key={item.key} event={item.event} />
          return <StageSummaryText key={item.key} stage={item.stage} count={item.count} />
        })}

        {contextPackLoading ? (
          <MessageShell icon={Loader2} actor="DeepCode" badge="Finalizing" tone="assistant">
            <TextCard>
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Finalizing skill context payload and actions...
              </span>
            </TextCard>
          </MessageShell>
        ) : null}

        {contextPackError ? (
          <MessageShell icon={AlertCircle} actor="System" badge="Error" tone="error">
            <TextCard>{contextPackError}</TextCard>
          </MessageShell>
        ) : null}

        {contextPack ? (
          <>
            <MessageShell icon={Package} actor="DeepCode" badge="Skill Context" tone="success">
              <TextCard>
                <div className="space-y-2">
                  <p>{contextPack.paper.title}</p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">
                      Confidence {Math.round(contextPack.confidence.overall * 100)}%
                    </Badge>
                    <Badge variant="outline">{contextPack.observations.length} observations</Badge>
                    <Badge variant="outline">{contextPack.task_roadmap.length} roadmap steps</Badge>
                  </div>
                  <div className="pt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setShowFullPack((value) => !value)}
                    >
                      {showFullPack ? (
                        <ChevronUp className="mr-1.5 h-4 w-4" />
                      ) : (
                        <ChevronDown className="mr-1.5 h-4 w-4" />
                      )}
                      {showFullPack ? "Hide skill context" : "Open skill context"}
                    </Button>
                  </div>
                </div>
              </TextCard>
            </MessageShell>

            {recommendedSkills.length > 0 ? (
              <MessageShell icon={Sparkles} actor="DeepCode" badge="Project Skills" tone="assistant">
                <div className="rounded-xl border bg-card px-3.5 py-3">
                  <div className="mb-3">
                    <p className="text-sm font-medium text-foreground">Apply a project skill to this paper</p>
                    <p className="mt-1 text-sm leading-6 text-foreground/70">
                      Studio can insert project skills into chat as slash commands and compile them into the managed Claude flow.
                    </p>
                  </div>
                  <div className="space-y-2">
                    {recommendedSkills.map((skill) => (
                      <div
                        key={`${skill.scope}:${skill.id}`}
                        className="rounded-xl border bg-background/70 px-3 py-3"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <p className="text-sm font-medium text-foreground">{skill.title}</p>
                              <Badge variant="outline" className="text-[10px]">
                                {skill.scope}
                              </Badge>
                              <Badge variant="outline" className="font-mono text-[10px]">
                                {skill.slashCommand}
                              </Badge>
                            </div>
                            {skill.description ? (
                              <p className="mt-1 text-sm leading-6 text-foreground/70">{skill.description}</p>
                            ) : null}
                            {skill.tools.length > 0 ? (
                              <p className="mt-2 text-xs text-muted-foreground">
                                Tools: {skill.tools.join(", ")}
                              </p>
                            ) : null}
                          </div>
                          {onInsertSkill ? (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => onInsertSkill(skill.slashCommand)}
                            >
                              Insert
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </MessageShell>
            ) : null}

            {showFullPack ? (
              <div className="rounded-2xl border bg-card/70">
                <ContextPackPanel
                  pack={contextPack}
                  onSessionCreated={onSessionCreated}
                  onDeployToBoard={onDeployToBoard}
                  className="h-auto overflow-visible p-4"
                />
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  )
}
