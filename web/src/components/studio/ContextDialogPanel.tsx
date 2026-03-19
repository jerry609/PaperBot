"use client"

import { useMemo, useState, type ReactNode } from "react"
import {
  Activity,
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
  FileText,
  FolderOpen,
  Loader2,
  Package,
  Play,
  Sparkles,
  Wrench,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ContextPackPanel } from "./ContextPackPanel"
import { cn } from "@/lib/utils"
import type { StudioSkillInfo } from "@/lib/studio-runtime"
import {
  buildStudioSkillAvailabilityLabel,
  formatStudioSkillEcosystemLabel,
} from "@/lib/studio-skills"
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
const SKILL_DIRECTORY_HINTS = [".claude/skills", ".opencode/skills", ".github/skills"]
const RECOMMENDED_TARGET_LABELS: Record<string, string> = {
  paper: "Paper",
  context_pack: "Paper context",
  workspace: "Workspace",
}

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

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => value.trim().length > 0)))
}

function formatRecommendedTargets(skill: StudioSkillInfo): string[] {
  return skill.recommendedFor.map((entry) => RECOMMENDED_TARGET_LABELS[entry] ?? entry)
}

function buildContextStatusLabel(
  generationStatus: GenerationStatus,
  contextPack: ReproContextPack | null,
  contextPackLoading: boolean,
  contextPackError: string | null,
): { label: string; className: string } {
  if (contextPackError) {
    return {
      label: "Needs attention",
      className: "border-rose-200 bg-rose-50 text-rose-700",
    }
  }
  if (generationStatus === "generating" || contextPackLoading) {
    return {
      label: "Generating",
      className: "border-amber-200 bg-amber-50 text-amber-700",
    }
  }
  if (contextPack) {
    return {
      label: "Ready",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    }
  }
  return {
    label: "Optional",
    className: "border-slate-200 bg-[#f7f8f4] text-slate-600",
  }
}

function SectionCard({
  eyebrow,
  title,
  description,
  action,
  children,
}: {
  eyebrow: string
  title: string
  description: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="rounded-[24px] border border-slate-200 bg-white/95 shadow-[0_16px_32px_rgba(15,23,42,0.04)]">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 px-4 py-4">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">{eyebrow}</p>
          <h2 className="mt-1 text-[18px] font-semibold tracking-[-0.02em] text-slate-950">{title}</h2>
          <p className="mt-1 max-w-[48rem] text-sm leading-6 text-slate-600">{description}</p>
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div className="px-4 py-4">{children}</div>
    </section>
  )
}

function ContextProgressCard({ event }: { event: StageProgressEvent }) {
  const pct = Math.round(event.progress * 100)
  const isDone = pct >= 100

  return (
    <div className="rounded-[20px] border border-slate-200 bg-[#fafaf7] px-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {isDone ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
          ) : (
            <Activity className="h-4 w-4 text-amber-600" />
          )}
          <p className="text-sm font-medium text-slate-900">{stageLabel(event.stage)}</p>
        </div>
        <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
          {isDone ? "Completed" : `${pct}%`}
        </span>
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-600">{event.message || "Running extraction..."}</p>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-200">
        <div
          className={cn("h-full rounded-full transition-all", isDone ? "bg-emerald-500" : "bg-slate-900")}
          style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
        />
      </div>
    </div>
  )
}

function ContextObservationCard({ event }: { event: StageObservationsEvent }) {
  return (
    <div className="rounded-[20px] border border-slate-200 bg-white px-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Wrench className="h-4 w-4 text-slate-500" />
          <p className="text-sm font-medium text-slate-900">{stageLabel(event.stage)}</p>
        </div>
        <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
          {event.observations.length} findings
        </span>
      </div>
      {event.observations.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">No observations emitted yet.</p>
      ) : (
        <div className="mt-3 space-y-2">
          {event.observations.slice(0, 5).map((observation) => (
            <div
              key={observation.id}
              className="flex items-center gap-2 rounded-[14px] border border-slate-200 bg-[#fafaf7] px-2.5 py-2"
            >
              <Badge variant="outline" className="text-[10px]">
                {observation.type}
              </Badge>
              <span className="min-w-0 flex-1 truncate text-sm text-slate-700">{observation.title}</span>
              <span className="text-[11px] text-slate-500">{Math.round(observation.confidence * 100)}%</span>
            </div>
          ))}
          {event.observations.length > 5 ? (
            <p className="text-xs text-slate-500">+{event.observations.length - 5} more findings</p>
          ) : null}
        </div>
      )}
    </div>
  )
}

function ContextSummaryCard({ stage, count }: { stage: string; count: number }) {
  return (
    <div className="rounded-[18px] border border-slate-200 bg-[#eef1ea] px-3 py-2.5 text-sm leading-6 text-slate-700">
      {stageLabel(stage)} completed with {count} extracted signal{count === 1 ? "" : "s"}.
    </div>
  )
}

function SkillCard({
  skill,
  recommended,
  copied,
  onInsertSkill,
  onCopySkill,
}: {
  skill: StudioSkillInfo
  recommended: boolean
  copied: boolean
  onInsertSkill?: (slashCommand: string) => void
  onCopySkill: (slashCommand: string, skillId: string) => void
}) {
  const ecosystems =
    skill.ecosystems.length > 0
      ? skill.ecosystems
      : skill.primaryEcosystem
        ? [skill.primaryEcosystem]
        : []
  const targets = formatRecommendedTargets(skill)
  const availabilityLabel = buildStudioSkillAvailabilityLabel(skill)
  const preferredPath = skill.path || skill.paths[0] || "Skill directory unavailable"

  return (
    <div className="rounded-[22px] border border-slate-200 bg-[#fafaf7] px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-[15px] font-semibold text-slate-950">{skill.title}</p>
            {recommended ? (
              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-emerald-700">
                Suggested
              </span>
            ) : null}
            <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
              {skill.scope}
            </span>
          </div>
          <p className="mt-1.5 text-sm leading-6 text-slate-600">
            {skill.description || "No description provided in this skill manifest."}
          </p>
        </div>
        <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-mono text-[10px] text-slate-600">
          {skill.slashCommand}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {ecosystems.map((ecosystem) => (
          <Badge key={`${skill.id}:${ecosystem}`} variant="outline" className="h-5 text-[10px]">
            {formatStudioSkillEcosystemLabel(ecosystem)}
          </Badge>
        ))}
        {targets.map((target) => (
          <Badge key={`${skill.id}:target:${target}`} variant="outline" className="h-5 text-[10px]">
            {target}
          </Badge>
        ))}
        {skill.tools.slice(0, 3).map((tool) => (
          <Badge key={`${skill.id}:tool:${tool}`} variant="outline" className="h-5 text-[10px]">
            {tool}
          </Badge>
        ))}
        {skill.tools.length > 3 ? (
          <Badge variant="outline" className="h-5 text-[10px]">
            +{skill.tools.length - 3} tools
          </Badge>
        ) : null}
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <div className="rounded-[16px] border border-slate-200 bg-white px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Availability</p>
          <p className="mt-1 text-[11px] text-slate-700">{availabilityLabel}</p>
        </div>
        <div className="rounded-[16px] border border-slate-200 bg-white px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Source path</p>
          <p className="mt-1 break-all font-mono text-[11px] text-slate-700">{preferredPath}</p>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {onInsertSkill ? (
          <Button
            type="button"
            className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
            onClick={() => onInsertSkill(skill.slashCommand)}
          >
            Use in chat
          </Button>
        ) : null}
        <Button
          type="button"
          variant="outline"
          className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700"
          onClick={() => onCopySkill(skill.slashCommand, skill.id)}
        >
          {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy slash"}
        </Button>
      </div>
    </div>
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
  const [copiedSkillId, setCopiedSkillId] = useState<string | null>(null)

  const { effectiveProgress, effectiveObservations } = useMemo(() => {
    if (generationProgress.length > 0 || liveObservations.length > 0 || !contextPack) {
      return { effectiveProgress: generationProgress, effectiveObservations: liveObservations }
    }

    const observationsByStage = new Map<string, StageObservationsEvent["observations"]>()
    for (const observation of contextPack.observations) {
      const list = observationsByStage.get(observation.stage) ?? []
      list.push({
        id: observation.id,
        type: observation.type,
        title: observation.title,
        confidence: observation.confidence,
      })
      observationsByStage.set(observation.stage, list)
    }

    const syntheticProgress: StageProgressEvent[] = []
    const syntheticObservations: StageObservationsEvent[] = []

    for (const stage of STAGE_ORDER) {
      const stageObservations = observationsByStage.get(stage)
      if (!stageObservations || stageObservations.length === 0) continue
      syntheticProgress.push({ stage, progress: 1, message: "Stage completed" })
      syntheticObservations.push({ stage, observations: stageObservations })
    }

    for (const [stage, stageObservations] of observationsByStage) {
      if (STAGE_ORDER.includes(stage)) continue
      syntheticProgress.push({ stage, progress: 1, message: "Stage completed" })
      syntheticObservations.push({ stage, observations: stageObservations })
    }

    return { effectiveProgress: syntheticProgress, effectiveObservations: syntheticObservations }
  }, [contextPack, generationProgress, liveObservations])

  const progressTimeline = useMemo(() => {
    const reduced: StageProgressEvent[] = []
    for (const event of effectiveProgress) {
      const previous = reduced[reduced.length - 1]
      const messageChanged = (event.message || "") !== (previous?.message || "")
      const stageChanged = previous?.stage !== event.stage
      const progressMoved = !previous || Math.abs(event.progress - previous.progress) >= 0.12
      if (!previous || stageChanged || messageChanged || progressMoved) {
        reduced.push(event)
      }
    }
    return reduced
  }, [effectiveProgress])

  const sortedLiveObservations = useMemo(
    () => [...effectiveObservations].sort((left, right) => stageRank(left.stage) - stageRank(right.stage)),
    [effectiveObservations],
  )

  const timelineItems = useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = []
    const observationsByStage = new Map(sortedLiveObservations.map((entry) => [entry.stage, entry]))
    const lastProgressIndexByStage = new Map<string, number>()
    const renderedObservationStages = new Set<string>()

    progressTimeline.forEach((event, index) => {
      lastProgressIndexByStage.set(event.stage, index)
    })

    progressTimeline.forEach((event, index) => {
      items.push({ kind: "progress", key: `progress-${event.stage}-${index}`, event })
      const observationEvent = observationsByStage.get(event.stage)
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

  const contextStatus = buildContextStatusLabel(
    generationStatus,
    contextPack,
    contextPackLoading,
    contextPackError,
  )
  const targetKinds = useMemo(() => (contextPack ? ["context_pack", "paper"] : ["paper"]), [contextPack])
  const recommendedSkillIds = useMemo(() => {
    return new Set(
      skills
        .filter((skill) => skill.recommendedFor.some((entry) => targetKinds.includes(entry)))
        .map((skill) => skill.id),
    )
  }, [skills, targetKinds])

  const orderedSkills = useMemo(() => {
    return [...skills].sort((left, right) => {
      const leftRecommended = recommendedSkillIds.has(left.id) ? 0 : 1
      const rightRecommended = recommendedSkillIds.has(right.id) ? 0 : 1
      if (leftRecommended !== rightRecommended) {
        return leftRecommended - rightRecommended
      }
      return left.title.localeCompare(right.title)
    })
  }, [recommendedSkillIds, skills])

  const ecosystemBadges = useMemo(
    () =>
      uniqueStrings(
        orderedSkills.flatMap((skill) =>
          skill.ecosystems.length > 0
            ? skill.ecosystems
            : skill.primaryEcosystem
              ? [skill.primaryEcosystem]
              : [],
        ),
      ),
    [orderedSkills],
  )

  const hasTimelineActivity =
    timelineItems.length > 0 || contextPackLoading || Boolean(contextPack) || Boolean(contextPackError)

  async function copySkillSlash(slashCommand: string, skillId: string) {
    try {
      await navigator.clipboard.writeText(slashCommand)
      setCopiedSkillId(skillId)
      window.setTimeout(() => {
        setCopiedSkillId((current) => (current === skillId ? null : current))
      }, 1500)
    } catch {
      setCopiedSkillId(null)
    }
  }

  return (
    <div className="h-full overflow-auto bg-[#f5f5f2]">
      <div className="mx-auto flex w-full max-w-[900px] flex-col gap-4 px-3 py-4 md:px-4 md:py-5">
        <SectionCard
          eyebrow="Skills"
          title="Project skills"
          description="Auto-discovered project skills should be the primary entry point here. They match the standard skill-directory model used by Claude Code, OpenCode, and compatible agent surfaces."
          action={
            <div className="flex flex-wrap justify-end gap-1.5">
              <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
                {orderedSkills.length} skill{orderedSkills.length === 1 ? "" : "s"}
              </span>
              {ecosystemBadges.map((ecosystem) => (
                <span
                  key={ecosystem}
                  className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500"
                >
                  {formatStudioSkillEcosystemLabel(ecosystem)}
                </span>
              ))}
            </div>
          }
        >
          {orderedSkills.length === 0 ? (
            <div className="rounded-[20px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 h-5 w-5 text-slate-400" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-900">No compatible skills discovered</p>
                  <p className="mt-1 text-sm leading-6 text-slate-600">
                    Studio now looks for standard skill folders instead of treating paper context as the skill surface.
                  </p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {SKILL_DIRECTORY_HINTS.map((hint) => (
                      <Badge key={hint} variant="outline" className="font-mono text-[10px]">
                        {hint}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {selectedPaper ? (
                <div className="rounded-[18px] border border-slate-200 bg-[#eef1ea] px-3 py-2.5 text-sm text-slate-700">
                  Suggested skills are prioritized for <span className="font-medium text-slate-900">{selectedPaper.title}</span>.
                </div>
              ) : null}
              <div className="grid gap-3 lg:grid-cols-2">
                {orderedSkills.map((skill) => (
                  <SkillCard
                    key={`${skill.scope}:${skill.id}`}
                    skill={skill}
                    recommended={recommendedSkillIds.has(skill.id)}
                    copied={copiedSkillId === skill.id}
                    onInsertSkill={onInsertSkill}
                    onCopySkill={copySkillSlash}
                  />
                ))}
              </div>
            </div>
          )}
        </SectionCard>

        <SectionCard
          eyebrow="Paper Context"
          title="Paper context"
          description="Paper context is secondary. Generate it when a skill or chat turn needs extracted roadmap steps, environment details, and reproducibility signals for the selected paper."
          action={
            <div className="flex flex-wrap items-center justify-end gap-2">
              <span
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]",
                  contextStatus.className,
                )}
              >
                {contextStatus.label}
              </span>
              <Button
                type="button"
                size="sm"
                className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
                onClick={() => selectedPaper && onGenerate(selectedPaper)}
                disabled={!selectedPaper || generationStatus === "generating"}
              >
                {generationStatus === "generating" ? (
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Play className="mr-1.5 h-3.5 w-3.5" />
                )}
                {contextPack ? "Refresh context" : "Build context"}
              </Button>
            </div>
          }
        >
          <div className="space-y-3">
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline" className="h-5 text-[10px]">
                {selectedPaper ? selectedPaper.title : "No paper selected"}
              </Badge>
              {contextPack ? (
                <>
                  <Badge variant="outline" className="h-5 text-[10px]">
                    {contextPack.observations.length} findings
                  </Badge>
                  <Badge variant="outline" className="h-5 text-[10px]">
                    {contextPack.task_roadmap.length} roadmap steps
                  </Badge>
                  <Badge variant="outline" className="h-5 text-[10px]">
                    Confidence {Math.round(contextPack.confidence.overall * 100)}%
                  </Badge>
                </>
              ) : null}
            </div>

            {!selectedPaper ? (
              <div className="rounded-[18px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4 text-sm leading-6 text-slate-600">
                Select a paper first. Skills can still be inspected now, but paper context needs a selected paper.
              </div>
            ) : null}

            {!hasTimelineActivity && selectedPaper ? (
              <div className="rounded-[18px] border border-slate-200 bg-[#fafaf7] px-4 py-4">
                <div className="flex items-start gap-3">
                  <FileText className="mt-0.5 h-5 w-5 text-slate-400" />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-900">No paper context yet</p>
                    <p className="mt-1 text-sm leading-6 text-slate-600">
                      Build it only when the current skill or Claude turn needs paper-specific extraction beyond the abstract and title.
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            {timelineItems.length > 0 ? (
              <div className="space-y-2.5">
                {timelineItems.map((item) => {
                  if (item.kind === "progress") {
                    return <ContextProgressCard key={item.key} event={item.event} />
                  }
                  if (item.kind === "observations") {
                    return <ContextObservationCard key={item.key} event={item.event} />
                  }
                  return <ContextSummaryCard key={item.key} stage={item.stage} count={item.count} />
                })}
              </div>
            ) : null}

            {contextPackLoading ? (
              <div className="rounded-[18px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
                  Finalizing the paper context payload...
                </span>
              </div>
            ) : null}

            {contextPackError ? (
              <div className="rounded-[18px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                <span className="inline-flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  {contextPackError}
                </span>
              </div>
            ) : null}

            {contextPack ? (
              <>
                <div className="rounded-[20px] border border-slate-200 bg-[#fafaf7] px-4 py-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <Package className="h-4 w-4 text-slate-500" />
                        <p className="text-sm font-medium text-slate-900">Compiled paper context</p>
                      </div>
                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        Use this when the skill needs extracted findings, roadmap steps, or board deployment.
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700"
                      onClick={() => setShowFullPack((value) => !value)}
                    >
                      {showFullPack ? (
                        <ChevronUp className="mr-1.5 h-3.5 w-3.5" />
                      ) : (
                        <ChevronDown className="mr-1.5 h-3.5 w-3.5" />
                      )}
                      {showFullPack ? "Hide full context" : "Open full context"}
                    </Button>
                  </div>
                </div>

                {showFullPack ? (
                  <div className="overflow-hidden rounded-[22px] border border-slate-200 bg-white">
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

            {contextPack && onDeployToBoard ? (
              <div className="rounded-[18px] border border-slate-200 bg-[#eef1ea] px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-start gap-2">
                    <FolderOpen className="mt-0.5 h-4 w-4 text-slate-500" />
                    <p className="text-sm leading-6 text-slate-700">
                      Once this context is ready, you can keep chat focused and move execution detail into Monitor.
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700"
                    onClick={onDeployToBoard}
                  >
                    Open Monitor
                  </Button>
                </div>
              </div>
            ) : null}
          </div>
        </SectionCard>
      </div>
    </div>
  )
}
