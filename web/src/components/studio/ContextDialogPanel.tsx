"use client"

import Link from "next/link"
import { useMemo, useState, type ReactNode } from "react"
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  FolderOpen,
  Loader2,
  Play,
  Sparkles,
  Wrench,
  X,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ContextPackPanel } from "./ContextPackPanel"
import { cn } from "@/lib/utils"
import { getStudioSkillPaths, getStudioSkillTools } from "@/lib/studio-skills"
import type { StudioAttachedSkill } from "@/lib/store/studio-store"
import type {
  ContextPackSession,
  GenerationStatus,
  ReproContextPack,
  StageObservationsEvent,
  StageProgressEvent,
} from "@/lib/types/p2c"

type PaperLite = {
  id: string
  title: string
  abstract: string
}

type ContextModuleState = "ready" | "missing" | "generating"

interface Props {
  selectedPaper: PaperLite | null
  projectDir: string | null
  generationStatus: GenerationStatus
  generationProgress: StageProgressEvent[]
  liveObservations: StageObservationsEvent[]
  contextPack: ReproContextPack | null
  contextPackLoading: boolean
  contextPackError: string | null
  attachedSkill: StudioAttachedSkill | null
  onGenerate: (paper: PaperLite) => void
  onInsertSkill?: (slashCommand: string) => void
  onClearSkill?: () => void
  onSessionCreated?: (session: ContextPackSession) => void
  onDeployToBoard?: () => void
}

const MODULE_LABELS: Record<string, string> = {
  paper_brief: "Paper brief",
  literature: "Literature",
  environment: "Environment",
  spec: "Spec",
  roadmap: "Roadmap",
  success_criteria: "Success criteria",
  workspace: "Workspace",
}

const MODULE_DESCRIPTIONS: Record<string, string> = {
  paper_brief: "Uses the selected paper title and abstract already loaded in Studio.",
  literature: "Needs extracted paper context before the skill can use distilled prior work.",
  environment: "Needs extracted environment and dependency details from the paper context pack.",
  spec: "Needs extracted implementation and hyperparameter details from the paper context pack.",
  roadmap: "Needs extracted roadmap steps and execution hints from the paper context pack.",
  success_criteria: "Needs extracted evaluation targets and acceptance criteria from the paper context pack.",
  workspace: "Needs the active Code workspace path so Claude Code can edit and run commands.",
}

const FALLBACK_MODULES_BY_TARGET: Record<string, string[]> = {
  paper: ["paper_brief"],
  context_pack: ["literature", "environment", "spec", "roadmap", "success_criteria"],
  workspace: ["workspace"],
}

const STAGE_LABELS: Record<string, string> = {
  literature_distill: "Literature Distill",
  blueprint_extract: "Blueprint Extract",
  environment_extract: "Environment Extract",
  spec_extract: "Spec & Hyperparams",
  roadmap_planning: "Roadmap Planning",
  success_criteria: "Success Criteria",
}

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>()
  const normalized: string[] = []
  for (const value of values) {
    const cleaned = value.trim()
    if (!cleaned || seen.has(cleaned)) continue
    seen.add(cleaned)
    normalized.push(cleaned)
  }
  return normalized
}

function resolveAttachedSkillModules(skill: StudioAttachedSkill | null): string[] {
  if (!skill) return []
  if (Array.isArray(skill.contextModules) && skill.contextModules.length > 0) {
    return uniqueStrings(skill.contextModules)
  }
  return uniqueStrings(
    (skill.recommendedFor ?? []).flatMap((target) => FALLBACK_MODULES_BY_TARGET[target] ?? []),
  )
}

function moduleLabel(module: string): string {
  return MODULE_LABELS[module] ?? module.replace(/[_-]+/g, " ")
}

function moduleDescription(module: string): string {
  return MODULE_DESCRIPTIONS[module] ?? "This module supports the current skill attachment."
}

function moduleStateClassName(state: ContextModuleState): string {
  if (state === "ready") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (state === "generating") return "border-amber-200 bg-amber-50 text-amber-700"
  return "border-slate-200 bg-[#f7f8f4] text-slate-600"
}

function moduleStateLabel(state: ContextModuleState): string {
  if (state === "ready") return "ready"
  if (state === "generating") return "generating"
  return "missing"
}

function contextStatusLabel(
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
    label: "Standby",
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
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 px-4 py-4 lg:px-5 lg:py-5">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">{eyebrow}</p>
          <h2 className="mt-1 text-[18px] font-semibold tracking-[-0.02em] text-slate-950">{title}</h2>
          <p className="mt-1 max-w-[54rem] text-sm leading-6 text-slate-600">{description}</p>
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div className="px-4 py-4 lg:px-5 lg:py-5">{children}</div>
    </section>
  )
}

function StatusPill({
  label,
  className,
}: {
  label: string
  className: string
}) {
  return (
    <span className={cn("rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]", className)}>
      {label}
    </span>
  )
}

function ContextModuleRow({
  module,
  state,
}: {
  module: string
  state: ContextModuleState
}) {
  return (
    <div className="rounded-[18px] border border-slate-200 bg-[#fafaf7] px-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-slate-900">{moduleLabel(module)}</p>
          <p className="mt-1 text-[12px] leading-5 text-slate-600">{moduleDescription(module)}</p>
        </div>
        <StatusPill label={moduleStateLabel(state)} className={moduleStateClassName(state)} />
      </div>
    </div>
  )
}

function ContextActivityRow({
  label,
  detail,
  state,
}: {
  label: string
  detail: string
  state: ContextModuleState
}) {
  return (
    <div className="rounded-[16px] border border-slate-200 bg-white px-3 py-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[12px] font-medium text-slate-900">{label}</p>
        <StatusPill label={moduleStateLabel(state)} className={moduleStateClassName(state)} />
      </div>
      <p className="mt-1 text-[12px] leading-5 text-slate-600">{detail}</p>
    </div>
  )
}

export function ContextDialogPanel({
  selectedPaper,
  projectDir,
  generationStatus,
  generationProgress,
  liveObservations,
  contextPack,
  contextPackLoading,
  contextPackError,
  attachedSkill,
  onGenerate,
  onInsertSkill,
  onClearSkill,
  onSessionCreated,
  onDeployToBoard,
}: Props) {
  const [showFullPack, setShowFullPack] = useState(false)

  const contextStatus = contextStatusLabel(
    generationStatus,
    contextPack,
    contextPackLoading,
    contextPackError,
  )
  const requiredModules = useMemo(() => resolveAttachedSkillModules(attachedSkill), [attachedSkill])
  const skillPaths = useMemo(() => (attachedSkill ? getStudioSkillPaths(attachedSkill) : []), [attachedSkill])
  const skillTools = useMemo(() => (attachedSkill ? getStudioSkillTools(attachedSkill) : []), [attachedSkill])
  const hasGeneratedContextModules = requiredModules.some(
    (module) => module !== "paper_brief" && module !== "workspace",
  )

  const moduleStates = useMemo(() => {
    const states = new Map<string, ContextModuleState>()
    for (const module of requiredModules) {
      if (module === "workspace") {
        states.set(module, projectDir ? "ready" : "missing")
        continue
      }
      if (module === "paper_brief") {
        states.set(module, selectedPaper ? "ready" : "missing")
        continue
      }
      if (generationStatus === "generating" || contextPackLoading) {
        states.set(module, "generating")
        continue
      }
      states.set(module, contextPack ? "ready" : "missing")
    }
    return states
  }, [contextPack, contextPackLoading, generationStatus, projectDir, requiredModules, selectedPaper])

  const missingGeneratedContext = requiredModules.filter(
    (module) => module !== "paper_brief" && module !== "workspace" && moduleStates.get(module) === "missing",
  )

  const recentActivity = useMemo(() => {
    const recentProgressEvents = generationProgress.slice(-3)
    const progressOffset = Math.max(generationProgress.length - recentProgressEvents.length, 0)
    const progressEvents = recentProgressEvents.map((event, index) => ({
      key: `progress:${progressOffset + index}:${event.stage}:${event.progress}:${event.message ?? ""}`,
      label: stageLabel(event.stage),
      detail: event.message || "Context extraction is in progress.",
      state: event.progress >= 1 ? ("ready" as const) : ("generating" as const),
    }))

    const recentObservationEvents = liveObservations.slice(-2)
    const observationOffset = Math.max(liveObservations.length - recentObservationEvents.length, 0)
    const observationEvents = recentObservationEvents.map((event, index) => ({
      key: `obs:${observationOffset + index}:${event.stage}:${event.observations.map((item) => item.id).join(",")}`,
      label: `${stageLabel(event.stage)} findings`,
      detail:
        event.observations.length > 0
          ? `${event.observations.length} extracted finding${event.observations.length === 1 ? "" : "s"} emitted.`
          : "No observations emitted yet.",
      state: contextPack || generationStatus === "completed" ? ("ready" as const) : ("generating" as const),
    }))

    return [...progressEvents, ...observationEvents]
  }, [contextPack, generationProgress, generationStatus, liveObservations])

  return (
    <div className="h-full overflow-auto bg-[#f5f5f2]">
      <div className="mx-auto flex w-full max-w-[1040px] flex-col gap-4 px-3 py-4 md:px-4 md:py-5 xl:px-5">
        <SectionCard
          eyebrow="Attached Skill"
          title={attachedSkill ? attachedSkill.title : "No skill attached"}
          description={
            attachedSkill
              ? "Studio keeps only the current thread’s skill attachment here. Change discovery and setup happens on the top-level Skills page."
              : "Attach a skill from the top-level Skills page before using Studio as a skill-shaped chat surface."
          }
          action={
            <div className="flex flex-wrap items-center gap-2">
              <Button asChild variant="outline" size="sm" className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700 hover:bg-slate-50">
                <Link href="/skills">{attachedSkill ? "Change skill" : "Browse Skills"}</Link>
              </Button>
              {attachedSkill && onInsertSkill ? (
                <Button
                  type="button"
                  size="sm"
                  className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
                  onClick={() => onInsertSkill(attachedSkill.slashCommand)}
                >
                  Use slash
                </Button>
              ) : null}
              {attachedSkill && onClearSkill ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700 hover:bg-slate-50"
                  onClick={onClearSkill}
                >
                  <X className="mr-1.5 h-3.5 w-3.5" />
                  Remove
                </Button>
              ) : null}
            </div>
          }
        >
          {attachedSkill ? (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="h-5 text-[10px]">
                  {attachedSkill.slashCommand}
                </Badge>
                <Badge variant="outline" className="h-5 text-[10px]">
                  {attachedSkill.scope}
                </Badge>
                {attachedSkill.repoLabel ? (
                  <Badge variant="outline" className="h-5 text-[10px]">
                    {attachedSkill.repoLabel}
                  </Badge>
                ) : null}
                {attachedSkill.repoRef ? (
                  <Badge variant="outline" className="h-5 text-[10px]">
                    {attachedSkill.repoRef}
                  </Badge>
                ) : null}
              </div>

              <div className="grid gap-3 lg:grid-cols-2">
                <div className="rounded-[18px] border border-slate-200 bg-[#fafaf7] px-3 py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">How Studio uses it</p>
                  <p className="mt-1 text-sm leading-6 text-slate-700">
                    This skill is attached to the current paper thread. Natural-language turns can reuse it without retyping the slash command.
                  </p>
                </div>
                <div className="rounded-[18px] border border-slate-200 bg-[#fafaf7] px-3 py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Source path</p>
                  <p className="mt-1 break-all font-mono text-[11px] text-slate-700">
                    {skillPaths[0] ?? "Skill path unavailable"}
                  </p>
                </div>
              </div>

              {skillTools.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {skillTools.slice(0, 4).map((tool) => (
                    <Badge key={`${attachedSkill.key}:${tool}`} variant="outline" className="h-5 text-[10px]">
                      {tool}
                    </Badge>
                  ))}
                  {skillTools.length > 4 ? (
                    <Badge variant="outline" className="h-5 text-[10px]">
                      +{skillTools.length - 4} tools
                    </Badge>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="rounded-[20px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 h-5 w-5 text-slate-400" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-900">Keep skill selection outside Studio</p>
                  <p className="mt-1 text-sm leading-6 text-slate-600">
                    Use the Skills page for discovery, install, detail, and setup. Studio only mirrors the skill already attached to this thread.
                  </p>
                  {contextPack ? (
                    <p className="mt-2 text-[12px] leading-5 text-slate-500">
                      A paper context pack is already available and will be reused once a skill is attached.
                    </p>
                  ) : null}
                </div>
              </div>
            </div>
          )}
        </SectionCard>

        <SectionCard
          eyebrow="Skill Context"
          title={attachedSkill ? "Readiness for the attached skill" : "Context standby"}
          description={
            attachedSkill
              ? "Only the context required by the current attached skill is shown here. Generate paper context when the skill actually needs extracted paper details."
              : "Attach a skill first. Then Studio will show only the context modules that skill depends on."
          }
          action={
            <div className="flex flex-wrap items-center gap-2">
              <StatusPill label={contextStatus.label} className={contextStatus.className} />
              {attachedSkill && hasGeneratedContextModules ? (
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
                  {contextPack ? "Refresh context" : "Generate context"}
                </Button>
              ) : null}
            </div>
          }
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline" className="h-5 text-[10px]">
                {selectedPaper ? selectedPaper.title : "No paper selected"}
              </Badge>
              <Badge variant="outline" className="h-5 text-[10px]">
                {projectDir ? "Workspace ready" : "Workspace missing"}
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

            {!attachedSkill ? (
              <div className="rounded-[18px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4 text-sm leading-6 text-slate-600">
                Studio will keep this panel narrow on purpose. Attach a skill to see only the paper-context modules that thread actually needs.
              </div>
            ) : (
              <>
                {requiredModules.length > 0 ? (
                  <div className="grid gap-3 lg:grid-cols-2">
                    {requiredModules.map((module) => (
                      <ContextModuleRow
                        key={`${attachedSkill.key}:${module}`}
                        module={module}
                        state={moduleStates.get(module) ?? "missing"}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-[18px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4 text-sm leading-6 text-slate-600">
                    This skill did not declare any explicit context modules. The selected paper and workspace will still be available in chat.
                  </div>
                )}

                {!selectedPaper ? (
                  <div className="rounded-[18px] border border-dashed border-slate-300 bg-[#fafaf7] px-4 py-4 text-sm leading-6 text-slate-600">
                    Select a paper first. Paper-specific context generation needs a selected paper before it can run.
                  </div>
                ) : null}

                {missingGeneratedContext.length > 0 ? (
                  <div className="rounded-[18px] border border-slate-200 bg-[#eef1ea] px-4 py-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex items-start gap-2">
                        <Wrench className="mt-0.5 h-4 w-4 text-slate-500" />
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-slate-900">Missing generated context</p>
                          <p className="mt-1 text-sm leading-6 text-slate-600">
                            {missingGeneratedContext.map(moduleLabel).join(", ")} still need a paper context pack for this skill.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}
              </>
            )}

            {recentActivity.length > 0 ? (
              <div className="space-y-2">
                {recentActivity.map((item) => (
                  <ContextActivityRow
                    key={item.key}
                    label={item.label}
                    detail={item.detail}
                    state={item.state}
                  />
                ))}
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
                        <FileText className="h-4 w-4 text-slate-500" />
                        <p className="text-sm font-medium text-slate-900">Compiled paper context</p>
                      </div>
                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        Keep this secondary. Open it only when the attached skill needs the full extracted payload or when you want to send execution into Monitor.
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
                      Keep chat focused on the next turn. Open Monitor only when you want the full execution surface.
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

            {attachedSkill && requiredModules.every((module) => moduleStates.get(module) === "ready") ? (
              <div className="rounded-[18px] border border-emerald-200 bg-emerald-50 px-4 py-3">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-700" />
                  <p className="text-sm leading-6 text-emerald-800">
                    The attached skill has everything it needs for the next chat turn.
                  </p>
                </div>
              </div>
            ) : null}
          </div>
        </SectionCard>
      </div>
    </div>
  )
}
