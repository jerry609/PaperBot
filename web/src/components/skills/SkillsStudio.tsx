"use client"

import { useEffect, useMemo, useState } from "react"
import type { LucideIcon } from "lucide-react"
import {
  Blocks,
  Bot,
  Cable,
  ChevronRight,
  FileSearch,
  Filter,
  GitFork,
  GraduationCap,
  Search,
  Sparkles,
  Telescope,
  Workflow,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import type {
  SkillCardData,
  SkillPipelineStep,
  SkillsStudioData,
  SkillStatusTone,
  SkillTone,
} from "@/lib/skills-studio-types"

type SkillsStudioProps = {
  data: SkillsStudioData
}

const toneClasses: Record<SkillTone, { icon: string; border: string; glow: string; bar: string; pill: string }> = {
  teal: {
    icon: "bg-teal-500/15 text-teal-700 dark:text-teal-300",
    border: "border-teal-500/35 hover:border-teal-500/45",
    glow: "from-teal-500/16 via-teal-500/6 to-transparent",
    bar: "bg-teal-500",
    pill: "border-teal-500/25 bg-teal-500/12 text-teal-700 dark:text-teal-200",
  },
  cyan: {
    icon: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
    border: "border-cyan-500/35 hover:border-cyan-500/45",
    glow: "from-cyan-500/16 via-cyan-500/6 to-transparent",
    bar: "bg-cyan-500",
    pill: "border-cyan-500/25 bg-cyan-500/12 text-cyan-700 dark:text-cyan-200",
  },
  amber: {
    icon: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
    border: "border-amber-500/35 hover:border-amber-500/45",
    glow: "from-amber-500/16 via-amber-500/6 to-transparent",
    bar: "bg-amber-500",
    pill: "border-amber-500/25 bg-amber-500/12 text-amber-700 dark:text-amber-200",
  },
  blue: {
    icon: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
    border: "border-blue-500/35 hover:border-blue-500/45",
    glow: "from-blue-500/16 via-blue-500/6 to-transparent",
    bar: "bg-blue-500",
    pill: "border-blue-500/25 bg-blue-500/12 text-blue-700 dark:text-blue-200",
  },
  rose: {
    icon: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
    border: "border-rose-500/35 hover:border-rose-500/45",
    glow: "from-rose-500/16 via-rose-500/6 to-transparent",
    bar: "bg-rose-500",
    pill: "border-rose-500/25 bg-rose-500/12 text-rose-700 dark:text-rose-200",
  },
  indigo: {
    icon: "bg-indigo-500/15 text-indigo-700 dark:text-indigo-300",
    border: "border-indigo-500/35 hover:border-indigo-500/45",
    glow: "from-indigo-500/16 via-indigo-500/6 to-transparent",
    bar: "bg-indigo-500",
    pill: "border-indigo-500/25 bg-indigo-500/12 text-indigo-700 dark:text-indigo-200",
  },
}

const statusToneClasses: Record<SkillStatusTone, string> = {
  success: "border-emerald-500/25 bg-emerald-500/12 text-emerald-700 dark:text-emerald-200",
  warning: "border-amber-500/25 bg-amber-500/12 text-amber-700 dark:text-amber-200",
  neutral: "border-border bg-muted text-muted-foreground",
  accent: "border-sky-500/25 bg-sky-500/12 text-sky-700 dark:text-sky-200",
}

const skillIcons: Record<string, LucideIcon> = {
  "feed-harvest": Cable,
  "scholar-tracking": GraduationCap,
  "research-workspace": Telescope,
  "deepcode-studio": Bot,
  "review-reporting": FileSearch,
  "workflow-orchestration": Workflow,
}

function matchesQuery(skill: SkillCardData, query: string) {
  if (!query.trim()) return true
  return [skill.title, skill.headline, skill.description, skill.category, ...skill.outputs, ...skill.sourcePaths]
    .join(" ")
    .toLowerCase()
    .includes(query.trim().toLowerCase())
}

function SkillPipelineCard({
  step,
  active,
  onSelect,
}: {
  step: SkillPipelineStep
  active: boolean
  onSelect: (skillId: string) => void
}) {
  const tone: SkillStatusTone =
    step.status === "Live"
      ? "success"
      : step.status === "Configured"
        ? "accent"
        : step.status === "Preview" || step.status === "Needs setup"
          ? "warning"
          : "neutral"

  return (
    <button
      type="button"
      onClick={() => onSelect(step.skillId)}
      className={cn(
        "flex h-full flex-col rounded-3xl border bg-background/75 p-5 text-left transition-all duration-200",
        "hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg",
        active && "border-primary/30 bg-primary/[0.045] shadow-lg",
      )}
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">{step.step}</span>
        <Badge variant="outline" className={cn("rounded-full", statusToneClasses[tone])}>
          {step.status}
        </Badge>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <h4 className="text-base font-semibold tracking-tight">{step.title}</h4>
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        </div>
        <p className="text-sm leading-6 text-muted-foreground">{step.body}</p>
      </div>
      <div className="mt-5 text-xs uppercase tracking-[0.18em] text-muted-foreground">{step.owner}</div>
    </button>
  )
}

function DetailList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-3xl border border-border/70 bg-background/70 p-5">
      <h4 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">{title}</h4>
      <ul className="mt-3 space-y-3 text-sm leading-6 text-muted-foreground">
        {items.map((item) => (
          <li key={item} className="flex gap-3">
            <span className="mt-2 h-1.5 w-1.5 rounded-full bg-primary" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function SkillsStudio({ data }: SkillsStudioProps) {
  const [query, setQuery] = useState("")
  const [category, setCategory] = useState("All")
  const [status, setStatus] = useState("All")
  const [selectedId, setSelectedId] = useState<string | null>(data.skills[0]?.id ?? null)

  const filteredSkills = useMemo(
    () =>
      data.skills.filter((skill) => {
        const categoryMatch = category === "All" || skill.category === category
        const statusMatch = status === "All" || skill.status.label === status
        return categoryMatch && statusMatch && matchesQuery(skill, query)
      }),
    [category, data.skills, query, status],
  )

  useEffect(() => {
    if (!filteredSkills.length) {
      setSelectedId(null)
      return
    }
    if (!selectedId || !filteredSkills.some((skill) => skill.id === selectedId)) {
      setSelectedId(filteredSkills[0].id)
    }
  }, [filteredSkills, selectedId])

  const selectedSkill = filteredSkills.find((skill) => skill.id === selectedId) ?? null

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(45,212,191,0.14),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(251,191,36,0.12),_transparent_24%),linear-gradient(180deg,rgba(248,250,252,1),rgba(241,245,249,0.92))] dark:bg-[radial-gradient(circle_at_top_left,_rgba(45,212,191,0.16),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(251,191,36,0.12),_transparent_24%),linear-gradient(180deg,rgba(2,6,23,1),rgba(9,9,11,0.98))]">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-8 px-4 py-8 md:px-8 md:py-10 xl:px-10">
        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card className="relative overflow-hidden border-slate-200/80 bg-white/80 shadow-xl shadow-slate-200/40 backdrop-blur dark:border-white/10 dark:bg-white/[0.04] dark:shadow-black/20">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.14),transparent_32%),radial-gradient(circle_at_bottom_right,rgba(14,165,233,0.10),transparent_28%)]" />
            <CardHeader className="relative gap-4">
              <div className="flex flex-wrap items-center gap-3">
                <Badge variant="outline" className="rounded-full border-teal-500/25 bg-teal-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-teal-700 dark:text-teal-200">
                  {data.title}
                </Badge>
                <Badge variant="outline" className="rounded-full px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em]">
                  latest dev architecture
                </Badge>
              </div>
              <div className="max-w-4xl space-y-4">
                <CardTitle className="text-4xl leading-tight tracking-tight md:text-5xl">Skills as a control surface for the research stack.</CardTitle>
                <CardDescription className="max-w-3xl text-base leading-7 text-slate-600 dark:text-slate-300">
                  {data.subtitle}
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent className="relative flex flex-wrap gap-3">
              {data.workspaceFacts.map((fact) => (
                <div key={fact.label} className="rounded-2xl border border-slate-200/70 bg-white/80 px-4 py-3 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">{fact.label}</p>
                  <p className="mt-1 text-sm font-medium text-foreground">{fact.value}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-2">
            {data.summary.map((item) => (
              <Card key={item.label} className="border-slate-200/80 bg-white/85 shadow-lg shadow-slate-200/30 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
                <CardHeader className="gap-2 pb-2">
                  <CardDescription className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">{item.label}</CardDescription>
                  <CardTitle className="text-3xl tracking-tight">{item.value}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-6 text-muted-foreground">{item.caption}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px] xl:grid-cols-[minmax(0,1fr)_260px]">
          <Card className="border-slate-200/80 bg-white/85 shadow-lg shadow-slate-200/30 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
            <CardContent className="pt-6">
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by skill, source path, output, or category" className="h-11 rounded-2xl bg-background/70 pl-10" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200/80 bg-white/85 shadow-lg shadow-slate-200/30 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
            <CardContent className="flex items-center gap-3 pt-6">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Select value={status} onValueChange={setStatus}>
                <SelectTrigger className="h-11 rounded-2xl bg-background/70">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  {data.statusOptions.map((option) => (
                    <SelectItem key={option} value={option}>{option}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>
        </section>
        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card className="border-slate-200/80 bg-white/85 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
            <CardHeader className="gap-4">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
                <div className="space-y-2">
                  <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">skills catalog</Badge>
                  <CardTitle className="text-2xl tracking-tight">Auto-discovered capability surfaces</CardTitle>
                  <CardDescription className="max-w-2xl text-sm leading-6">Mapped from the current `web/src` routes and `src/paperbot` services, not a detached prototype shell.</CardDescription>
                </div>
                <div className="flex flex-wrap gap-2">
                  {data.categories.map((option) => (
                    <Button key={option} type="button" variant={option === category ? "default" : "outline"} className="rounded-full" onClick={() => setCategory(option)}>
                      {option}
                    </Button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {filteredSkills.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-slate-300/80 bg-background/60 px-6 py-16 text-center dark:border-white/10">
                  <Search className="mx-auto h-5 w-5 text-muted-foreground" />
                  <h3 className="mt-4 text-lg font-semibold tracking-tight">No skills match this filter</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">Try a broader search or reset the current category/status filters.</p>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {filteredSkills.map((skill) => {
                    const theme = toneClasses[skill.tone]
                    const Icon = skillIcons[skill.id] ?? Sparkles
                    const selected = selectedSkill?.id === skill.id

                    return (
                      <button
                        key={skill.id}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => setSelectedId(skill.id)}
                        className={cn(
                          "group relative overflow-hidden rounded-[28px] border bg-background/80 p-5 text-left shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg",
                          theme.border,
                          selected && "bg-background shadow-lg",
                        )}
                      >
                        <div className={cn("pointer-events-none absolute inset-0 bg-gradient-to-br opacity-0 transition-opacity duration-200 group-hover:opacity-100", theme.glow, selected && "opacity-100")} />
                        <div className="relative flex h-full flex-col gap-5">
                          <div className="flex items-start justify-between gap-3">
                            <div className="space-y-3">
                              <div className={cn("flex h-11 w-11 items-center justify-center rounded-2xl", theme.icon)}>
                                <Icon className="h-5 w-5" />
                              </div>
                              <div>
                                <div className="flex flex-wrap items-center gap-2">
                                  <h3 className="text-lg font-semibold tracking-tight">{skill.title}</h3>
                                  <Badge variant="outline" className={cn("rounded-full", statusToneClasses[skill.status.tone])}>{skill.status.label}</Badge>
                                </div>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">{skill.headline}</p>
                              </div>
                            </div>
                            <Badge variant="outline" className={cn("rounded-full", theme.pill)}>{skill.category}</Badge>
                          </div>

                          <div className="grid grid-cols-3 gap-3">
                            {[
                              { label: "readiness", value: `${skill.readiness}%` },
                              { label: "footprint", value: String(skill.footprint) },
                              { label: "signal", value: skill.signalValue },
                            ].map((metric) => (
                              <div key={metric.label} className="rounded-2xl border border-border/70 bg-background/80 p-3">
                                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">{metric.label}</p>
                                <p className="mt-2 text-xl font-semibold tracking-tight">{metric.value}</p>
                              </div>
                            ))}
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                              <span>{skill.signalLabel}</span>
                              <span>{skill.updatedAt}</span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-muted">
                              <div className={cn("h-full rounded-full transition-all duration-300", theme.bar)} style={{ width: `${skill.readiness}%` }} />
                            </div>
                            <p className="text-xs leading-5 text-muted-foreground">{skill.signalCaption}</p>
                          </div>

                          <div className="flex flex-wrap gap-2">
                            {skill.tags.map((tag) => (
                              <Badge key={tag} variant="outline" className="rounded-full bg-background/75">{tag}</Badge>
                            ))}
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="xl:sticky xl:top-6 xl:self-start">
            <Card className="border-slate-200/80 bg-white/88 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.05]">
              <CardHeader className="gap-4">
                <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">skill detail</Badge>
                {selectedSkill ? (
                  <>
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center gap-3">
                        <div className={cn("flex h-12 w-12 items-center justify-center rounded-2xl", toneClasses[selectedSkill.tone].icon)}>
                          {(() => {
                            const Icon = skillIcons[selectedSkill.id] ?? Sparkles
                            return <Icon className="h-5 w-5" />
                          })()}
                        </div>
                        <div className="space-y-1">
                          <CardTitle className="text-2xl tracking-tight">{selectedSkill.title}</CardTitle>
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className={cn("rounded-full", statusToneClasses[selectedSkill.status.tone])}>{selectedSkill.status.label}</Badge>
                            <Badge variant="outline" className={cn("rounded-full", toneClasses[selectedSkill.tone].pill)}>{selectedSkill.category}</Badge>
                          </div>
                        </div>
                      </div>
                      <CardDescription className="text-sm leading-6">{selectedSkill.description}</CardDescription>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1 2xl:grid-cols-3">
                      {[
                        { label: "readiness", value: `${selectedSkill.readiness}%` },
                        { label: "signal", value: selectedSkill.signalValue },
                        { label: "updated", value: selectedSkill.updatedAt },
                      ].map((item) => (
                        <div key={item.label} className="rounded-2xl border border-border/70 bg-background/70 p-4">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">{item.label}</p>
                          <p className="mt-2 text-2xl font-semibold tracking-tight">{item.value}</p>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <CardTitle className="text-2xl tracking-tight">Select a skill</CardTitle>
                    <CardDescription className="text-sm leading-6">Pick a visible card to inspect prerequisites, outputs, and mapped source paths.</CardDescription>
                  </>
                )}
              </CardHeader>
              {selectedSkill ? (
                <CardContent className="space-y-4">
                  <div className="rounded-3xl border border-border/70 bg-background/70 p-5">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <h4 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">Runtime signal</h4>
                      <Badge variant="outline" className={cn("rounded-full", toneClasses[selectedSkill.tone].pill)}>{selectedSkill.signalLabel}</Badge>
                    </div>
                    <p className="text-sm leading-6 text-muted-foreground">{selectedSkill.signalCaption}</p>
                  </div>
                  <DetailList title="Prerequisites" items={selectedSkill.prerequisites} />
                  <DetailList title="Outputs" items={selectedSkill.outputs} />
                  <div className="rounded-3xl border border-border/70 bg-background/70 p-5">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <h4 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">Source paths</h4>
                      <Badge variant="outline" className="rounded-full">{selectedSkill.sourcePaths.length} mapped</Badge>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {selectedSkill.sourcePaths.map((sourcePath) => (
                        <code key={sourcePath} className="rounded-2xl border border-border/70 bg-background px-3 py-2 text-xs leading-5 text-foreground">{sourcePath}</code>
                      ))}
                    </div>
                  </div>
                </CardContent>
              ) : null}
            </Card>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <Card className="border-slate-200/80 bg-white/85 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
            <CardHeader className="gap-3">
              <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">automation lane</Badge>
              <CardTitle className="text-2xl tracking-tight">Pipeline states</CardTitle>
              <CardDescription className="text-sm leading-6">Click any stage to sync the detail panel with the underlying skill surface.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {data.pipeline.map((step) => (
                  <SkillPipelineCard key={step.step} step={step} active={selectedSkill?.id === step.skillId} onSelect={setSelectedId} />
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6">
            <Card className="border-slate-200/80 bg-white/85 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
              <CardHeader className="gap-3">
                <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">runtime signals</Badge>
                <CardTitle className="text-2xl tracking-tight">Environment hooks</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3">
                {data.signals.map((signal) => (
                  <div key={signal.label} className="flex items-center justify-between gap-4 rounded-3xl border border-border/70 bg-background/70 px-4 py-4">
                    <div>
                      <p className="text-sm font-semibold tracking-tight">{signal.label}</p>
                      <p className="text-xs leading-5 text-muted-foreground">Current runtime hook for this workspace.</p>
                    </div>
                    <Badge variant="outline" className={cn("rounded-full", statusToneClasses[signal.tone])}>{signal.value}</Badge>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="border-slate-200/80 bg-white/85 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.04]">
              <CardHeader className="gap-3">
                <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">architecture fit</Badge>
                <CardTitle className="text-2xl tracking-tight">Why this now matches dev</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
                <div className="flex gap-3 rounded-3xl border border-border/70 bg-background/70 px-4 py-4">
                  <GitFork className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <p>Source mapping now targets `web/src` routes and `src/paperbot` services instead of the deleted `agent_module` proof of concept.</p>
                </div>
                <div className="flex gap-3 rounded-3xl border border-border/70 bg-background/70 px-4 py-4">
                  <Blocks className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <p>The UI runs inside the existing Next.js app router, uses current shadcn primitives, and plugs directly into the sidebar navigation.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <section>
          <Card className="border-slate-200/80 bg-white/88 shadow-xl shadow-slate-200/35 backdrop-blur dark:border-white/10 dark:bg-white/[0.05]">
            <CardHeader className="gap-3">
              <Badge variant="outline" className="w-fit rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">recent outputs</Badge>
              <CardTitle className="text-2xl tracking-tight">Generated artifacts</CardTitle>
              <CardDescription className="text-sm leading-6">Live files detected from the repository workspace, including generated reports and experiment outputs.</CardDescription>
            </CardHeader>
            <CardContent>
              {data.recentOutputs.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {data.recentOutputs.map((item) => (
                    <div key={`${item.path}:${item.updatedAt}`} className="rounded-[28px] border border-border/70 bg-background/75 p-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-lg font-semibold tracking-tight">{item.name}</p>
                          <p className="mt-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">artifact</p>
                        </div>
                        <Badge variant="outline" className="rounded-full">{item.updatedAt}</Badge>
                      </div>
                      <code className="mt-4 block rounded-2xl border border-border/70 bg-background px-3 py-3 text-xs leading-5 text-foreground">{item.path}</code>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-3xl border border-dashed border-slate-300/80 bg-background/60 px-6 py-14 text-center dark:border-white/10">
                  <Sparkles className="mx-auto h-6 w-6 text-muted-foreground" />
                  <h3 className="mt-4 text-lg font-semibold tracking-tight">No generated artifacts detected</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">Run a harvest, research, review, or studio flow and this section will populate automatically.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  )
}
