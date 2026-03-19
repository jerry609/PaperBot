"use client"

import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { useEffect, useMemo, useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  ArrowLeft,
  ArrowUpRight,
  CheckCircle2,
  FolderOpen,
  GitBranch,
  Loader2,
  Package2,
  RefreshCcw,
  Search,
  Sparkles,
  Wrench,
} from "lucide-react"

import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import { useStudioSkillDetail, useStudioSkillsCatalog } from "@/hooks/useStudioSkillsCatalog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import {
  getStudioSkillPaths,
  skillNeedsGeneratedContext,
  skillNeedsWorkspace,
} from "@/lib/studio-skills"
import { cn } from "@/lib/utils"
import type { StudioInstalledRepoInfo, StudioMarketplaceRepoInfo } from "@/lib/studio-skill-catalog"
import { useStudioStore } from "@/lib/store/studio-store"
import type { StudioSkillInfo } from "@/lib/studio-runtime"

type SkillsViewFilter = "all" | "project" | "installed"

type StudioCwdPayload = {
  cwd?: string | null
  actual_cwd?: string | null
  home?: string | null
}

const CONTEXT_MODULE_LABELS: Record<string, string> = {
  paper_brief: "Paper brief",
  literature: "Literature",
  environment: "Environment",
  spec: "Spec",
  roadmap: "Roadmap",
  success_criteria: "Success criteria",
  workspace: "Workspace",
}

function scopeBadgeClassName(scope: string): string {
  if (scope === "installed") return "border-blue-200 bg-blue-50 text-blue-700"
  return "border-slate-200 bg-[#f7f8f4] text-slate-700"
}

function formatContextModule(value: string): string {
  return CONTEXT_MODULE_LABELS[value] ?? value.replace(/[_-]+/g, " ")
}

function shortCommit(value: string | null | undefined): string | null {
  if (!value) return null
  return value.slice(0, 8)
}

function normalizeSearchValue(value: string): string {
  return value.trim().toLowerCase()
}

function formatWorkspaceSummary(value: string | null | undefined): string {
  const cleaned = value?.trim()
  if (!cleaned) return "Not set"
  return cleaned
}

function buildSkillsQuery(params: { toString(): string }, updates: Record<string, string | null | undefined>): string {
  const next = new URLSearchParams(params.toString())
  for (const [key, value] of Object.entries(updates)) {
    const cleaned = typeof value === "string" ? value.trim() : value
    if (!cleaned) {
      next.delete(key)
    } else {
      next.set(key, cleaned)
    }
  }
  return next.toString()
}

function FlowStepCard({
  step,
  title,
  description,
  status,
  children,
}: {
  step: number
  title: string
  description: string
  status: "ready" | "pending" | "active"
  children?: React.ReactNode
}) {
  const tone =
    status === "ready"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : status === "active"
        ? "border-slate-300 bg-[#eef1ea] text-slate-900"
        : "border-slate-200 bg-white text-slate-500"

  const label = status === "ready" ? "Ready" : status === "active" ? "Next" : "Waiting"

  return (
    <div className="rounded-[22px] border border-slate-200 bg-[#fbfbf8] p-4">
      <div className="flex items-start gap-3">
        <div className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-[12px] font-semibold", tone)}>
          {step}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-[14px] font-semibold text-slate-950">{title}</p>
            <Badge className={cn("h-6 border px-2.5 text-[10px]", tone)}>{label}</Badge>
          </div>
          <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>
          {children ? <div className="mt-3">{children}</div> : null}
        </div>
      </div>
    </div>
  )
}

function PageShell({
  eyebrow,
  title,
  description,
  actions,
  children,
}: {
  eyebrow: string
  title: string
  description: string
  actions?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-[#f5f6f1]">
      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-6 px-4 py-6 lg:px-8">
        <div className="rounded-[28px] border border-slate-200 bg-white/95 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.06)]">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                {eyebrow}
              </p>
              <h1 className="mt-2 text-[34px] font-semibold tracking-[-0.04em] text-slate-950">
                {title}
              </h1>
              <p className="mt-3 text-[15px] leading-7 text-slate-600">{description}</p>
            </div>
            {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
          </div>
        </div>

        {children}
      </div>
    </div>
  )
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-[22px] border border-slate-200 bg-white p-4 shadow-[0_14px_30px_rgba(15,23,42,0.04)]">
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <div className="mt-2 text-[28px] font-semibold tracking-[-0.03em] text-slate-950">{value}</div>
      <p className="mt-1 text-[13px] leading-6 text-slate-500">{hint}</p>
    </div>
  )
}

function Section({
  title,
  description,
  count,
  children,
}: {
  title: string
  description: string
  count?: number
  children: React.ReactNode
}) {
  return (
    <section className="rounded-[26px] border border-slate-200 bg-white/95 p-5 shadow-[0_18px_36px_rgba(15,23,42,0.04)]">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-[22px] font-semibold tracking-[-0.03em] text-slate-950">{title}</h2>
          <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>
        </div>
        {typeof count === "number" ? (
          <Badge variant="outline" className="h-6 border-slate-200 bg-[#f7f8f4] px-2.5 text-[10px] text-slate-600">
            {count}
          </Badge>
        ) : null}
      </div>
      {children}
    </section>
  )
}

function SkillCard({
  skill,
  selected,
  attached,
  onSelect,
}: {
  skill: StudioSkillInfo
  selected: boolean
  attached: boolean
  onSelect: (skillKey: string) => void
}) {
  const primaryPath = getStudioSkillPaths(skill)[0] ?? "Path unavailable"

  return (
    <div
      className={cn(
        "rounded-[24px] border p-4 transition-colors",
        selected
          ? "border-slate-300 bg-[#eef1ea]"
          : "border-slate-200 bg-[#fbfbf8]",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[16px] font-semibold text-slate-950">{skill.title}</h3>
            <span className={cn("rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]", scopeBadgeClassName(skill.scope))}>
              {skill.scope}
            </span>
            {attached ? (
              <Badge className="h-6 border-emerald-200 bg-emerald-50 px-2.5 text-[10px] text-emerald-700">
                attached
              </Badge>
            ) : null}
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            {skill.description || "No description provided for this Studio skill."}
          </p>
        </div>
        <Badge variant="outline" className="h-6 border-slate-200 bg-white px-2.5 text-[10px] text-slate-600">
          {skill.slashCommand}
        </Badge>
      </div>

      <div className="mt-3 rounded-[18px] border border-slate-200 bg-white/80 px-3 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Source path</p>
        <p className="mt-1 break-all text-[12px] text-slate-600">{primaryPath}</p>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {skill.repoLabel ? (
          <Badge variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
            {skill.repoLabel}
          </Badge>
        ) : null}
        {skill.contextModules.map((module) => (
          <Badge
            key={`${skill.key}:${module}`}
            variant="outline"
            className="h-6 border-slate-200 bg-white text-[10px] text-slate-600"
          >
            {formatContextModule(module)}
          </Badge>
        ))}
        {skill.tools.slice(0, 3).map((tool) => (
          <Badge
            key={`${skill.key}:${tool}`}
            variant="outline"
            className="h-6 border-slate-200 bg-[#f0f3ea] text-[10px] text-slate-700"
          >
            {tool}
          </Badge>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button
          type="button"
          size="sm"
          className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
          onClick={() => onSelect(skill.key)}
        >
          {selected ? "Selected" : "Use this skill"}
        </Button>
        <Button asChild variant="outline" size="sm" className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700">
          <Link href={`/skills/${encodeURIComponent(skill.key)}`}>
            Detail
          </Link>
        </Button>
        <Button asChild variant="ghost" size="sm" className="h-8 rounded-full px-3 text-[11px] text-slate-600 hover:bg-white">
          <Link href={`/skills/${encodeURIComponent(skill.key)}/setup`}>
            Open setup
          </Link>
        </Button>
      </div>
    </div>
  )
}

function RepoCard({
  repo,
  onInstall,
  onUpdate,
  pending,
}: {
  repo: StudioMarketplaceRepoInfo | StudioInstalledRepoInfo
  onInstall?: (repoUrl: string, repoRef?: string | null) => void
  onUpdate?: (repoSlug: string) => void
  pending: boolean
}) {
  const isInstalledRepo = "skills" in repo
  const installPath = repo.installPath
  const count = isInstalledRepo ? repo.skills.length : repo.installedSkillCount

  return (
    <div className="rounded-[22px] border border-slate-200 bg-[#fbfbf8] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[16px] font-semibold text-slate-950">{repo.title}</h3>
            <Badge variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
              {count} skill{count === 1 ? "" : "s"}
            </Badge>
            {"updateAvailable" in repo && repo.updateAvailable ? (
              <Badge className="h-6 border-amber-200 bg-amber-50 text-[10px] text-amber-700">
                update ready
              </Badge>
            ) : null}
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">{repo.description}</p>
          <p className="mt-2 break-all text-[12px] text-slate-500">{repo.repoUrl}</p>
          {installPath ? <p className="mt-1 text-[12px] text-slate-500">Installed at {installPath}</p> : null}
          {isInstalledRepo && repo.lastKnownCommit ? (
            <p className="mt-1 text-[12px] text-slate-500">
              Commit {shortCommit(repo.lastKnownCommit)}
              {repo.remoteCommit && repo.remoteCommit !== repo.lastKnownCommit
                ? ` -> ${shortCommit(repo.remoteCommit)}`
                : ""}
            </p>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {!isInstalledRepo && onInstall ? (
          <Button
            type="button"
            size="sm"
            className="h-8 rounded-full bg-slate-900 px-3 text-[11px] text-white hover:bg-slate-800"
            disabled={pending}
            onClick={() => onInstall(repo.repoUrl, repo.repoRef)}
          >
            {pending ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <ArrowUpRight className="mr-1.5 h-3.5 w-3.5" />}
            Install repo
          </Button>
        ) : null}
        {isInstalledRepo && onUpdate ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700"
            disabled={pending}
            onClick={() => onUpdate(repo.slug)}
          >
            {pending ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <RefreshCcw className="mr-1.5 h-3.5 w-3.5" />}
            Refresh repo
          </Button>
        ) : null}
      </div>
    </div>
  )
}

export function SkillsDirectoryPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { catalog, loading, refreshing, refresh } = useStudioSkillsCatalog()
  const { generate, status: generationStatus } = useContextPackGeneration()
  const {
    papers,
    selectedPaperId: activePaperId,
    attachedSkill,
    contextPack,
    loadPapers,
    selectPaper,
    updatePaper,
    setAttachedSkill,
  } = useStudioStore()
  const [filter, setFilter] = useState<SkillsViewFilter>("all")
  const [skillQuery, setSkillQuery] = useState("")
  const [selectedSkillKey, setSelectedSkillKey] = useState("")
  const [selectedPaperId, setSelectedPaperId] = useState("")
  const [workspacePath, setWorkspacePath] = useState("")
  const [cwd, setCwd] = useState<string | null>(null)
  const [installUrl, setInstallUrl] = useState("")
  const [installRef, setInstallRef] = useState("")
  const [pendingRepoAction, setPendingRepoAction] = useState<string | null>(null)
  const [repoError, setRepoError] = useState<string | null>(null)
  const [flowError, setFlowError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const selectedSkillDetailState = useStudioSkillDetail(selectedSkillKey || null)
  const selectedSkillDetail = selectedSkillDetailState.detail
  const selectedSkillLoading = selectedSkillDetailState.loading

  const queryPaperId = searchParams.get("paperId") || searchParams.get("paper_id")
  const querySkillKey = searchParams.get("skill") || searchParams.get("skillKey")

  useEffect(() => {
    loadPapers()
  }, [loadPapers])

  useEffect(() => {
    let active = true
    void fetch("/api/studio/cwd", { cache: "no-store" })
      .then((response) => response.json() as Promise<StudioCwdPayload>)
      .then((payload) => {
        if (!active) return
        setCwd(payload.actual_cwd || payload.cwd || payload.home || null)
      })
      .catch(() => {
        if (!active) return
        setCwd(null)
      })
    return () => {
      active = false
    }
  }, [])

  const skillInventory = useMemo(
    () =>
      [...catalog.projectSkills, ...catalog.installedSkills].sort((left, right) => {
        if (left.scope !== right.scope) {
          return left.scope === "project" ? -1 : 1
        }
        return left.title.localeCompare(right.title)
      }),
    [catalog.installedSkills, catalog.projectSkills],
  )

  useEffect(() => {
    if (skillInventory.length === 0) {
      if (selectedSkillKey) setSelectedSkillKey("")
      return
    }

    const requestedKey = querySkillKey?.trim()
    if (requestedKey && skillInventory.some((skill) => skill.key === requestedKey)) {
      if (selectedSkillKey !== requestedKey) setSelectedSkillKey(requestedKey)
      return
    }

    const attachedKey = attachedSkill?.key?.trim()
    if (attachedKey && skillInventory.some((skill) => skill.key === attachedKey)) {
      if (selectedSkillKey !== attachedKey) setSelectedSkillKey(attachedKey)
      return
    }

    if (!selectedSkillKey || !skillInventory.some((skill) => skill.key === selectedSkillKey)) {
      setSelectedSkillKey(skillInventory[0].key)
    }
  }, [attachedSkill?.key, querySkillKey, selectedSkillKey, skillInventory])

  useEffect(() => {
    if (papers.length === 0) {
      if (selectedPaperId) setSelectedPaperId("")
      return
    }

    const requestedPaperId = queryPaperId?.trim()
    if (requestedPaperId && papers.some((paper) => paper.id === requestedPaperId)) {
      if (selectedPaperId !== requestedPaperId) setSelectedPaperId(requestedPaperId)
      return
    }

    if (activePaperId && papers.some((paper) => paper.id === activePaperId)) {
      if (selectedPaperId !== activePaperId) setSelectedPaperId(activePaperId)
      return
    }

    if (!selectedPaperId || !papers.some((paper) => paper.id === selectedPaperId)) {
      setSelectedPaperId(papers[0].id)
    }
  }, [activePaperId, papers, queryPaperId, selectedPaperId])

  const selectedPaper = useMemo(
    () => papers.find((paper) => paper.id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  )

  useEffect(() => {
    if (!selectedPaperId) return
    selectPaper(selectedPaperId)
  }, [selectPaper, selectedPaperId])

  useEffect(() => {
    const nextWorkspace = selectedPaper?.outputDir?.trim() || cwd || ""
    setWorkspacePath(nextWorkspace)
  }, [cwd, selectedPaper?.id, selectedPaper?.outputDir])

  const selectedSkill = useMemo(
    () =>
      selectedSkillDetail.skill ??
      skillInventory.find((skill) => skill.key === selectedSkillKey) ??
      null,
    [selectedSkillDetail.skill, selectedSkillKey, skillInventory],
  )

  const selectedContextModules = useMemo(() => {
    if (selectedSkillDetail.contextModules.length > 0) return selectedSkillDetail.contextModules
    return selectedSkill?.contextModules ?? []
  }, [selectedSkill?.contextModules, selectedSkillDetail.contextModules])

  const requiresGeneratedContext = skillNeedsGeneratedContext(selectedContextModules)
  const requiresWorkspace = skillNeedsWorkspace(selectedContextModules, {
    requiresWorkspaceHint: selectedSkillDetail.requiresWorkspace,
  })
  const contextReady = Boolean(contextPack?.context_pack_id || selectedPaper?.contextPackId)
  const selectedSkillAttached = Boolean(selectedSkill && attachedSkill?.key === selectedSkill.key)

  const filteredSkills = useMemo(() => {
    const query = normalizeSearchValue(skillQuery)
    return skillInventory.filter((skill) => {
      if (filter === "project" && skill.scope !== "project") return false
      if (filter === "installed" && skill.scope !== "installed") return false
      if (!query) return true

      const haystack = [
        skill.title,
        skill.description,
        skill.slashCommand,
        skill.scope,
        skill.repoLabel ?? "",
        ...skill.contextModules,
        ...skill.tools,
      ]
        .join(" ")
        .toLowerCase()

      return haystack.includes(query)
    })
  }, [filter, skillInventory, skillQuery])

  const syncDirectoryQuery = (updates: Record<string, string | null | undefined>) => {
    const query = buildSkillsQuery(searchParams, updates)
    router.replace(query ? `/skills?${query}` : "/skills", { scroll: false })
  }

  const installRepo = async (repoUrl: string, repoRef?: string | null) => {
    const normalizedUrl = repoUrl.trim()
    if (!normalizedUrl) {
      setRepoError("Enter a Git repository URL first.")
      return
    }

    setPendingRepoAction(normalizedUrl)
    setRepoError(null)
    setNotice(null)
    try {
      const response = await fetch("/api/studio/skills/install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_url: normalizedUrl,
          repo_ref: repoRef?.trim() || undefined,
        }),
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Failed to install repository (${response.status})`)
      }
      setInstallUrl("")
      setInstallRef("")
      await refresh()
      setNotice("Repository installed. Pick one of the discovered skills and continue on the right.")
    } catch (installError) {
      setRepoError(installError instanceof Error ? installError.message : "Failed to install repository")
    } finally {
      setPendingRepoAction(null)
    }
  }

  const updateRepo = async (repoSlug: string) => {
    setPendingRepoAction(repoSlug)
    setRepoError(null)
    setNotice(null)
    try {
      const response = await fetch(`/api/studio/skills/repos/${encodeURIComponent(repoSlug)}/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Failed to refresh repository (${response.status})`)
      }
      await refresh()
      setNotice("Repository metadata refreshed.")
    } catch (updateError) {
      setRepoError(updateError instanceof Error ? updateError.message : "Failed to refresh repository")
    } finally {
      setPendingRepoAction(null)
    }
  }

  const handleSelectSkill = (skillKey: string) => {
    setFlowError(null)
    setNotice(null)
    setSelectedSkillKey(skillKey)
    syncDirectoryQuery({
      skill: skillKey,
      paperId: selectedPaperId || queryPaperId || undefined,
    })
  }

  const handlePaperChange = (paperId: string) => {
    setFlowError(null)
    setSelectedPaperId(paperId)
    syncDirectoryQuery({
      skill: selectedSkillKey || undefined,
      paperId,
    })
  }

  const handleGenerateContext = async () => {
    if (!selectedPaper) {
      setFlowError("Choose a paper before generating context.")
      return
    }

    setFlowError(null)
    setNotice(null)
    selectPaper(selectedPaper.id)
    await generate({
      paperId: selectedPaper.id,
      title: selectedPaper.title,
      abstract: selectedPaper.abstract,
    })
  }

  const handleAttachAndContinue = () => {
    if (!selectedSkill) {
      setFlowError("Choose a skill before continuing.")
      return
    }
    if (!selectedPaper) {
      setFlowError("Choose a paper before continuing.")
      return
    }
    if (requiresWorkspace && !workspacePath.trim()) {
      setFlowError("Set a workspace before continuing.")
      return
    }

    setFlowError(null)
    selectPaper(selectedPaper.id)
    if (workspacePath.trim()) {
      updatePaper(selectedPaper.id, { outputDir: workspacePath.trim() })
    }
    setAttachedSkill({
      ...selectedSkill,
      contextModules: selectedContextModules,
    })
    router.push(`/studio?paperId=${encodeURIComponent(selectedPaper.id)}`)
  }

  const handlePrimaryAction = async () => {
    if (!selectedSkill) {
      setFlowError("Choose a skill first.")
      return
    }
    if (!selectedPaper) {
      setFlowError("Choose a paper before continuing.")
      return
    }
    if (requiresWorkspace && !workspacePath.trim()) {
      setFlowError("Set a workspace before continuing.")
      return
    }
    if (requiresGeneratedContext && !contextReady) {
      await handleGenerateContext()
      return
    }
    handleAttachAndContinue()
  }

  const primaryActionLabel = !selectedSkill
    ? "Choose a skill"
    : !selectedPaper
      ? "Choose a paper"
      : requiresWorkspace && !workspacePath.trim()
        ? "Set workspace"
        : requiresGeneratedContext && !contextReady
          ? generationStatus === "generating"
            ? "Generating context..."
            : "Generate context"
          : selectedSkillAttached
            ? "Continue in Studio"
            : "Attach and continue"

  return (
    <PageShell
      eyebrow="Studio Skills"
      title="Pick a skill, then launch Claude Code in one pass"
      description="Use this page as the single handoff surface: choose the skill, bind it to a paper, confirm the workspace, generate paper context only if the skill actually needs it, then continue into Studio chat."
      actions={
        <>
          <Button
            type="button"
            variant="outline"
            className="h-10 rounded-full border-slate-200 bg-white px-4 text-sm text-slate-700"
            onClick={() => void refresh(true)}
          >
            {refreshing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCcw className="mr-2 h-4 w-4" />}
            Refresh
          </Button>
          <Button
            type="button"
            className="h-10 rounded-full bg-slate-900 px-4 text-sm text-white hover:bg-slate-800"
            onClick={() => {
              const query = buildSkillsQuery(searchParams, {
                paperId: selectedPaperId || queryPaperId || undefined,
              })
              router.push(query ? `/studio?${query}` : "/studio")
            }}
          >
            Open Studio
          </Button>
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-4">
        <MetricCard
          label="Ready Skills"
          value={String(skillInventory.length)}
          hint="Project and installed skills available for immediate launch."
        />
        <MetricCard
          label="Papers"
          value={String(papers.length)}
          hint="Studio papers available for this launch flow."
        />
        <MetricCard
          label="Context"
          value={contextReady ? "Ready" : requiresGeneratedContext ? "Needed" : "Optional"}
          hint={
            requiresGeneratedContext
              ? "The selected skill expects extracted paper context."
              : "The selected skill can launch without a generated context pack."
          }
        />
        <MetricCard
          label="Attached"
          value={selectedSkillAttached ? "Yes" : "No"}
          hint={selectedSkill ? `${selectedSkill.title} ${selectedSkillAttached ? "is" : "is not"} attached to the current paper.` : "Choose a skill first."}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.12fr)_400px]">
        <div className="space-y-4">
          <Section
            title="Choose a skill"
            description="This is the primary entry. Select the skill once here, then finish the paper, workspace, and context handoff in the launch rail."
            count={filteredSkills.length}
          >
            <div className="space-y-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="relative max-w-xl flex-1">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <Input
                    value={skillQuery}
                    onChange={(event) => setSkillQuery(event.target.value)}
                    placeholder="Search skills, slash commands, tools, or context modules"
                    className="h-11 rounded-2xl border-slate-200 bg-white pl-11 text-sm text-slate-800"
                  />
                </div>
                <div className="flex flex-wrap gap-2">
                  {([
                    ["all", "All skills"],
                    ["project", "Project"],
                    ["installed", "Installed"],
                  ] as const).map(([value, label]) => (
                    <Button
                      key={value}
                      type="button"
                      variant="ghost"
                      className={cn(
                        "h-9 rounded-full border px-4 text-sm",
                        filter === value
                          ? "border-slate-300 bg-[#eef1ea] text-slate-900"
                          : "border-slate-200 bg-white text-slate-600 hover:bg-[#f7f8f4]",
                      )}
                      onClick={() => setFilter(value)}
                    >
                      {label}
                    </Button>
                  ))}
                </div>
              </div>

              {loading ? (
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading launch-ready skills...
                </div>
              ) : filteredSkills.length === 0 ? (
                <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
                  No skills matched the current filter. Try clearing the search, switching source filters, or install a Git skill repo below.
                </div>
              ) : (
                <div className="grid gap-4 xl:grid-cols-2">
                  {filteredSkills.map((skill) => (
                    <SkillCard
                      key={skill.key}
                      skill={skill}
                      selected={selectedSkillKey === skill.key}
                      attached={attachedSkill?.key === skill.key}
                      onSelect={handleSelectSkill}
                    />
                  ))}
                </div>
              )}
            </div>
          </Section>

          <Section
            title="Install more skills"
            description="If the skill you want is not already available, install the repo here. Once it lands in the catalog, the same launch flow on the right will pick it up."
          >
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_240px_180px]">
              <Input
                value={installUrl}
                onChange={(event) => setInstallUrl(event.target.value)}
                placeholder="https://github.com/org/skill-pack.git"
                className="h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-800"
              />
              <Input
                value={installRef}
                onChange={(event) => setInstallRef(event.target.value)}
                placeholder="branch, tag, or commit"
                className="h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-800"
              />
              <Button
                type="button"
                className="h-11 rounded-2xl bg-slate-900 px-4 text-sm text-white hover:bg-slate-800"
                disabled={pendingRepoAction === installUrl.trim()}
                onClick={() => void installRepo(installUrl, installRef)}
              >
                {pendingRepoAction === installUrl.trim() ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <GitBranch className="mr-2 h-4 w-4" />
                )}
                Install repo
              </Button>
            </div>
            {notice ? <p className="mt-3 text-sm text-emerald-700">{notice}</p> : null}
            {repoError ? <p className="mt-3 text-sm text-rose-600">{repoError}</p> : null}
          </Section>

          <Section
            title="Marketplace repos"
            description="Curated Git repos stay secondary here. Install one, then use the same launch flow without leaving this page."
            count={catalog.marketplaceRepos.length}
          >
            {catalog.marketplaceRepos.length === 0 ? (
              <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
                No curated marketplace list is configured yet. Paste a Git repo above to install directly.
              </div>
            ) : (
              <div className="grid gap-4 xl:grid-cols-2">
                {catalog.marketplaceRepos.map((repo) => (
                  <RepoCard
                    key={repo.slug}
                    repo={repo}
                    onInstall={(repoUrl, repoRef) => void installRepo(repoUrl, repoRef)}
                    pending={pendingRepoAction === repo.repoUrl || pendingRepoAction === repo.slug}
                  />
                ))}
              </div>
            )}
          </Section>

          {catalog.updates.length > 0 ? (
            <Section
              title="Updates"
              description="Refresh Git-backed skill repos here when their remote head changes."
              count={catalog.updates.length}
            >
              <div className="grid gap-4 xl:grid-cols-2">
                {catalog.updates.map((repo) => (
                  <RepoCard
                    key={repo.slug}
                    repo={repo}
                    onUpdate={(repoSlug) => void updateRepo(repoSlug)}
                    pending={pendingRepoAction === repo.slug}
                  />
                ))}
              </div>
            </Section>
          ) : null}
        </div>

        <div className="xl:sticky xl:top-6 xl:self-start">
          <Section
            title="Launch flow"
            description="Finish setup here. The primary action always advances the next required step instead of sending you through separate pages."
          >
            {!selectedSkill ? (
              <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
                Choose a skill from the left to start the launch flow.
              </div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-[24px] border border-slate-200 bg-[linear-gradient(180deg,#fafaf7_0%,#eef1ea_100%)] p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        Selected skill
                      </p>
                      <h3 className="mt-1 text-[20px] font-semibold tracking-[-0.03em] text-slate-950">
                        {selectedSkill.title}
                      </h3>
                      <p className="mt-2 text-sm leading-6 text-slate-600">
                        {selectedSkill.description || "No description provided for this skill."}
                      </p>
                    </div>
                    <Badge variant="outline" className="h-6 border-slate-200 bg-white px-2.5 text-[10px] text-slate-600">
                      {selectedSkill.slashCommand}
                    </Badge>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    <Badge className={cn("h-6 px-2.5 text-[10px]", scopeBadgeClassName(selectedSkill.scope))}>
                      {selectedSkill.scope}
                    </Badge>
                    {selectedContextModules.map((module) => (
                      <Badge key={`${selectedSkill.key}:${module}`} variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
                        {formatContextModule(module)}
                      </Badge>
                    ))}
                  </div>
                </div>

                <FlowStepCard
                  step={1}
                  title="Bind a paper"
                  description="Choose the paper this skill should drive. The same paper will receive the skill attachment when you enter Studio."
                  status={selectedPaper ? "ready" : "active"}
                >
                  {papers.length === 0 ? (
                    <div className="rounded-[18px] border border-dashed border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
                      No Studio papers exist yet. Open Studio once to import or create a paper first.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <Select value={selectedPaperId} onValueChange={handlePaperChange}>
                        <SelectTrigger className="h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-700">
                          <SelectValue placeholder="Choose a paper" />
                        </SelectTrigger>
                        <SelectContent>
                          {papers.map((paper) => (
                            <SelectItem key={paper.id} value={paper.id}>
                              {paper.title}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {selectedPaper ? (
                        <div className="rounded-[18px] border border-slate-200 bg-white px-3 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">Selected paper</p>
                          <p className="mt-1 text-sm font-medium text-slate-900">{selectedPaper.title}</p>
                          <p className="mt-1 text-[13px] leading-6 text-slate-600">
                            {selectedPaper.abstract || "No abstract stored for this paper yet."}
                          </p>
                        </div>
                      ) : null}
                    </div>
                  )}
                </FlowStepCard>

                <FlowStepCard
                  step={2}
                  title="Confirm workspace"
                  description={
                    requiresWorkspace
                      ? "This skill expects a writable workspace before Claude Code enters Code mode."
                      : "Workspace is optional for this skill, but you can still pin one now."
                  }
                  status={!requiresWorkspace || workspacePath.trim() ? "ready" : selectedPaper ? "active" : "pending"}
                >
                  <Input
                    value={workspacePath}
                    onChange={(event) => setWorkspacePath(event.target.value)}
                    placeholder={cwd ?? "/tmp"}
                    className="h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-800"
                  />
                  <p className="mt-2 text-[12px] text-slate-500">
                    Current launch path: {formatWorkspaceSummary(workspacePath || cwd)}
                  </p>
                </FlowStepCard>

                <FlowStepCard
                  step={3}
                  title="Prepare paper context"
                  description={
                    requiresGeneratedContext
                      ? "This skill declares generated paper context modules, so the next action will build them before chat starts."
                      : "This skill can start from the selected paper plus workspace alone. Generated context stays optional."
                  }
                  status={contextReady || !requiresGeneratedContext ? "ready" : selectedPaper ? "active" : "pending"}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      className="h-9 rounded-full border-slate-200 bg-white px-4 text-sm text-slate-700"
                      disabled={!selectedPaper || generationStatus === "generating"}
                      onClick={() => void handleGenerateContext()}
                    >
                      {generationStatus === "generating" ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Sparkles className="mr-2 h-4 w-4" />
                      )}
                      {contextReady ? "Refresh context" : "Generate context"}
                    </Button>
                    {contextReady ? (
                      <Badge className="h-7 border-emerald-200 bg-emerald-50 px-2.5 text-[10px] text-emerald-700">
                        context ready
                      </Badge>
                    ) : null}
                    {!requiresGeneratedContext ? (
                      <Badge variant="outline" className="h-7 border-slate-200 bg-white px-2.5 text-[10px] text-slate-600">
                        optional
                      </Badge>
                    ) : null}
                  </div>
                </FlowStepCard>

                <FlowStepCard
                  step={4}
                  title="Attach and continue"
                  description="Studio will open with this skill already attached to the selected paper, so normal chat turns can reuse the skill workflow without typing the slash command again."
                  status={selectedSkillAttached && (!requiresGeneratedContext || contextReady) ? "ready" : selectedPaper ? "active" : "pending"}
                >
                  <Textarea
                    value={`Skill: ${selectedSkill.title}
Paper: ${selectedPaper?.title ?? "Choose a paper"}
Workspace: ${formatWorkspaceSummary(workspacePath || cwd)}
Context: ${
  requiresGeneratedContext
    ? contextReady
      ? "Ready"
      : "Needs generation"
    : "Optional"
}
Attachment: ${selectedSkillAttached ? "Already attached" : "Will attach on launch"}`}
                    readOnly
                    className="min-h-[138px] rounded-[20px] border-slate-200 bg-white px-4 py-3 text-sm text-slate-700"
                  />
                </FlowStepCard>

                <div className="rounded-[22px] border border-slate-200 bg-[#fbfbf8] p-4">
                  <Button
                    type="button"
                    className="h-11 w-full rounded-full bg-slate-900 px-4 text-sm text-white hover:bg-slate-800"
                    disabled={selectedSkillLoading || generationStatus === "generating" || papers.length === 0}
                    onClick={() => void handlePrimaryAction()}
                  >
                    {selectedSkillLoading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : null}
                    {primaryActionLabel}
                  </Button>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button asChild variant="outline" size="sm" className="h-8 rounded-full border-slate-200 bg-white px-3 text-[11px] text-slate-700">
                      <Link href={`/skills/${encodeURIComponent(selectedSkill.key)}`}>
                        View README
                      </Link>
                    </Button>
                    <Button asChild variant="ghost" size="sm" className="h-8 rounded-full px-3 text-[11px] text-slate-600 hover:bg-white">
                      <Link href={`/skills/${encodeURIComponent(selectedSkill.key)}/setup`}>
                        Open full setup page
                      </Link>
                    </Button>
                  </div>
                  {flowError ? <p className="mt-3 text-sm text-rose-600">{flowError}</p> : null}
                </div>
              </div>
            )}
          </Section>
        </div>
      </div>
    </PageShell>
  )
}

export function SkillDetailPage({ skillKey }: { skillKey: string }) {
  const { detail, loading } = useStudioSkillDetail(skillKey)

  return (
    <PageShell
      eyebrow="Skill Detail"
      title={detail.skill?.title ?? "Skill"}
      description={
        detail.skill?.description ||
        "Review the skill metadata, source repo, and required paper context before launching Studio."
      }
      actions={
        <>
          <Button asChild variant="outline" className="h-10 rounded-full border-slate-200 bg-white px-4 text-sm text-slate-700">
            <Link href="/skills">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to skills
            </Link>
          </Button>
          {detail.skill ? (
            <Button asChild className="h-10 rounded-full bg-slate-900 px-4 text-sm text-white hover:bg-slate-800">
              <Link href={`/skills?skill=${encodeURIComponent(detail.skill.key)}`}>
                Open launch flow
              </Link>
            </Button>
          ) : null}
        </>
      }
    >
      {loading ? (
        <Section title="Loading" description="Fetching skill metadata...">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading skill detail...
          </div>
        </Section>
      ) : !detail.skill ? (
        <Section title="Skill not found" description="This skill key is not available in the current Studio catalog.">
          <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
            Check whether the skill repo is still installed, or refresh the Skills directory.
          </div>
        </Section>
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(320px,0.7fr)]">
            <Section title="Overview" description="Studio uses this metadata when attaching the skill to chat and setup.">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="h-6 border-slate-200 bg-[#f7f8f4] text-[10px] text-slate-600">
                  {detail.skill.slashCommand}
                </Badge>
                <Badge className={cn("h-6 px-2.5 text-[10px]", scopeBadgeClassName(detail.skill.scope))}>
                  {detail.skill.scope}
                </Badge>
                {detail.skill.repoLabel ? (
                  <Badge variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
                    {detail.skill.repoLabel}
                  </Badge>
                ) : null}
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Context modules</p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {detail.contextModules.length > 0 ? (
                      detail.contextModules.map((module) => (
                        <Badge key={module} variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
                          {formatContextModule(module)}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-sm text-slate-500">No explicit modules declared.</span>
                    )}
                  </div>
                </div>
                <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Tools</p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {detail.skill.tools.length > 0 ? (
                      detail.skill.tools.map((tool) => (
                        <Badge key={tool} variant="outline" className="h-6 border-slate-200 bg-white text-[10px] text-slate-600">
                          {tool}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-sm text-slate-500">No tool hints declared.</span>
                    )}
                  </div>
                </div>
              </div>
            </Section>

            <Section title="Source" description="Use the exact installed path when you want Claude Code to locate the skill deterministically.">
              <div className="space-y-3 text-sm text-slate-600">
                {detail.skill.repoUrl ? (
                  <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Git repo</p>
                    <p className="mt-2 break-all">{detail.skill.repoUrl}</p>
                  </div>
                ) : null}
                {detail.skill.paths.length > 0 ? (
                  <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Paths</p>
                    <div className="mt-2 space-y-1">
                      {detail.skill.paths.map((path) => (
                        <p key={path} className="break-all text-[13px] text-slate-600">
                          {path}
                        </p>
                      ))}
                    </div>
                  </div>
                ) : null}
                <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Workspace</p>
                  <p className="mt-2">
                    {detail.requiresWorkspace
                      ? "This setup flow assumes a workspace path before the first Code run."
                      : "This skill can launch without selecting a workspace first."}
                  </p>
                </div>
              </div>
            </Section>
          </div>

          <Section
            title="Skill README"
            description="The README is shown here so the setup wizard and chat entry stay grounded in the skill author’s workflow."
          >
            {detail.readme ? (
              <div className="prose max-w-none prose-slate">
                <Markdown remarkPlugins={[remarkGfm]}>{detail.readme}</Markdown>
              </div>
            ) : (
              <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
                No SKILL.md content was available for this skill.
              </div>
            )}
          </Section>
        </>
      )}
    </PageShell>
  )
}

export function SkillSetupPage({ skillKey }: { skillKey: string }) {
  const router = useRouter()
  const { detail, loading } = useStudioSkillDetail(skillKey)
  const { generate, status: generationStatus } = useContextPackGeneration()
  const {
    papers,
    loadPapers,
    selectPaper,
    updatePaper,
    contextPack,
    setAttachedSkill,
  } = useStudioStore()
  const [selectedPaperId, setSelectedPaperId] = useState<string>("")
  const [workspacePath, setWorkspacePath] = useState("")
  const [cwd, setCwd] = useState<string | null>(null)
  const [contextModules, setContextModules] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadPapers()
  }, [loadPapers])

  useEffect(() => {
    let active = true
    void fetch("/api/studio/cwd", { cache: "no-store" })
      .then((response) => response.json() as Promise<StudioCwdPayload>)
      .then((payload) => {
        if (!active) return
        const nextCwd = payload.actual_cwd || payload.cwd || payload.home || null
        setCwd(nextCwd)
      })
      .catch(() => {
        if (!active) return
        setCwd(null)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!detail.skill) return
    setContextModules(detail.contextModules)
  }, [detail.contextModules, detail.skill])

  useEffect(() => {
    if (papers.length === 0) {
      setSelectedPaperId("")
      return
    }
    if (selectedPaperId && papers.some((paper) => paper.id === selectedPaperId)) {
      return
    }
    setSelectedPaperId(papers[0].id)
  }, [papers, selectedPaperId])

  const selectedPaper = useMemo(
    () => papers.find((paper) => paper.id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  )

  useEffect(() => {
    if (!selectedPaper) {
      setWorkspacePath(cwd ?? "")
      return
    }
    selectPaper(selectedPaper.id)
    setWorkspacePath(selectedPaper.outputDir || cwd || "")
  }, [cwd, selectPaper, selectedPaper?.id, selectedPaper?.outputDir])

  const requiresGeneratedContext = skillNeedsGeneratedContext(contextModules)
  const requiresWorkspace = skillNeedsWorkspace(contextModules, {
    requiresWorkspaceHint: detail.requiresWorkspace,
  })

  const handleGenerateContext = async () => {
    if (!selectedPaper) {
      setError("Choose a paper before generating context.")
      return
    }
    setError(null)
    selectPaper(selectedPaper.id)
    await generate({
      paperId: selectedPaper.id,
      title: selectedPaper.title,
      abstract: selectedPaper.abstract,
    })
  }

  const handleLaunch = () => {
    if (!detail.skill) {
      setError("Skill metadata is not ready yet.")
      return
    }
    if (!selectedPaper) {
      setError("Choose a paper before launching Studio.")
      return
    }
    if (requiresWorkspace && !workspacePath.trim()) {
      setError("Set a workspace before launching Studio.")
      return
    }

    selectPaper(selectedPaper.id)
    if (workspacePath.trim()) {
      updatePaper(selectedPaper.id, { outputDir: workspacePath.trim() })
    }
    setAttachedSkill({
      ...detail.skill,
      contextModules,
    })
    router.push(`/studio?paperId=${encodeURIComponent(selectedPaper.id)}`)
  }

  return (
    <PageShell
      eyebrow="Skill Setup"
      title={detail.skill?.title ?? "Set up skill"}
      description="Choose the paper, workspace, and context modules first. Paper context is generated only when this selected skill actually needs it."
      actions={
        <>
          <Button asChild variant="outline" className="h-10 rounded-full border-slate-200 bg-white px-4 text-sm text-slate-700">
            <Link href={detail.skill ? `/skills?skill=${encodeURIComponent(detail.skill.key)}` : "/skills"}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <Button
            type="button"
            className="h-10 rounded-full bg-slate-900 px-4 text-sm text-white hover:bg-slate-800"
            disabled={loading || !detail.skill || !selectedPaper || (requiresWorkspace && !workspacePath.trim())}
            onClick={handleLaunch}
          >
            Continue in Studio
          </Button>
        </>
      }
    >
      {loading ? (
        <Section title="Loading" description="Fetching setup metadata...">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading setup...
          </div>
        </Section>
      ) : !detail.skill ? (
        <Section title="Skill not found" description="This skill is not available in the current Studio catalog.">
          <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
            Refresh the Skills directory or reinstall the skill repo first.
          </div>
        </Section>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(360px,1.05fr)]">
          <Section
            title="Paper"
            description="Select the paper that this skill should work against. The setup wizard only generates paper context when the skill requires it."
          >
            {papers.length === 0 ? (
              <div className="rounded-[22px] border border-dashed border-slate-200 bg-[#fbfbf8] p-6 text-sm text-slate-500">
                No Studio papers are available yet. Open Studio once to create or import a paper first.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                  <Label className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Selected paper
                  </Label>
                  <Select value={selectedPaperId} onValueChange={setSelectedPaperId}>
                    <SelectTrigger className="mt-3 h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-700">
                      <SelectValue placeholder="Choose a paper" />
                    </SelectTrigger>
                    <SelectContent>
                      {papers.map((paper) => (
                        <SelectItem key={paper.id} value={paper.id}>
                          {paper.title}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {selectedPaper ? (
                  <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Abstract</p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{selectedPaper.abstract || "No abstract stored yet."}</p>
                  </div>
                ) : null}
              </div>
            )}
          </Section>

          <Section
            title="Setup"
            description="Keep workspace and paper context explicit before you enter chat. This replaces the old in-chat skills panel as the primary launch path."
          >
            <div className="space-y-4">
              <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                <div className="flex items-center gap-2">
                  <FolderOpen className="h-4 w-4 text-slate-500" />
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Workspace</p>
                    <p className="text-sm text-slate-600">Pick the directory Claude Code should use when this skill enters Code mode.</p>
                  </div>
                </div>
                <Input
                  value={workspacePath}
                  onChange={(event) => setWorkspacePath(event.target.value)}
                  placeholder={cwd ?? "/tmp"}
                  className="mt-3 h-11 rounded-2xl border-slate-200 bg-white px-4 text-sm text-slate-800"
                />
                <p className="mt-2 text-[12px] text-slate-500">
                  {requiresWorkspace
                    ? "Workspace is required by the selected setup contract."
                    : "Workspace stays optional for this skill."}
                </p>
              </div>

              <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                <div className="flex items-center gap-2">
                  <Package2 className="h-4 w-4 text-slate-500" />
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Context modules</p>
                    <p className="text-sm text-slate-600">These are the paper-context slices this skill expects to consume.</p>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {detail.contextModules.length > 0 ? (
                    detail.contextModules.map((module) => {
                      const selected = contextModules.includes(module)
                      return (
                        <button
                          key={module}
                          type="button"
                          className={cn(
                            "rounded-full border px-3 py-1 text-[11px] transition-colors",
                            selected
                              ? "border-slate-300 bg-[#eef1ea] text-slate-900"
                              : "border-slate-200 bg-white text-slate-600 hover:bg-[#f7f8f4]",
                          )}
                          onClick={() =>
                            setContextModules((current) =>
                              current.includes(module)
                                ? current.filter((item) => item !== module)
                                : [...current, module],
                            )
                          }
                        >
                          {formatContextModule(module)}
                        </button>
                      )
                    })
                  ) : (
                    <p className="text-sm text-slate-500">No explicit context modules were declared for this skill.</p>
                  )}
                </div>
              </div>

              <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                <div className="flex items-center gap-2">
                  <Search className="h-4 w-4 text-slate-500" />
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Paper context</p>
                    <p className="text-sm text-slate-600">
                      {requiresGeneratedContext
                        ? "This skill requests extracted paper context. Generate it here before launching chat."
                        : "This skill can start from the paper title, abstract, and workspace alone."}
                    </p>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="h-9 rounded-full border-slate-200 bg-white px-4 text-sm text-slate-700"
                    disabled={!selectedPaper || generationStatus === "generating"}
                    onClick={() => void handleGenerateContext()}
                  >
                    {generationStatus === "generating" ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Sparkles className="mr-2 h-4 w-4" />
                    )}
                    Generate paper context
                  </Button>
                  {contextPack?.context_pack_id ? (
                    <Badge className="h-7 border-emerald-200 bg-emerald-50 px-2.5 text-[10px] text-emerald-700">
                      context ready: {contextPack.context_pack_id}
                    </Badge>
                  ) : null}
                  {!requiresGeneratedContext ? (
                    <Badge variant="outline" className="h-7 border-slate-200 bg-white px-2.5 text-[10px] text-slate-600">
                      generation optional
                    </Badge>
                  ) : null}
                </div>
              </div>

              <div className="rounded-[18px] border border-slate-200 bg-[#fbfbf8] p-4">
                <div className="flex items-center gap-2">
                  <Wrench className="h-4 w-4 text-slate-500" />
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Launch summary</p>
                    <p className="text-sm text-slate-600">
                      Studio will open with this skill attached so natural-language turns can use the skill workflow without typing the slash command again.
                    </p>
                  </div>
                </div>
                <Textarea
                  value={`Skill: ${detail.skill.title}\nPaper: ${selectedPaper?.title ?? "Choose a paper"}\nWorkspace: ${workspacePath || cwd || "Not set"}\nContext modules: ${contextModules.map(formatContextModule).join(", ") || "None"}`}
                  readOnly
                  className="mt-3 min-h-[132px] rounded-[20px] border-slate-200 bg-white px-4 py-3 text-sm text-slate-700"
                />
              </div>

              {error ? <p className="text-sm text-rose-600">{error}</p> : null}
            </div>
          </Section>
        </div>
      )}
    </PageShell>
  )
}
