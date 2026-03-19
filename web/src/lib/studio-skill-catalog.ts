import type { StudioSkillInfo, StudioSkillPayload } from "@/lib/studio-runtime"

export interface StudioInstalledRepoPayload {
  slug?: string | null
  title?: string | null
  description?: string | null
  repo_url?: string | null
  repo_ref?: string | null
  install_path?: string | null
  installed_at?: string | null
  last_known_commit?: string | null
  remote_commit?: string | null
  update_available?: boolean | null
  skills?: StudioSkillPayload[] | null
}

export interface StudioMarketplaceRepoPayload {
  slug?: string | null
  title?: string | null
  description?: string | null
  repo_url?: string | null
  repo_ref?: string | null
  installed?: boolean | null
  install_path?: string | null
  installed_repo_slug?: string | null
  installed_skill_count?: number | null
  update_available?: boolean | null
}

export interface StudioSkillCatalogResponse {
  project_skills?: StudioSkillPayload[] | null
  installed_skills?: StudioSkillPayload[] | null
  installed_repos?: StudioInstalledRepoPayload[] | null
  marketplace_repos?: StudioMarketplaceRepoPayload[] | null
  updates?: StudioInstalledRepoPayload[] | null
  summary?: {
    project_skill_count?: number | null
    installed_skill_count?: number | null
    installed_repo_count?: number | null
    update_count?: number | null
    marketplace_repo_count?: number | null
  } | null
}

export interface StudioSkillDetailResponse {
  skill?: StudioSkillPayload | null
  readme?: string | null
  setup?: {
    requires_workspace?: boolean | null
    context_modules?: string[] | null
    recommended_for?: string[] | null
  } | null
}

export interface StudioInstalledRepoInfo {
  slug: string
  title: string
  description: string
  repoUrl: string
  repoRef: string | null
  installPath: string | null
  installedAt: string | null
  lastKnownCommit: string | null
  remoteCommit: string | null
  updateAvailable: boolean
  skills: StudioSkillInfo[]
}

export interface StudioMarketplaceRepoInfo {
  slug: string
  title: string
  description: string
  repoUrl: string
  repoRef: string | null
  installed: boolean
  installPath: string | null
  installedRepoSlug: string | null
  installedSkillCount: number
  updateAvailable: boolean
}

export interface StudioSkillCatalogInfo {
  projectSkills: StudioSkillInfo[]
  installedSkills: StudioSkillInfo[]
  installedRepos: StudioInstalledRepoInfo[]
  marketplaceRepos: StudioMarketplaceRepoInfo[]
  updates: StudioInstalledRepoInfo[]
  summary: {
    projectSkillCount: number
    installedSkillCount: number
    installedRepoCount: number
    updateCount: number
    marketplaceRepoCount: number
  }
}

export interface StudioSkillDetailInfo {
  skill: StudioSkillInfo | null
  readme: string
  requiresWorkspace: boolean
  contextModules: string[]
  recommendedFor: string[]
}

function cleanString(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function cleanStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const normalized: string[] = []
  const seen = new Set<string>()
  for (const item of value) {
    if (typeof item !== "string") continue
    const cleaned = item.trim()
    if (!cleaned || seen.has(cleaned)) continue
    seen.add(cleaned)
    normalized.push(cleaned)
  }
  return normalized
}

function normalizeStudioSkills(value: unknown): StudioSkillInfo[] {
  if (!Array.isArray(value)) return []

  const normalized: StudioSkillInfo[] = []
  const seen = new Set<string>()

  for (const item of value) {
    if (!item || typeof item !== "object") continue
    const payload = item as StudioSkillPayload
    const id = cleanString(payload.id)
    const title = cleanString(payload.title)
    const slashCommand = cleanString(payload.slash_command)
    const path = cleanString(payload.path)
    const primaryEcosystem = cleanString(payload.primary_ecosystem)
    const key = cleanString(payload.key)
    const paths = cleanStringList(payload.paths)
    if (!id || !title || !slashCommand) continue

    const dedupeKey = key ?? `${cleanString(payload.scope) ?? "project"}:${id}:${slashCommand}`
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)

    normalized.push({
      key: key ?? dedupeKey.replace(/[^\w:-]+/g, "-"),
      id,
      title,
      description: cleanString(payload.description) ?? "",
      slashCommand,
      scope: cleanString(payload.scope) ?? "project",
      tools: cleanStringList(payload.tools),
      recommendedFor: cleanStringList(payload.recommended_for),
      ecosystems: cleanStringList(payload.ecosystems),
      primaryEcosystem,
      paths: paths.length > 0 ? paths : path ? [path] : [],
      manifestSource: cleanString(payload.manifest_source),
      path,
      promptHint: cleanString(payload.prompt_hint),
      repoSlug: cleanString(payload.repo_slug),
      repoUrl: cleanString(payload.repo_url),
      repoLabel: cleanString(payload.repo_label),
      repoRef: cleanString(payload.repo_ref),
      repoCommit: cleanString(payload.repo_commit),
      contextModules: cleanStringList(payload.context_modules),
    })
  }

  return normalized
}

function normalizeInstalledRepos(value: unknown): StudioInstalledRepoInfo[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const payload = item as StudioInstalledRepoPayload
    const slug = cleanString(payload.slug)
    const title = cleanString(payload.title)
    const repoUrl = cleanString(payload.repo_url)
    if (!slug || !title || !repoUrl) return []
    return [
      {
        slug,
        title,
        description: cleanString(payload.description) ?? "",
        repoUrl,
        repoRef: cleanString(payload.repo_ref),
        installPath: cleanString(payload.install_path),
        installedAt: cleanString(payload.installed_at),
        lastKnownCommit: cleanString(payload.last_known_commit),
        remoteCommit: cleanString(payload.remote_commit),
        updateAvailable: payload.update_available === true,
        skills: normalizeStudioSkills(payload.skills),
      },
    ]
  })
}

function normalizeMarketplaceRepos(value: unknown): StudioMarketplaceRepoInfo[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const payload = item as StudioMarketplaceRepoPayload
    const slug = cleanString(payload.slug)
    const title = cleanString(payload.title)
    const repoUrl = cleanString(payload.repo_url)
    if (!slug || !title || !repoUrl) return []
    return [
      {
        slug,
        title,
        description: cleanString(payload.description) ?? "",
        repoUrl,
        repoRef: cleanString(payload.repo_ref),
        installed: payload.installed === true,
        installPath: cleanString(payload.install_path),
        installedRepoSlug: cleanString(payload.installed_repo_slug),
        installedSkillCount:
          typeof payload.installed_skill_count === "number" ? payload.installed_skill_count : 0,
        updateAvailable: payload.update_available === true,
      },
    ]
  })
}

export function buildStudioSkillCatalogInfo(
  payload?: StudioSkillCatalogResponse | null,
): StudioSkillCatalogInfo {
  const projectSkills = normalizeStudioSkills(payload?.project_skills)
  const installedSkills = normalizeStudioSkills(payload?.installed_skills)
  const installedRepos = normalizeInstalledRepos(payload?.installed_repos)
  const marketplaceRepos = normalizeMarketplaceRepos(payload?.marketplace_repos)
  const updates = normalizeInstalledRepos(payload?.updates)

  return {
    projectSkills,
    installedSkills,
    installedRepos,
    marketplaceRepos,
    updates,
    summary: {
      projectSkillCount:
        typeof payload?.summary?.project_skill_count === "number"
          ? payload.summary.project_skill_count
          : projectSkills.length,
      installedSkillCount:
        typeof payload?.summary?.installed_skill_count === "number"
          ? payload.summary.installed_skill_count
          : installedSkills.length,
      installedRepoCount:
        typeof payload?.summary?.installed_repo_count === "number"
          ? payload.summary.installed_repo_count
          : installedRepos.length,
      updateCount:
        typeof payload?.summary?.update_count === "number"
          ? payload.summary.update_count
          : updates.length,
      marketplaceRepoCount:
        typeof payload?.summary?.marketplace_repo_count === "number"
          ? payload.summary.marketplace_repo_count
          : marketplaceRepos.length,
    },
  }
}

export function buildStudioSkillDetailInfo(
  payload?: StudioSkillDetailResponse | null,
): StudioSkillDetailInfo {
  const skills = normalizeStudioSkills(payload?.skill ? [payload.skill] : [])
  return {
    skill: skills[0] ?? null,
    readme: cleanString(payload?.readme) ?? "",
    requiresWorkspace: payload?.setup?.requires_workspace !== false,
    contextModules: cleanStringList(payload?.setup?.context_modules),
    recommendedFor: cleanStringList(payload?.setup?.recommended_for),
  }
}
