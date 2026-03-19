import type { StudioSkillInfo } from "@/lib/studio-runtime"

const STUDIO_SKILL_ECOSYSTEM_LABELS: Record<string, string> = {
  claude_code: "Claude Code",
  opencode: "OpenCode",
  github_copilot: "GitHub Copilot",
}

const GENERATED_CONTEXT_MODULES = new Set([
  "literature",
  "environment",
  "spec",
  "roadmap",
  "success_criteria",
])

export type ParsedStudioSkillCommand = {
  skill: StudioSkillInfo
  args: string
  slashInput: string
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
    const cleaned = cleanString(item)
    if (!cleaned || seen.has(cleaned)) continue
    seen.add(cleaned)
    normalized.push(cleaned)
  }
  return normalized
}

type StudioSkillLike = Partial<StudioSkillInfo>

type SkillPromptSkill = StudioSkillLike & {
  id: string
  title: string
  description: string
  slashCommand: string
}

function splitSlashInput(input: string): { command: string; args: string; slashInput: string } | null {
  const trimmed = input.trim()
  if (!trimmed.startsWith("/")) return null

  const body = trimmed.slice(1).trim()
  if (!body) return null

  const whitespaceIndex = body.search(/\s/)
  if (whitespaceIndex === -1) {
    return {
      command: body.toLowerCase(),
      args: "",
      slashInput: trimmed,
    }
  }

  return {
    command: body.slice(0, whitespaceIndex).toLowerCase(),
    args: body.slice(whitespaceIndex + 1).trim(),
    slashInput: trimmed,
  }
}

export function parseStudioSkillSlashCommand(
  input: string,
  skills: StudioSkillInfo[],
): ParsedStudioSkillCommand | null {
  const parsed = splitSlashInput(input)
  if (!parsed) return null

  const matchedSkill = skills.find((skill) => skill.slashCommand.replace(/^\//, "").toLowerCase() === parsed.command)
  if (!matchedSkill) return null

  return {
    skill: matchedSkill,
    args: parsed.args,
    slashInput: parsed.slashInput,
  }
}

export function buildStudioSkillPrompt(
  skill: SkillPromptSkill,
  args: string,
  options: {
    paperTitle?: string | null
    contextPackId?: string | null
    workspacePath?: string | null
  } = {},
): string {
  const tools = getStudioSkillTools(skill)
  const paths = getStudioSkillPaths(skill)
  const contextModules = cleanStringList(skill.contextModules)
  const repoLabel = cleanString(skill.repoLabel)
  const repoUrl = cleanString(skill.repoUrl)
  const userRequest =
    args.trim() ||
    (options.paperTitle
      ? `Apply this skill to the selected paper "${options.paperTitle}".`
      : "Apply this skill to the current Studio context.")

  const lines = [
    `Use the available Studio skill \`${skill.id}\` if it is present in this repository.`,
    skill.description ? `Skill description: ${skill.description}` : "",
    repoLabel ? `Skill source: ${repoLabel}` : "",
    repoUrl ? `Git source: ${repoUrl}` : "",
    paths.length > 0 ? `Skill paths: ${paths.join(", ")}` : "",
    tools.length > 0 ? `Preferred tools from the skill: ${tools.join(", ")}` : "",
    contextModules.length > 0 ? `Requested context modules: ${contextModules.join(", ")}` : "",
    skill.promptHint ? `Studio hint: ${skill.promptHint}` : "",
    options.paperTitle ? `Selected paper: ${options.paperTitle}` : "",
    options.contextPackId ? `Current context pack id: ${options.contextPackId}` : "",
    options.workspacePath ? `Workspace path: ${options.workspacePath}` : "",
    "",
    "User request:",
    userRequest,
    "",
    "If this transport cannot invoke the skill directly, follow the same workflow manually and state that you are mirroring the project skill in managed chat.",
  ].filter((line) => line.trim().length > 0)

  return lines.join("\n")
}

export function formatStudioSkillEcosystemLabel(ecosystem: string): string {
  return STUDIO_SKILL_ECOSYSTEM_LABELS[ecosystem] ?? ecosystem
}

export function buildStudioSkillAvailabilityLabel(skill: StudioSkillInfo): string {
  const ecosystems = getStudioSkillEcosystems(skill)
  if (ecosystems.length === 0) {
    return "Project skill"
  }
  return ecosystems.map((entry) => formatStudioSkillEcosystemLabel(entry)).join(" + ")
}

export function getStudioSkillTools(skill: StudioSkillLike): string[] {
  return cleanStringList(skill.tools)
}

export function getStudioSkillRecommendedFor(skill: StudioSkillLike): string[] {
  return cleanStringList(skill.recommendedFor)
}

export function getStudioSkillEcosystems(skill: StudioSkillLike): string[] {
  const ecosystems = cleanStringList(skill.ecosystems)
  if (ecosystems.length > 0) return ecosystems

  const primary = cleanString(skill.primaryEcosystem)
  return primary ? [primary] : []
}

export function getStudioSkillPaths(skill: StudioSkillLike): string[] {
  const paths = cleanStringList(skill.paths)
  if (paths.length > 0) return paths

  const primaryPath = cleanString(skill.path)
  return primaryPath ? [primaryPath] : []
}

export function skillNeedsGeneratedContext(input: StudioSkillLike | string[]): boolean {
  const modules = Array.isArray(input) ? cleanStringList(input) : cleanStringList(input.contextModules)
  return modules.some((module) => GENERATED_CONTEXT_MODULES.has(module))
}

export function skillNeedsWorkspace(
  input: StudioSkillLike | string[],
  options: { requiresWorkspaceHint?: boolean } = {},
): boolean {
  if (options.requiresWorkspaceHint) return true
  const modules = Array.isArray(input) ? cleanStringList(input) : cleanStringList(input.contextModules)
  return modules.includes("workspace")
}
