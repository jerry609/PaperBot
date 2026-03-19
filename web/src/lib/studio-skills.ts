import type { StudioSkillInfo } from "@/lib/studio-runtime"

const STUDIO_SKILL_ECOSYSTEM_LABELS: Record<string, string> = {
  claude_code: "Claude Code",
  opencode: "OpenCode",
  github_copilot: "GitHub Copilot",
}

export type ParsedStudioSkillCommand = {
  skill: StudioSkillInfo
  args: string
  slashInput: string
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
  skill: StudioSkillInfo,
  args: string,
  options: {
    paperTitle?: string | null
    contextPackId?: string | null
    workspacePath?: string | null
  } = {},
): string {
  const userRequest =
    args.trim() ||
    (options.paperTitle
      ? `Apply this skill to the selected paper "${options.paperTitle}".`
      : "Apply this skill to the current Studio context.")

  const lines = [
    `Use the available project skill \`${skill.id}\` if it is present in this repository.`,
    skill.description ? `Skill description: ${skill.description}` : "",
    skill.tools.length > 0 ? `Preferred tools from the skill: ${skill.tools.join(", ")}` : "",
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
  const ecosystems = skill.ecosystems.length > 0 ? skill.ecosystems : skill.primaryEcosystem ? [skill.primaryEcosystem] : []
  if (ecosystems.length === 0) {
    return "Project skill"
  }
  return ecosystems.map((entry) => formatStudioSkillEcosystemLabel(entry)).join(" + ")
}
