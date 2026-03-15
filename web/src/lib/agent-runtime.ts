export type AgentRuntimeKind = "commander" | "subagent" | "retry" | "custom"

export type AgentPresentation = {
  kind: AgentRuntimeKind
  label: string
  shortLabel: string
}

function titleCase(value: string): string {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

export function getAgentPresentation(assignee?: string | null): AgentPresentation {
  const raw = (assignee || "").trim()
  const normalized = raw.toLowerCase()

  if (!normalized || normalized === "claude" || normalized === "cc" || normalized.startsWith("claude-") || normalized.startsWith("cc-")) {
    return {
      kind: "commander",
      label: "CC Commander",
      shortLabel: "CC",
    }
  }

  if (normalized.startsWith("codex-retry")) {
    return {
      kind: "retry",
      label: "Codex Retry",
      shortLabel: "Codex",
    }
  }

  if (normalized.startsWith("codex")) {
    return {
      kind: "subagent",
      label: "Codex Subagent",
      shortLabel: "Codex",
    }
  }

  if (normalized.startsWith("opencode")) {
    return {
      kind: "subagent",
      label: "OpenCode Subagent",
      shortLabel: "OpenCode",
    }
  }

  return {
    kind: "custom",
    label: titleCase(raw),
    shortLabel: titleCase(raw.split(/[-_\s]+/)[0] || raw),
  }
}

export function isCommanderAssignee(assignee?: string | null): boolean {
  return getAgentPresentation(assignee).kind === "commander"
}
