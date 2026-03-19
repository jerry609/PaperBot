import type { AgentAction } from "@/lib/store/studio-store"

type ActivityCategory = "read" | "search" | "write" | "command" | "delegation" | "web" | "other"

const EMPTY_COUNTS: Record<ActivityCategory, number> = {
  read: 0,
  search: 0,
  write: 0,
  command: 0,
  delegation: 0,
  web: 0,
  other: 0,
}

function summarizePayload(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") {
    if (typeof payload === "string" && payload.trim()) {
      return payload.trim().replace(/\s+/g, " ").slice(0, 96)
    }
    return null
  }

  const record = payload as Record<string, unknown>
  const priorityKeys = [
    "path",
    "file_path",
    "filename",
    "target_path",
    "target_file",
    "command",
    "query",
    "pattern",
    "description",
    "prompt",
    "task_title",
    "message",
    "instructions",
    "assignee",
    "subagent_type",
  ]

  for (const key of priorityKeys) {
    const value = record[key]
    if (typeof value === "string" && value.trim()) {
      return value.trim().replace(/\s+/g, " ").slice(0, 96)
    }
  }

  return null
}

function isDelegationPayload(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") {
    return false
  }

  const record = payload as Record<string, unknown>
  const candidateKeys = [
    "agent",
    "assignee",
    "delegate_to",
    "runtime",
    "executor",
    "runner",
    "subagent",
    "subagent_type",
    "backend",
    "provider",
    "target",
  ]

  return candidateKeys.some((key) => {
    const value = record[key]
    if (typeof value !== "string") return false
    const normalized = value.trim().toLowerCase()
    if (!normalized) return false
    return (
      normalized.includes("codex") ||
      normalized.includes("opencode") ||
      normalized.includes("subagent") ||
      normalized.includes("worker") ||
      normalized.includes("team")
    )
  })
}

function classifyTool(action: AgentAction): ActivityCategory {
  const normalized = action.metadata?.functionName?.trim().toLowerCase() || ""

  if (
    normalized === "agent" ||
    normalized === "task" ||
    normalized.includes("delegate") ||
    normalized.includes("spawn_agent") ||
    normalized.includes("subagent") ||
    normalized.includes("team") ||
    isDelegationPayload(action.metadata?.params)
  ) {
    return "delegation"
  }

  if (
    normalized.includes("read") ||
    normalized.includes("open") ||
    normalized.includes("cat") ||
    normalized === "glob" ||
    normalized === "ls" ||
    normalized.includes("list")
  ) {
    return "read"
  }

  if (
    normalized.includes("grep") ||
    normalized.includes("search") ||
    normalized.includes("find") ||
    normalized.includes("match")
  ) {
    return "search"
  }

  if (
    normalized.includes("write") ||
    normalized.includes("edit") ||
    normalized.includes("replace") ||
    normalized.includes("patch") ||
    normalized.includes("create") ||
    normalized.includes("delete")
  ) {
    return "write"
  }

  if (
    normalized === "bash" ||
    normalized.includes("shell") ||
    normalized.includes("command") ||
    normalized.includes("terminal") ||
    normalized.includes("exec") ||
    normalized.includes("run")
  ) {
    return "command"
  }
  if (normalized.includes("web") || normalized.includes("fetch") || normalized.includes("browser")) {
    return "web"
  }

  return "other"
}

function buildSummaryLabel(counts: Record<ActivityCategory, number>): string {
  if (counts.delegation > 0 && counts.write > 0) {
    return "Coordinating subagents and edits"
  }
  if (counts.delegation > 0) {
    return "Coordinating subagents"
  }
  if (counts.write > 0 && (counts.read > 0 || counts.search > 0)) {
    return "Inspecting and editing files"
  }
  if (counts.write > 0) {
    return "Editing project files"
  }
  if (counts.command > 0 && (counts.read > 0 || counts.search > 0)) {
    return "Scanning the workspace and running commands"
  }
  if (counts.read > 0 || counts.search > 0) {
    return "Scanning the workspace"
  }
  if (counts.command > 0) {
    return "Running command-line tools"
  }
  if (counts.web > 0) {
    return "Gathering external context"
  }
  return "Using tools"
}

function formatRecentTool(action: AgentAction): string {
  const functionName = action.metadata?.functionName?.trim() || "tool"
  const payloadSummary =
    summarizePayload(action.metadata?.params) ?? summarizePayload(action.metadata?.result)
  return payloadSummary ? `${functionName}: ${payloadSummary}` : functionName
}

function buildStageSequence(actions: AgentAction[]): ActivityCategory[] {
  const sequence: ActivityCategory[] = []

  for (const action of actions) {
    const category = classifyTool(action)
    if (category === "other") continue
    if (sequence[sequence.length - 1] === category) continue
    sequence.push(category)
  }

  return sequence.length > 0 ? sequence : ["other"]
}

function buildActivitySummaryAction(actions: AgentAction[]): AgentAction {
  const counts = { ...EMPTY_COUNTS }

  for (const action of actions) {
    const category = classifyTool(action)
    counts[category] += 1
  }

  const recent = actions.slice(-3).map(formatRecentTool)
  const latest = actions[actions.length - 1]
  const latestDelegationAction = [...actions]
    .reverse()
    .find((action) => classifyTool(action) === "delegation" && typeof action.metadata?.toolId === "string" && action.metadata.toolId.trim().length > 0)

  return {
    id: `activity-summary-${actions[0]?.id ?? "tool"}`,
    type: "activity_summary",
    timestamp: latest?.timestamp ?? new Date(),
    content: buildSummaryLabel(counts),
    metadata: {
      activitySummary: {
        label: buildSummaryLabel(counts),
        status: latest?.metadata?.result === undefined ? "running" : "done",
        totalTools: actions.length,
        counts,
        stageSequence: buildStageSequence(actions),
        recent,
        delegationTaskId: latestDelegationAction?.metadata?.toolId?.trim() || undefined,
        toolActions: actions,
      },
    },
  }
}

export function collapseToolActivityActions(actions: AgentAction[]): AgentAction[] {
  const collapsed: AgentAction[] = []
  let currentRun: AgentAction[] = []

  const flushRun = () => {
    if (currentRun.length === 0) return
    collapsed.push(buildActivitySummaryAction(currentRun))
    currentRun = []
  }

  for (const action of actions) {
    if (action.type === "function_call" && action.metadata?.functionName) {
      currentRun.push(action)
      continue
    }

    flushRun()
    collapsed.push(action)
  }

  flushRun()
  return collapsed
}
