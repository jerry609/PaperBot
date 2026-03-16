import type { AgentTaskLog, BlockType } from "@/lib/store/studio-store"

export type TaskTimelineFilterMode = "all" | "diffs" | "thinking"
export type TaskTimelineDisplayMode = "summary" | "raw"

export type TaskTimelineEntry =
  | {
      kind: "log"
      id: string
      log: AgentTaskLog
      blockType: BlockType
    }
  | {
      kind: "group"
      id: string
      blockType: "think" | "tool" | "info"
      logs: AgentTaskLog[]
      title: string
      preview: string[]
      status: "neutral" | "running" | "success" | "error"
      toolNames: string[]
    }

export function inferTaskLogBlockType(log: AgentTaskLog): BlockType {
  const bt = log.blockType ?? ((log as unknown as Record<string, unknown>)["block_type"] as BlockType | undefined)
  if (bt) return bt
  if (log.event === "thinking" || log.event === "reasoning") return "think"
  if (log.event === "file_write" || log.event === "file_edit") return "diff"
  if (
    log.event === "verify_result" ||
    log.event === "review_result" ||
    log.event === "executor_finished" ||
    log.event === "task_done" ||
    log.event === "human_approved" ||
    log.event === "human_rejected" ||
    log.event === "human_requested_changes"
  ) {
    return "result"
  }
  if (log.event === "tool_call" || log.event === "shell_exec" || log.event === "run_command") {
    const tool = log.details?.tool as string | undefined
    if (tool === "write_file") return "diff"
    return "tool"
  }
  return "info"
}

export function stripLogPrefix(message: string): string {
  return message.replace(/^\[step \d+\]\s*/, "").trim()
}

function extractToolName(log: AgentTaskLog): string {
  const explicitTool = log.details?.tool
  if (typeof explicitTool === "string" && explicitTool.trim()) {
    return explicitTool.trim()
  }

  const stripped = stripLogPrefix(log.message)
  const beforeColon = stripped.split(":")[0] ?? stripped
  const beforeParen = beforeColon.split("(")[0] ?? beforeColon
  return beforeParen.trim() || "tool"
}

function groupStatus(logs: AgentTaskLog[]): "neutral" | "running" | "success" | "error" {
  if (logs.some((log) => log.level === "error")) return "error"
  if (logs.some((log) => log.level === "success")) return "success"
  if (logs.some((log) => log.level === "warning")) return "running"
  return "neutral"
}

function buildGroupEntry(blockType: "think" | "tool" | "info", logs: AgentTaskLog[]): TaskTimelineEntry {
  const messages = logs.map((log) => stripLogPrefix(log.message)).filter(Boolean)
  const preview = messages.slice(-2)

  if (blockType === "think") {
    return {
      kind: "group",
      id: `group-${logs[0]?.id ?? "think"}`,
      blockType,
      logs,
      title: logs.length === 1 ? "Reasoning step" : `${logs.length} reasoning steps`,
      preview,
      status: "running",
      toolNames: [],
    }
  }

  if (blockType === "tool") {
    const toolNames = Array.from(new Set(logs.map(extractToolName))).slice(0, 5)
    const errorCount = logs.filter((log) => log.level === "error").length
    const title =
      errorCount > 0
        ? `${logs.length} tool actions with ${errorCount} error${errorCount === 1 ? "" : "s"}`
        : logs.length === 1
          ? "Tool action"
          : `${logs.length} tool actions`
    return {
      kind: "group",
      id: `group-${logs[0]?.id ?? "tool"}`,
      blockType,
      logs,
      title,
      preview: toolNames.length > 0 ? toolNames : preview,
      status: groupStatus(logs),
      toolNames,
    }
  }

  return {
    kind: "group",
    id: `group-${logs[0]?.id ?? "info"}`,
    blockType,
    logs,
    title: logs.length === 1 ? "Status update" : `${logs.length} status updates`,
    preview,
    status: groupStatus(logs),
    toolNames: [],
  }
}

export function filterTaskLogs(logs: AgentTaskLog[], filter: TaskTimelineFilterMode): AgentTaskLog[] {
  return logs.filter((log) => {
    if (filter === "all") return true
    const bt = inferTaskLogBlockType(log)
    if (filter === "diffs") return bt === "diff"
    if (filter === "thinking") return bt === "think"
    return true
  })
}

export function buildTaskTimelineEntries(logs: AgentTaskLog[]): TaskTimelineEntry[] {
  const entries: TaskTimelineEntry[] = []
  let currentGroupType: "think" | "tool" | "info" | null = null
  let currentGroup: AgentTaskLog[] = []

  const flushGroup = () => {
    if (!currentGroupType || currentGroup.length === 0) return
    entries.push(buildGroupEntry(currentGroupType, currentGroup))
    currentGroupType = null
    currentGroup = []
  }

  for (const log of logs) {
    const blockType = inferTaskLogBlockType(log)
    if (blockType === "diff" || blockType === "result") {
      flushGroup()
      entries.push({
        kind: "log",
        id: log.id,
        log,
        blockType,
      })
      continue
    }

    const groupableType = blockType === "think" || blockType === "tool" ? blockType : "info"
    if (currentGroupType === groupableType) {
      currentGroup.push(log)
      continue
    }

    flushGroup()
    currentGroupType = groupableType
    currentGroup = [log]
  }

  flushGroup()
  return entries
}
