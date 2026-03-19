import { parseStudioBridgeResult, type StudioBridgeResult } from "./studio-bridge-result"

export interface StudioApprovalRequest {
  message: string
  command: string | null
  workerAgentId: string | null
  bridgeResult?: StudioBridgeResult | null
}

const APPROVAL_COMMAND_PATTERNS = [
  /approve(?: the)? `([^`]+)` command/i,
  /run `([^`]+)`/i,
]

const APPROVAL_AGENT_ID_PATTERN = /\bagentId:\s*([A-Za-z0-9_-]+)/

export function parseStudioApprovalRequest(content: string): StudioApprovalRequest | null {
  const message = content.trim()
  if (!message) return null

  const bridgeResult = parseStudioBridgeResult(message)
  if (bridgeResult?.status === "approval_required" || bridgeResult?.taskKind === "approval_required") {
    const resumeHint =
      bridgeResult.payload.resume_hint && typeof bridgeResult.payload.resume_hint === "object"
        ? (bridgeResult.payload.resume_hint as Record<string, unknown>)
        : null
    const command =
      typeof bridgeResult.payload.command === "string" && bridgeResult.payload.command.trim().length > 0
        ? bridgeResult.payload.command.trim()
        : null
    const workerAgentId =
      typeof bridgeResult.payload.worker_agent_id === "string" && bridgeResult.payload.worker_agent_id.trim().length > 0
        ? bridgeResult.payload.worker_agent_id.trim()
        : typeof resumeHint?.worker_agent_id === "string" && resumeHint.worker_agent_id.trim().length > 0
          ? resumeHint.worker_agent_id.trim()
          : null

    return {
      message: bridgeResult.summary,
      command,
      workerAgentId,
      bridgeResult,
    }
  }

  const normalized = message.toLowerCase()
  if (!normalized.includes("approval") || !normalized.includes("require")) {
    return null
  }

  let command: string | null = null
  for (const pattern of APPROVAL_COMMAND_PATTERNS) {
    const match = message.match(pattern)
    if (match?.[1]?.trim()) {
      command = match[1].trim()
      break
    }
  }

  if (!command) {
    const fallbackMatches = [...message.matchAll(/`([^`\n]+)`/g)]
    const candidate = fallbackMatches
      .map((match) => match[1]?.trim() ?? "")
      .find((value) => value.length > 0 && (value.includes(" ") || value.includes("/") || value.includes("-")))
    command = candidate ?? null
  }

  const workerAgentId = message.match(APPROVAL_AGENT_ID_PATTERN)?.[1]?.trim() ?? null

  if (!command && !workerAgentId) {
    return null
  }

  return {
    message,
    command,
    workerAgentId,
    bridgeResult: null,
  }
}

export function buildStudioApprovalContinuePrompt(request: {
  command?: string | null
  workerAgentId?: string | null
  bridgeResult?: StudioBridgeResult | null
}): string {
  const lines = [
    "Continue the previous task. The previously blocked command is now approved.",
  ]

  if (request.command?.trim()) {
    lines.push(`Approved command: \`${request.command.trim()}\``)
  }

  if (request.workerAgentId?.trim()) {
    lines.push(`If the paused worker can be resumed, resume worker agentId: ${request.workerAgentId.trim()}.`)
  } else {
    lines.push("If there is a paused worker, resume it instead of starting over.")
  }

  if (request.bridgeResult) {
    lines.push("Return the final result using the same JSON bridge-result envelope schema.")
  } else {
    lines.push("Finish the task and return the final result only.")
  }
  return lines.join("\n")
}
