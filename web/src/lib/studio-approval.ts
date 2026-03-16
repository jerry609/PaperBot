export interface StudioApprovalRequest {
  message: string
  command: string | null
  workerAgentId: string | null
}

const APPROVAL_COMMAND_PATTERNS = [
  /approve(?: the)? `([^`]+)` command/i,
  /run `([^`]+)`/i,
]

const APPROVAL_AGENT_ID_PATTERN = /\bagentId:\s*([A-Za-z0-9_-]+)/

export function parseStudioApprovalRequest(content: string): StudioApprovalRequest | null {
  const message = content.trim()
  if (!message) return null

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
  }
}

export function buildStudioApprovalContinuePrompt(request: {
  command?: string | null
  workerAgentId?: string | null
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

  lines.push("Finish the task and return the final result only.")
  return lines.join("\n")
}
