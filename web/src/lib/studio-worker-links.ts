import type { AgentAction, Task } from "@/lib/store/studio-store"
import {
  getStudioBridgeDelegationTaskId,
  getStudioBridgeWorkerRunId,
  type StudioBridgeResult,
} from "@/lib/studio-bridge-result"

export interface RelatedWorkerThread {
  task: Task
  matchedActions: AgentAction[]
  latestMatchedAction: AgentAction
  latestBridgeResult: StudioBridgeResult | null
  latestApprovalAction: AgentAction | null
  pendingApproval: boolean
}

function extractBridgeResult(action: AgentAction): StudioBridgeResult | null {
  if (action.type === "approval_request") {
    return action.metadata?.approvalRequest?.bridgeResult ?? null
  }
  return action.metadata?.bridgeResult ?? null
}

function matchesWorkerTarget(
  bridgeResult: StudioBridgeResult | null,
  delegationTaskId: string | null,
  workerRunId: string | null,
): boolean {
  if (!bridgeResult) return false

  const bridgeTaskId = getStudioBridgeDelegationTaskId(bridgeResult)
  if (delegationTaskId && bridgeTaskId === delegationTaskId) {
    return true
  }

  const bridgeWorkerRunId = getStudioBridgeWorkerRunId(bridgeResult)
  if (workerRunId && bridgeWorkerRunId === workerRunId) {
    return true
  }

  return false
}

function toTimestamp(action: AgentAction): number {
  return action.timestamp instanceof Date ? action.timestamp.getTime() : new Date(action.timestamp).getTime()
}

export function buildWorkerThreadPreview(task: Task): string {
  const lastAssistant = [...task.history].reverse().find((entry) => entry.role === "assistant")?.content
  if (typeof lastAssistant === "string" && lastAssistant.trim()) {
    return lastAssistant.replace(/\s+/g, " ").trim()
  }

  const lastActionText = [...task.actions]
    .reverse()
    .find((action) => action.type === "text" || action.type === "user")
    ?.content
  if (typeof lastActionText === "string" && lastActionText.trim()) {
    return lastActionText.replace(/\s+/g, " ").trim()
  }

  return task.status === "running" ? "Waiting for Claude Code..." : "No recent thread preview."
}

export function findRelatedWorkerThread(
  tasks: Task[],
  options: {
    paperId?: string | null
    delegationTaskId?: string | null
    workerRunId?: string | null
  },
): RelatedWorkerThread | null {
  const paperId = options.paperId ?? null
  const delegationTaskId = options.delegationTaskId ?? null
  const workerRunId = options.workerRunId ?? null

  let best: RelatedWorkerThread | null = null

  for (const task of tasks) {
    if (task.kind !== "chat") continue
    if (paperId && task.paperId !== paperId) continue

    const matchedActions = task.actions.filter((action) =>
      matchesWorkerTarget(extractBridgeResult(action), delegationTaskId, workerRunId),
    )
    if (matchedActions.length === 0) continue

    const latestMatchedAction = [...matchedActions].sort((left, right) => toTimestamp(right) - toTimestamp(left))[0]
    const latestApprovalAction = [...matchedActions]
      .filter((action) => action.type === "approval_request")
      .sort((left, right) => toTimestamp(right) - toTimestamp(left))[0] ?? null
    const latestBridgeResult = extractBridgeResult(latestMatchedAction)
    const pendingApproval =
      latestMatchedAction.type === "approval_request" &&
      Boolean(latestMatchedAction.metadata?.approvalRequest?.cliSessionId)

    const candidate: RelatedWorkerThread = {
      task,
      matchedActions: matchedActions.sort((left, right) => toTimestamp(right) - toTimestamp(left)),
      latestMatchedAction,
      latestBridgeResult,
      latestApprovalAction,
      pendingApproval,
    }

    if (!best || toTimestamp(candidate.latestMatchedAction) > toTimestamp(best.latestMatchedAction)) {
      best = candidate
    }
  }

  return best
}
