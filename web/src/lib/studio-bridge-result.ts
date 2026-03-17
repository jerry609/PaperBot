export type StudioBridgeTaskKind =
  | "code"
  | "review"
  | "research"
  | "plan"
  | "ops"
  | "approval_required"
  | "failure"
  | "other"

export type StudioBridgeStatus =
  | "completed"
  | "partial"
  | "failed"
  | "approval_required"

export interface StudioBridgeArtifact {
  kind: string
  label: string
  path: string | null
  value: string | null
}

export interface StudioBridgeResult {
  version: string
  executor: string
  taskKind: StudioBridgeTaskKind
  status: StudioBridgeStatus
  summary: string
  artifacts: StudioBridgeArtifact[]
  payload: Record<string, unknown>
  raw: Record<string, unknown>
}

const BRIDGE_JSON_BLOCK_PATTERN = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/i

function cleanString(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function normalizeTaskKind(value: unknown): StudioBridgeTaskKind | null {
  const normalized = cleanString(value)?.toLowerCase()
  if (!normalized) return null
  if (
    normalized === "code" ||
    normalized === "review" ||
    normalized === "research" ||
    normalized === "plan" ||
    normalized === "ops" ||
    normalized === "approval_required" ||
    normalized === "failure"
  ) {
    return normalized
  }
  return "other"
}

function normalizeStatus(value: unknown): StudioBridgeStatus | null {
  const normalized = cleanString(value)?.toLowerCase()
  if (!normalized) return null
  if (
    normalized === "completed" ||
    normalized === "partial" ||
    normalized === "failed" ||
    normalized === "approval_required"
  ) {
    return normalized
  }
  return null
}

function normalizeArtifacts(value: unknown): StudioBridgeArtifact[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const label = cleanString((item as Record<string, unknown>).label)
    if (!label) return []
    return [
      {
        kind: cleanString((item as Record<string, unknown>).kind) ?? "other",
        label,
        path: cleanString((item as Record<string, unknown>).path),
        value: cleanString((item as Record<string, unknown>).value),
      },
    ]
  })
}

function normalizeBridgeResult(value: unknown): StudioBridgeResult | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null

  const raw = value as Record<string, unknown>
  const taskKind = normalizeTaskKind(raw.task_kind)
  const status = normalizeStatus(raw.status)
  const summary = cleanString(raw.summary)
  if (!taskKind || !status || !summary) return null

  return {
    version: cleanString(raw.version) ?? "1",
    executor: cleanString(raw.executor) ?? "unknown",
    taskKind,
    status,
    summary,
    artifacts: normalizeArtifacts(raw.artifacts),
    payload: raw.payload && typeof raw.payload === "object" && !Array.isArray(raw.payload)
      ? (raw.payload as Record<string, unknown>)
      : {},
    raw,
  }
}

function extractJsonCandidates(text: string): string[] {
  const trimmed = text.trim()
  const candidates: string[] = []

  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    candidates.push(trimmed)
  }

  const fenced = trimmed.match(BRIDGE_JSON_BLOCK_PATTERN)?.[1]?.trim()
  if (fenced) {
    candidates.push(fenced)
  }

  const firstBrace = trimmed.indexOf("{")
  const lastBrace = trimmed.lastIndexOf("}")
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    candidates.push(trimmed.slice(firstBrace, lastBrace + 1).trim())
  }

  return [...new Set(candidates)]
}

export function parseStudioBridgeResult(value: unknown): StudioBridgeResult | null {
  const direct = normalizeBridgeResult(value)
  if (direct) return direct

  if (typeof value !== "string") return null
  for (const candidate of extractJsonCandidates(value)) {
    try {
      const parsed = JSON.parse(candidate) as unknown
      const normalized = normalizeBridgeResult(parsed)
      if (normalized) return normalized
    } catch {
      continue
    }
  }
  return null
}
