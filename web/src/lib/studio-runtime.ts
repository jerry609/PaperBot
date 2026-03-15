export interface StudioRuntimeStatusResponse {
  claude_cli?: boolean
  claude_path?: string | null
  claude_version?: string | null
  fallback?: string | null
  error?: string | null
}

export interface StudioRuntimeCwdResponse {
  cwd?: string | null
  actual_cwd?: string | null
  home?: string | null
  source?: string | null
  error?: string | null
}

export type StudioRuntimeSource = "unknown" | "claude_code" | "anthropic_api"

export interface StudioRuntimeInfo {
  source: StudioRuntimeSource
  label: string
  statusLabel: string
  detail: string
  version: string | null
  claudePath: string | null
  cwd: string | null
  actualCwd: string | null
  workspaceLabel: string
  error: string | null
}

function cleanString(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

export function formatRuntimePath(path: string | null | undefined): string {
  const normalized = cleanString(path)
  if (!normalized) return "Workspace pending"
  if (normalized === "/") return "/"

  const segments = normalized.split("/").filter(Boolean)
  if (segments.length <= 2) return normalized
  return `.../${segments.slice(-2).join("/")}`
}

export function buildStudioRuntimeInfo(
  status?: StudioRuntimeStatusResponse | null,
  cwdPayload?: StudioRuntimeCwdResponse | null,
): StudioRuntimeInfo {
  const cwd = cleanString(cwdPayload?.cwd)
  const actualCwd = cleanString(cwdPayload?.actual_cwd)
  const version = cleanString(status?.claude_version)
  const claudePath = cleanString(status?.claude_path)
  const error = cleanString(status?.error) ?? cleanString(cwdPayload?.error)
  const workspaceLabel = formatRuntimePath(cwd ?? actualCwd)

  if (status?.claude_cli) {
    return {
      source: "claude_code",
      label: "Claude Code",
      statusLabel: version ? `CLI ${version}` : "CLI connected",
      detail: cwd ? `Running in ${workspaceLabel}` : "CLI connected to the current Studio workspace",
      version,
      claudePath,
      cwd,
      actualCwd,
      workspaceLabel,
      error,
    }
  }

  if (status || cwdPayload) {
    return {
      source: "anthropic_api",
      label: "Anthropic API fallback",
      statusLabel: "Claude Code unavailable",
      detail: error ?? "Studio is falling back to the direct Anthropic API path.",
      version,
      claudePath,
      cwd,
      actualCwd,
      workspaceLabel,
      error,
    }
  }

  return {
    source: "unknown",
    label: "Checking runtime",
    statusLabel: "Resolving Claude Code status",
    detail: "Fetching Studio runtime metadata...",
    version: null,
    claudePath: null,
    cwd: null,
    actualCwd: null,
    workspaceLabel,
    error: null,
  }
}
