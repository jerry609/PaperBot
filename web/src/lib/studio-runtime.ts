export interface StudioRuntimeStatusResponse {
  claude_cli?: boolean
  claude_path?: string | null
  claude_version?: string | null
  code_mode_enabled?: boolean
  known_model_aliases?: string[] | null
  opencode_cli?: boolean
  opencode_path?: string | null
  opencode_version?: string | null
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
  codeModeEnabled: boolean | null
  knownModelAliases: string[]
  opencodeAvailable: boolean
  opencodePath: string | null
  opencodeVersion: string | null
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
  const codeModeEnabled = typeof status?.code_mode_enabled === "boolean" ? status.code_mode_enabled : null
  const knownModelAliases = Array.isArray(status?.known_model_aliases)
    ? status!.known_model_aliases.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const opencodeAvailable = status?.opencode_cli === true
  const opencodePath = cleanString(status?.opencode_path)
  const opencodeVersion = cleanString(status?.opencode_version)
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
      codeModeEnabled,
      knownModelAliases,
      opencodeAvailable,
      opencodePath,
      opencodeVersion,
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
      codeModeEnabled,
      knownModelAliases,
      opencodeAvailable,
      opencodePath,
      opencodeVersion,
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
    codeModeEnabled: null,
    knownModelAliases: [],
    opencodeAvailable: false,
    opencodePath: null,
    opencodeVersion: null,
    cwd: null,
    actualCwd: null,
    workspaceLabel,
    error: null,
  }
}
