export interface StudioRuntimeStatusResponse {
  claude_cli?: boolean
  claude_agent_sdk?: boolean
  claude_path?: string | null
  claude_version?: string | null
  chat_surface?: string | null
  chat_transport?: string | null
  preferred_chat_transport?: string | null
  slash_commands?: string[] | null
  permission_profiles?: string[] | null
  runtime_commands?: string[] | null
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
  allowed_prefixes?: string[] | null
  allowlist_mutation_enabled?: boolean | null
  source?: string | null
  error?: string | null
}

export type StudioRuntimeSource = "unknown" | "claude_code" | "anthropic_api"
export type StudioChatSurface = "managed_session" | "unknown"
export type StudioChatTransport = "claude_agent_sdk" | "claude_cli_print" | "anthropic_api" | "unknown"

export interface StudioRuntimeInfo {
  source: StudioRuntimeSource
  chatSurface: StudioChatSurface
  chatTransport: StudioChatTransport
  preferredChatTransport: StudioChatTransport
  claudeAgentSdkAvailable: boolean
  supportedSlashCommands: string[]
  supportedPermissionProfiles: string[]
  runtimeCommands: string[]
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

function normalizeChatSurface(value: unknown): StudioChatSurface {
  return value === "managed_session" ? "managed_session" : "unknown"
}

function normalizeChatTransport(value: unknown): StudioChatTransport {
  if (
    value === "claude_agent_sdk" ||
    value === "claude_cli_print" ||
    value === "anthropic_api"
  ) {
    return value
  }
  return "unknown"
}

function formatTransportLabel(transport: StudioChatTransport): string {
  if (transport === "claude_agent_sdk") return "Agent SDK"
  if (transport === "claude_cli_print") return "CLI print"
  if (transport === "anthropic_api") return "API fallback"
  return "Unknown transport"
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
  const chatSurface = normalizeChatSurface(status?.chat_surface)
  const chatTransport = normalizeChatTransport(status?.chat_transport)
  const preferredChatTransport = normalizeChatTransport(status?.preferred_chat_transport)
  const claudeAgentSdkAvailable = status?.claude_agent_sdk === true
  const supportedSlashCommands = Array.isArray(status?.slash_commands)
    ? status.slash_commands.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const supportedPermissionProfiles = Array.isArray(status?.permission_profiles)
    ? status.permission_profiles.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const runtimeCommands = Array.isArray(status?.runtime_commands)
    ? status.runtime_commands.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const codeModeEnabled = typeof status?.code_mode_enabled === "boolean" ? status.code_mode_enabled : null
  const knownModelAliases = Array.isArray(status?.known_model_aliases)
    ? status!.known_model_aliases.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const opencodeAvailable = status?.opencode_cli === true
  const opencodePath = cleanString(status?.opencode_path)
  const opencodeVersion = cleanString(status?.opencode_version)
  const error = cleanString(status?.error) ?? cleanString(cwdPayload?.error)
  const workspaceLabel = formatRuntimePath(cwd ?? actualCwd)
  const currentTransportLabel = formatTransportLabel(chatTransport)
  const preferredTransportLabel = formatTransportLabel(preferredChatTransport)

  if (status?.claude_cli) {
    const detail =
      chatTransport === "claude_cli_print"
        ? claudeAgentSdkAvailable
          ? "Managed chat is currently running on Claude CLI print transport. Agent SDK is available for the CodePilot route."
          : "Managed chat is currently running on Claude CLI print transport. Install claude_agent_sdk to match the CodePilot route."
        : cwd
          ? `Running in ${workspaceLabel}`
          : "CLI connected to the current Studio workspace"

    return {
      source: "claude_code",
      chatSurface,
      chatTransport,
      preferredChatTransport,
      claudeAgentSdkAvailable,
      supportedSlashCommands,
      supportedPermissionProfiles,
      runtimeCommands,
      label: "Claude Code",
      statusLabel: version
        ? `Managed chat · ${currentTransportLabel} · CLI ${version}`
        : `Managed chat · ${currentTransportLabel}`,
      detail,
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
      chatSurface,
      chatTransport,
      preferredChatTransport,
      claudeAgentSdkAvailable,
      supportedSlashCommands,
      supportedPermissionProfiles,
      runtimeCommands,
      label: "Managed chat fallback",
      statusLabel: `Managed chat · ${currentTransportLabel}`,
      detail:
        error ??
        (preferredChatTransport === "claude_agent_sdk"
          ? `Claude Code is unavailable, so Studio is using direct API fallback. Preferred route remains ${preferredTransportLabel}.`
          : "Studio is falling back to the direct Anthropic API path."),
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
    chatSurface: "unknown",
    chatTransport: "unknown",
    preferredChatTransport: "claude_agent_sdk",
    claudeAgentSdkAvailable: false,
    supportedSlashCommands: [],
    supportedPermissionProfiles: ["default", "full_access"],
    runtimeCommands: [],
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
