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
  skills?: StudioSkillPayload[] | null
  code_mode_enabled?: boolean
  known_model_aliases?: string[] | null
  detected_default_model?: string | null
  detected_default_model_source?: string | null
  project_agents?: string[] | null
  project_agent_count?: number | null
  claude_agents_error?: string | null
  codex_worker_available?: boolean
  codex_worker_name?: string | null
  opencode_worker_available?: boolean
  opencode_worker_name?: string | null
  opencode_cli?: boolean
  opencode_path?: string | null
  opencode_version?: string | null
  fallback?: string | null
  error?: string | null
}

export interface StudioSkillPayload {
  id?: string | null
  title?: string | null
  description?: string | null
  slash_command?: string | null
  scope?: string | null
  tools?: string[] | null
  recommended_for?: string[] | null
  ecosystems?: string[] | null
  primary_ecosystem?: string | null
  paths?: string[] | null
  manifest_source?: string | null
  path?: string | null
  prompt_hint?: string | null
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

export interface StudioSkillInfo {
  id: string
  title: string
  description: string
  slashCommand: string
  scope: string
  tools: string[]
  recommendedFor: string[]
  ecosystems: string[]
  primaryEcosystem: string | null
  paths: string[]
  manifestSource: string | null
  path: string | null
  promptHint: string | null
}

export interface StudioRuntimeInfo {
  source: StudioRuntimeSource
  chatSurface: StudioChatSurface
  chatTransport: StudioChatTransport
  preferredChatTransport: StudioChatTransport
  claudeAgentSdkAvailable: boolean
  supportedSlashCommands: string[]
  supportedPermissionProfiles: string[]
  runtimeCommands: string[]
  skills: StudioSkillInfo[]
  label: string
  statusLabel: string
  detail: string
  version: string | null
  claudePath: string | null
  codeModeEnabled: boolean | null
  knownModelAliases: string[]
  detectedDefaultModel: string | null
  detectedDefaultModelSource: string | null
  projectAgents: string[]
  projectAgentCount: number
  claudeAgentsError: string | null
  codexWorkerAvailable: boolean
  codexWorkerName: string | null
  opencodeWorkerAvailable: boolean
  opencodeWorkerName: string | null
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

function cleanStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []

  const normalized: string[] = []
  const seen = new Set<string>()
  for (const item of value) {
    if (typeof item !== "string") continue
    const cleaned = item.trim()
    if (!cleaned || seen.has(cleaned)) continue
    seen.add(cleaned)
    normalized.push(cleaned)
  }
  return normalized
}

function formatTransportLabel(transport: StudioChatTransport): string {
  if (transport === "claude_agent_sdk") return "Agent SDK"
  if (transport === "claude_cli_print") return "CLI print"
  if (transport === "anthropic_api") return "API fallback"
  return "Managed transport"
}

function normalizeStudioSkills(value: unknown): StudioSkillInfo[] {
  if (!Array.isArray(value)) return []

  const normalized: StudioSkillInfo[] = []
  const seen = new Set<string>()

  for (const item of value) {
    if (!item || typeof item !== "object") continue
    const payload = item as StudioSkillPayload
    const id = cleanString(payload.id)
    const title = cleanString(payload.title)
    const slashCommand = cleanString(payload.slash_command)
    const path = cleanString(payload.path)
    const primaryEcosystem = cleanString(payload.primary_ecosystem)
    const paths = cleanStringList(payload.paths)
    if (!id || !title || !slashCommand) continue

    const key = `${id}:${slashCommand}`
    if (seen.has(key)) continue
    seen.add(key)

    normalized.push({
      id,
      title,
      description: cleanString(payload.description) ?? "",
      slashCommand,
      scope: cleanString(payload.scope) ?? "project",
      tools: cleanStringList(payload.tools),
      recommendedFor: cleanStringList(payload.recommended_for),
      ecosystems: cleanStringList(payload.ecosystems),
      primaryEcosystem,
      paths: paths.length > 0 ? paths : path ? [path] : [],
      manifestSource: cleanString(payload.manifest_source),
      path,
      promptHint: cleanString(payload.prompt_hint),
    })
  }

  return normalized
}

export function formatRuntimePath(path: string | null | undefined): string {
  const normalized = cleanString(path)
  if (!normalized) return "Workspace pending"
  if (normalized === "/") return "/"

  const segments = normalized.split("/").filter(Boolean)
  if (segments.length <= 2) return normalized
  return `.../${segments.slice(-2).join("/")}`
}

export function resolveDetectedModelSelection(
  detectedModel: string | null | undefined,
  knownModelAliases: string[],
): { modelOption: string; customModel: string } | null {
  const normalized = cleanString(detectedModel)
  if (!normalized) return null
  return knownModelAliases.includes(normalized)
    ? { modelOption: normalized, customModel: "" }
    : { modelOption: "custom", customModel: normalized }
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
  const skills = normalizeStudioSkills(status?.skills)
  const codeModeEnabled = typeof status?.code_mode_enabled === "boolean" ? status.code_mode_enabled : null
  const knownModelAliases = Array.isArray(status?.known_model_aliases)
    ? status!.known_model_aliases.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const detectedDefaultModel = cleanString(status?.detected_default_model)
  const detectedDefaultModelSource = cleanString(status?.detected_default_model_source)
  const projectAgents = Array.isArray(status?.project_agents)
    ? status.project_agents.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : []
  const projectAgentCount =
    typeof status?.project_agent_count === "number" && Number.isFinite(status.project_agent_count)
      ? status.project_agent_count
      : projectAgents.length
  const claudeAgentsError = cleanString(status?.claude_agents_error)
  const codexWorkerAvailable = status?.codex_worker_available === true
  const codexWorkerName = cleanString(status?.codex_worker_name)
  const opencodeWorkerAvailable = status?.opencode_worker_available === true
  const opencodeWorkerName = cleanString(status?.opencode_worker_name)
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
      skills,
      label: "Claude Code",
      statusLabel: version
        ? `Managed chat · ${currentTransportLabel} · CLI ${version}`
        : `Managed chat · ${currentTransportLabel}`,
      detail,
      version,
      claudePath,
      codeModeEnabled,
      knownModelAliases,
      detectedDefaultModel,
      detectedDefaultModelSource,
      projectAgents,
      projectAgentCount,
      claudeAgentsError,
      codexWorkerAvailable,
      codexWorkerName,
      opencodeWorkerAvailable,
      opencodeWorkerName,
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
      skills,
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
      detectedDefaultModel,
      detectedDefaultModelSource,
      projectAgents,
      projectAgentCount,
      claudeAgentsError,
      codexWorkerAvailable,
      codexWorkerName,
      opencodeWorkerAvailable,
      opencodeWorkerName,
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
    skills: [],
    label: "Checking runtime",
    statusLabel: "Resolving Claude Code status",
    detail: "Fetching Studio runtime metadata...",
    version: null,
    claudePath: null,
    codeModeEnabled: null,
    knownModelAliases: [],
    detectedDefaultModel: null,
    detectedDefaultModelSource: null,
    projectAgents: [],
    projectAgentCount: 0,
    claudeAgentsError: null,
    codexWorkerAvailable: false,
    codexWorkerName: null,
    opencodeWorkerAvailable: false,
    opencodeWorkerName: null,
    opencodeAvailable: false,
    opencodePath: null,
    opencodeVersion: null,
    cwd: null,
    actualCwd: null,
    workspaceLabel,
    error: null,
  }
}
