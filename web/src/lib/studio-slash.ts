import type { CommandRuntime } from "@/components/studio/CliCommandRunner"

export type StudioComposerMode = "Code" | "Plan" | "Ask"
export type StudioEffortOption = "default" | "low" | "medium" | "high" | "max"

export type ParsedStudioSlashCommand =
  | { kind: "mode"; mode: StudioComposerMode; remainder: string }
  | { kind: "view"; view: "log" | "context" | "board" }
  | { kind: "quick_command"; runtime: CommandRuntime; presetId: string; args: string }
  | { kind: "continue"; enabled: boolean; remainder: string }
  | { kind: "model"; modelOption: string; customModel: string; remainder: string }
  | { kind: "effort"; effort: StudioEffortOption; remainder: string }
  | { kind: "resume"; value: string }
  | { kind: "session"; value: string }
  | { kind: "agent"; value: string }
  | { kind: "tools"; value: string }
  | { kind: "allowed_tools"; value: string }
  | { kind: "add_dirs"; value: string }
  | { kind: "mcp_config"; value: string }
  | { kind: "settings"; value: string }

type SlashInput = {
  command: string
  args: string
}

function splitSlashInput(input: string): SlashInput | null {
  const trimmed = input.trim()
  if (!trimmed.startsWith("/")) return null

  const body = trimmed.slice(1).trim()
  if (!body) return null

  const whitespaceIndex = body.search(/\s/)
  if (whitespaceIndex === -1) {
    return {
      command: body.toLowerCase(),
      args: "",
    }
  }

  return {
    command: body.slice(0, whitespaceIndex).toLowerCase(),
    args: body.slice(whitespaceIndex + 1).trim(),
  }
}

function splitValueAndRemainder(args: string): { value: string; remainder: string } {
  const trimmed = args.trim()
  if (!trimmed) {
    return { value: "", remainder: "" }
  }

  const whitespaceIndex = trimmed.search(/\s/)
  if (whitespaceIndex === -1) {
    return { value: trimmed, remainder: "" }
  }

  return {
    value: trimmed.slice(0, whitespaceIndex),
    remainder: trimmed.slice(whitespaceIndex + 1).trim(),
  }
}

const QUICK_COMMAND_MAP: Record<string, { runtime: CommandRuntime; presetId: string }> = {
  "claude-agents": { runtime: "claude", presetId: "claude-agents" },
  "claude-mcp": { runtime: "claude", presetId: "claude-mcp" },
  "claude-auth": { runtime: "claude", presetId: "claude-auth" },
  "opencode-models": { runtime: "opencode", presetId: "opencode-models" },
  "opencode-agent": { runtime: "opencode", presetId: "opencode-agent" },
  "opencode-mcp": { runtime: "opencode", presetId: "opencode-mcp" },
  "opencode-providers": { runtime: "opencode", presetId: "opencode-providers" },
}

export function parseStudioSlashCommand(
  input: string,
  knownModelAliases: string[],
): ParsedStudioSlashCommand | null {
  const parsed = splitSlashInput(input)
  if (!parsed) return null

  if (parsed.command === "code") return { kind: "mode", mode: "Code", remainder: parsed.args }
  if (parsed.command === "plan") return { kind: "mode", mode: "Plan", remainder: parsed.args }
  if (parsed.command === "ask") return { kind: "mode", mode: "Ask", remainder: parsed.args }
  if (parsed.command === "chat") return { kind: "view", view: "log" }
  if (parsed.command === "context") return { kind: "view", view: "context" }
  if (parsed.command === "monitor") return { kind: "view", view: "board" }
  if (parsed.command === "continue") return { kind: "continue", enabled: true, remainder: parsed.args }
  if (parsed.command === "fresh") return { kind: "continue", enabled: false, remainder: parsed.args }

  if (parsed.command === "model") {
    const { value, remainder } = splitValueAndRemainder(parsed.args)
    if (!value) return null
    if (knownModelAliases.includes(value)) {
      return { kind: "model", modelOption: value, customModel: "", remainder }
    }
    return { kind: "model", modelOption: "custom", customModel: value, remainder }
  }

  if (parsed.command.startsWith("model-")) {
    const alias = parsed.command.slice("model-".length)
    if (!alias) return null
    if (knownModelAliases.includes(alias)) {
      return { kind: "model", modelOption: alias, customModel: "", remainder: parsed.args }
    }
    return { kind: "model", modelOption: "custom", customModel: alias, remainder: parsed.args }
  }

  if (parsed.command === "effort") {
    const { value, remainder } = splitValueAndRemainder(parsed.args)
    if (
      value === "default" ||
      value === "low" ||
      value === "medium" ||
      value === "high" ||
      value === "max"
    ) {
      return { kind: "effort", effort: value, remainder }
    }
    return null
  }

  if (parsed.command.startsWith("effort-")) {
    const effort = parsed.command.slice("effort-".length)
    if (
      effort === "default" ||
      effort === "low" ||
      effort === "medium" ||
      effort === "high" ||
      effort === "max"
    ) {
      return { kind: "effort", effort, remainder: parsed.args }
    }
    return null
  }

  if (parsed.command === "resume") {
    return { kind: "resume", value: parsed.args }
  }

  if (parsed.command === "session" || parsed.command === "session-id") {
    return { kind: "session", value: parsed.args }
  }

  if (parsed.command === "agent") {
    return { kind: "agent", value: parsed.args }
  }

  if (parsed.command === "tools") {
    return { kind: "tools", value: parsed.args }
  }

  if (parsed.command === "allow" || parsed.command === "allowed-tools") {
    return { kind: "allowed_tools", value: parsed.args }
  }

  if (parsed.command === "add-dir" || parsed.command === "dir") {
    return { kind: "add_dirs", value: parsed.args }
  }

  if (parsed.command === "mcp") {
    return { kind: "mcp_config", value: parsed.args }
  }

  if (parsed.command === "settings") {
    return { kind: "settings", value: parsed.args }
  }

  const quickCommand = QUICK_COMMAND_MAP[parsed.command]
  if (quickCommand) {
    return {
      kind: "quick_command",
      runtime: quickCommand.runtime,
      presetId: quickCommand.presetId,
      args: parsed.args,
    }
  }

  return null
}
