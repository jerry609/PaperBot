import type { CommandRuntime } from "@/components/studio/CliCommandRunner"

export type StudioComposerMode = "Code" | "Plan" | "Ask"

export type ParsedStudioSlashCommand =
  | { kind: "help" }
  | { kind: "status" }
  | { kind: "clear" }
  | { kind: "new_thread" }
  | { kind: "mode"; mode: StudioComposerMode; remainder: string }
  | { kind: "quick_command"; runtime: CommandRuntime; presetId: string; args: string }
  | { kind: "model"; modelOption: string; customModel: string; remainder: string }

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
  agents: { runtime: "claude", presetId: "claude-agents" },
  mcp: { runtime: "claude", presetId: "claude-mcp" },
  auth: { runtime: "claude", presetId: "claude-auth" },
  doctor: { runtime: "claude", presetId: "claude-doctor" },
}

export function parseStudioSlashCommand(
  input: string,
  knownModelAliases: string[],
): ParsedStudioSlashCommand | null {
  const parsed = splitSlashInput(input)
  if (!parsed) return null

  if (parsed.command === "help") return { kind: "help" }
  if (parsed.command === "status") return { kind: "status" }
  if (parsed.command === "clear") return { kind: "clear" }
  if (parsed.command === "new") return { kind: "new_thread" }
  if (parsed.command === "plan") return { kind: "mode", mode: "Plan", remainder: parsed.args }

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
