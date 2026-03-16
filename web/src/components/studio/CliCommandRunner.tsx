"use client"

import { useMemo, useState } from "react"
import { TerminalSquare, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { StudioRuntimeInfo } from "@/lib/studio-runtime"
import { cn } from "@/lib/utils"

export type CommandRuntime = "claude" | "opencode"

export type CommandPreset = {
  id: string
  label: string
  command: string
  defaultArgs: string
  helpText: string
}

export type CommandResult = {
  ok: boolean
  command: string[]
  returncode: number
  stdout: string
  stderr: string
  cwd?: string | null
}

export type ActiveCommand = {
  runtime: CommandRuntime
  preset: CommandPreset
}

export type LastCommandOutput = {
  preview: string
  result: CommandResult | null
  error: string | null
}

const CLAUDE_PRESETS: CommandPreset[] = [
  {
    id: "claude-agents",
    label: "claude agents",
    command: "agents",
    defaultArgs: "",
    helpText: "List configured Claude Code agents in the current environment.",
  },
  {
    id: "claude-mcp",
    label: "claude mcp list",
    command: "mcp",
    defaultArgs: "list",
    helpText: "List configured Claude Code MCP servers.",
  },
  {
    id: "claude-auth",
    label: "claude auth status",
    command: "auth",
    defaultArgs: "status",
    helpText: "Inspect Claude Code authentication state without entering chat mode.",
  },
]

const OPENCODE_PRESETS: CommandPreset[] = [
  {
    id: "opencode-models",
    label: "opencode models",
    command: "models",
    defaultArgs: "",
    helpText: "List available OpenCode models.",
  },
  {
    id: "opencode-agent",
    label: "opencode agent list",
    command: "agent",
    defaultArgs: "list",
    helpText: "List available OpenCode agents.",
  },
  {
    id: "opencode-mcp",
    label: "opencode mcp list",
    command: "mcp",
    defaultArgs: "list",
    helpText: "List OpenCode MCP servers and status.",
  },
  {
    id: "opencode-providers",
    label: "opencode providers list",
    command: "providers",
    defaultArgs: "list",
    helpText: "Show configured OpenCode providers and login state.",
  },
]

export function getCommandPresets(runtime: CommandRuntime): CommandPreset[] {
  return runtime === "claude" ? CLAUDE_PRESETS : OPENCODE_PRESETS
}

export function buildCommandPreview(
  runtime: CommandRuntime,
  preset: CommandPreset,
  args: string,
): string {
  const parts = [runtime === "claude" ? "claude" : "opencode", preset.command]
  if (args.trim()) parts.push(args.trim())
  return parts.join(" ")
}

interface CliCommandRunnerProps {
  runtimeInfo: StudioRuntimeInfo
  activeCommand: ActiveCommand | null
  activeArgs?: string
  defaultOpen?: boolean
  showActiveBadge?: boolean
  onSelectCommand: (command: ActiveCommand) => void
  onClearCommand: () => void
}

export function CliCommandRunner({
  runtimeInfo,
  activeCommand,
  activeArgs = "",
  defaultOpen = false,
  showActiveBadge = true,
  onSelectCommand,
  onClearCommand,
}: CliCommandRunnerProps) {
  const [popoverOpen, setPopoverOpen] = useState(defaultOpen)
  const [draftRuntime, setDraftRuntime] = useState<CommandRuntime>(() =>
    activeCommand?.runtime ??
      (runtimeInfo.source === "claude_code" ? "claude" : runtimeInfo.opencodeAvailable ? "opencode" : "claude"),
  )
  const effectiveRuntime =
    draftRuntime === "claude" && runtimeInfo.source !== "claude_code" && runtimeInfo.opencodeAvailable
      ? "opencode"
      : draftRuntime
  const presets = useMemo(() => getCommandPresets(effectiveRuntime), [effectiveRuntime])
  const [draftPresetId, setDraftPresetId] = useState<string>(() => activeCommand?.preset.id ?? "")
  const presetId = presets.some((preset) => preset.id === draftPresetId)
    ? draftPresetId
    : presets[0]?.id ?? ""
  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === presetId) ?? presets[0] ?? null,
    [presetId, presets],
  )

  const runtimeUnavailable =
    effectiveRuntime === "claude"
      ? runtimeInfo.source !== "claude_code"
      : !runtimeInfo.opencodeAvailable

  const runtimeLabel =
    effectiveRuntime === "claude"
      ? runtimeInfo.label
      : runtimeInfo.opencodeAvailable
        ? `OpenCode ${runtimeInfo.opencodeVersion ?? ""}`.trim()
        : "OpenCode unavailable"

  const draftArgs =
    activeCommand?.runtime === effectiveRuntime && activeCommand.preset.id === selectedPreset?.id
      ? activeArgs
      : selectedPreset?.defaultArgs ?? ""

  const draftPreview = useMemo(() => {
    if (!selectedPreset) return ""
    return buildCommandPreview(effectiveRuntime, selectedPreset, draftArgs)
  }, [draftArgs, effectiveRuntime, selectedPreset])

  const activePreview = activeCommand
    ? buildCommandPreview(activeCommand.runtime, activeCommand.preset, activeArgs)
    : ""

  const handleUseCommand = () => {
    if (!selectedPreset || runtimeUnavailable) return
    onSelectCommand({
      runtime: effectiveRuntime,
      preset: selectedPreset,
    })
    setPopoverOpen(false)
  }

  return (
    <div className="flex max-w-full items-center gap-2">
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-8 w-8 shrink-0 rounded-md border border-transparent bg-transparent text-slate-500 hover:bg-[#e7e9e3] hover:text-slate-900",
              (popoverOpen || activeCommand) && "border-slate-200 bg-[#e7e9e3] text-slate-900",
            )}
            title="Quick commands"
          >
            <TerminalSquare className="h-3.5 w-3.5" />
          </Button>
        </PopoverTrigger>

        <PopoverContent
          align="start"
          side="top"
          sideOffset={10}
          className="w-[320px] border-slate-200 bg-[#f7f7f4] p-0 text-slate-900 shadow-[0_18px_40px_rgba(15,23,42,0.10)]"
        >
          <div className="border-b border-slate-200 bg-[#f1f3ed] px-3 py-2.5">
            <div className="flex items-start gap-2">
              <TerminalSquare className="mt-0.5 h-3.5 w-3.5 text-slate-500" />
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                  Quick Commands
                </div>
                <div className="mt-1 text-[12px] text-slate-700">
                  Pick a safe Claude Code or OpenCode utility command.
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3 px-3 py-3">
            <div className="flex items-center justify-between gap-3">
              <Select
                value={effectiveRuntime}
                onValueChange={(value) => {
                  const nextRuntime = value as CommandRuntime
                  const nextPresets = getCommandPresets(nextRuntime)
                  setDraftRuntime(nextRuntime)
                  setDraftPresetId(nextPresets[0]?.id ?? "")
                }}
              >
                <SelectTrigger className="h-8 border-slate-200 bg-white text-xs text-slate-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="claude">Claude Code</SelectItem>
                  <SelectItem value="opencode">OpenCode</SelectItem>
                </SelectContent>
              </Select>

              <span
                className={cn(
                  "rounded-full border px-2 py-1 text-[10px] font-medium",
                  runtimeUnavailable
                    ? "border-amber-200 bg-amber-50 text-amber-700"
                    : "border-emerald-200 bg-emerald-50 text-emerald-700",
                )}
              >
                {runtimeLabel}
              </span>
            </div>

            <ScrollArea className="max-h-44 rounded-lg border border-slate-200 bg-white">
              <div className="space-y-1 p-1">
                {presets.map((preset) => {
                  const selected = preset.id === presetId
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      className={cn(
                        "w-full rounded-md border px-3 py-2 text-left transition-colors",
                        selected
                          ? "border-slate-300 bg-[#f1f3ed]"
                          : "border-transparent bg-transparent hover:border-slate-200 hover:bg-[#f8f8f5]",
                      )}
                      onClick={() => setDraftPresetId(preset.id)}
                    >
                      <div className="font-mono text-[11px] text-slate-900">{preset.label}</div>
                      <div className="mt-1 line-clamp-2 text-[11px] leading-5 text-slate-500">
                        {preset.helpText}
                      </div>
                    </button>
                  )
                })}
              </div>
            </ScrollArea>

            <div className="rounded-lg border border-slate-200 bg-[#f1f3ed] px-3 py-2">
              <div className="text-[10px] font-medium uppercase tracking-[0.16em] text-slate-400">
                Preview
              </div>
              <div className="mt-1 font-mono text-[11px] text-slate-700">
                {draftPreview || "Pick a command"}
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                The main input box becomes the argument line.
              </div>
            </div>

            <div className="flex items-center justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-xs text-slate-600 hover:bg-[#e7e9e3] hover:text-slate-900"
                onClick={() => setPopoverOpen(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                className="h-8 text-xs"
                onClick={handleUseCommand}
                disabled={!selectedPreset || runtimeUnavailable}
              >
                Use
              </Button>
            </div>
          </div>
        </PopoverContent>
      </Popover>

      {activeCommand && showActiveBadge ? (
        <div className="inline-flex min-w-0 max-w-[260px] items-center gap-1.5 rounded-full border border-slate-200 bg-[#eef0ea] px-2.5 py-1 text-xs text-slate-700">
          <TerminalSquare className="h-3.5 w-3.5 shrink-0 text-slate-500" />
          <span className="truncate font-mono">{activePreview}</span>
          <button
            type="button"
            className="rounded-full p-1 text-slate-500 transition-colors hover:bg-white hover:text-slate-900"
            onClick={onClearCommand}
            title="Return to chat mode"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : null}
    </div>
  )
}
