"use client"

import { useEffect, useMemo, useState } from "react"
import { TerminalSquare, Play, RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { StudioRuntimeInfo } from "@/lib/studio-runtime"
import { cn } from "@/lib/utils"

type CommandRuntime = "claude" | "opencode"

type CommandPreset = {
  id: string
  label: string
  command: string
  defaultArgs: string
  helpText: string
}

type CommandResult = {
  ok: boolean
  command: string[]
  returncode: number
  stdout: string
  stderr: string
  cwd?: string | null
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

function presetsForRuntime(runtime: CommandRuntime): CommandPreset[] {
  return runtime === "claude" ? CLAUDE_PRESETS : OPENCODE_PRESETS
}

interface CliCommandRunnerProps {
  runtimeInfo: StudioRuntimeInfo
  projectDir: string | null
}

export function CliCommandRunner({ runtimeInfo, projectDir }: CliCommandRunnerProps) {
  const [runtime, setRuntime] = useState<CommandRuntime>("claude")
  const presets = useMemo(() => presetsForRuntime(runtime), [runtime])
  const [presetId, setPresetId] = useState<string>(presets[0]?.id ?? "")
  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === presetId) ?? presets[0] ?? null,
    [presetId, presets],
  )
  const [args, setArgs] = useState<string>(selectedPreset?.defaultArgs ?? "")
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<CommandResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (runtime === "claude" && runtimeInfo.source !== "claude_code" && runtimeInfo.opencodeAvailable) {
      setRuntime("opencode")
    }
  }, [runtime, runtimeInfo.opencodeAvailable, runtimeInfo.source])

  useEffect(() => {
    if (presets.some((preset) => preset.id === presetId)) return
    setPresetId(presets[0]?.id ?? "")
  }, [presetId, presets])

  useEffect(() => {
    if (!selectedPreset) return
    setArgs(selectedPreset.defaultArgs)
  }, [selectedPreset])

  const runtimeUnavailable =
    runtime === "claude"
      ? runtimeInfo.source !== "claude_code"
      : !runtimeInfo.opencodeAvailable

  const runtimeLabel =
    runtime === "claude"
      ? runtimeInfo.label
      : runtimeInfo.opencodeAvailable
        ? `OpenCode ${runtimeInfo.opencodeVersion ?? ""}`.trim()
        : "OpenCode unavailable"

  const commandPreview = useMemo(() => {
    if (!selectedPreset) return ""
    const parts = [runtime === "claude" ? "claude" : "opencode", selectedPreset.command]
    if (args.trim()) parts.push(args.trim())
    return parts.join(" ")
  }, [args, runtime, selectedPreset])

  const handleRun = async () => {
    if (!selectedPreset || runtimeUnavailable || running) return

    setRunning(true)
    setError(null)

    try {
      const response = await fetch("/api/studio/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          runtime,
          command: selectedPreset.command,
          args,
          project_dir: projectDir ?? undefined,
        }),
      })

      const payload = (await response.json()) as CommandResult
      setResult(payload)
      if (!payload.ok) {
        setError(payload.stderr || `Command failed (${payload.returncode})`)
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to run command"
      setError(message)
      setResult(null)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f5f5f2]">
      <div className="border-b border-slate-200 bg-[#eef0ea] px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <TerminalSquare className="h-4 w-4 text-slate-600" />
              <h3 className="text-sm font-semibold text-slate-900">Claude Code Command Runner</h3>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Run Claude Code and OpenCode management commands that do not belong in the print-mode chat surface.
            </p>
          </div>
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
      </div>

      <div className="border-b border-slate-200 bg-[#f2f3ef] px-4 py-3">
        <div className="grid gap-3 md:grid-cols-[160px_220px_1fr_auto]">
          <Select
            value={runtime}
            onValueChange={(value) => {
              const nextRuntime = value as CommandRuntime
              const nextPresets = presetsForRuntime(nextRuntime)
              setRuntime(nextRuntime)
              setPresetId(nextPresets[0]?.id ?? "")
            }}
          >
            <SelectTrigger className="h-9 border-slate-200 bg-white text-xs text-slate-700">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="claude">Claude Code</SelectItem>
              <SelectItem value="opencode">OpenCode</SelectItem>
            </SelectContent>
          </Select>

          <Select value={presetId} onValueChange={setPresetId}>
            <SelectTrigger className="h-9 border-slate-200 bg-white text-xs text-slate-700">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {presets.map((preset) => (
                <SelectItem key={preset.id} value={preset.id}>
                  {preset.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            value={args}
            onChange={(event) => setArgs(event.target.value)}
            className="h-9 border-slate-200 bg-white text-xs text-slate-700"
            placeholder="Extra args, e.g. list --json"
          />

          <Button
            className="h-9 gap-1.5"
            onClick={handleRun}
            disabled={running || runtimeUnavailable || !selectedPreset}
          >
            {running ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Run
          </Button>
        </div>

        {selectedPreset ? (
          <div className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div className="font-mono text-[11px] text-slate-700">{commandPreview}</div>
            <div className="mt-1 text-[11px] text-slate-500">{selectedPreset.helpText}</div>
            <div className="mt-1 text-[11px] text-slate-400">
              CWD: {projectDir ?? runtimeInfo.cwd ?? runtimeInfo.actualCwd ?? "runtime default"}
            </div>
          </div>
        ) : null}
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-3 p-4">
          {error ? (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {error}
            </div>
          ) : null}

          {result ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                <div className="text-[11px] font-medium text-slate-500">Executed</div>
                <div className="mt-1 font-mono text-[11px] text-slate-800">
                  {result.command.join(" ")}
                </div>
                <div className="mt-1 text-[11px] text-slate-500">
                  exit {result.returncode} · cwd {result.cwd ?? "n/a"}
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white">
                <div className="border-b border-slate-200 px-3 py-2 text-[11px] font-medium text-slate-500">
                  STDOUT
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap px-3 py-3 text-[12px] leading-5 text-slate-800">
                  {result.stdout || "(no stdout)"}
                </pre>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white">
                <div className="border-b border-slate-200 px-3 py-2 text-[11px] font-medium text-slate-500">
                  STDERR
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap px-3 py-3 text-[12px] leading-5 text-slate-700">
                  {result.stderr || "(no stderr)"}
                </pre>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-300 bg-white/70 px-4 py-6 text-sm text-slate-500">
              Pick a runtime preset and run it here instead of overloading the chat input with management commands.
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
