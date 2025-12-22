"use client"

import { useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { GenCodeResult, useStudioStore } from "@/lib/store/studio-store"
import { readSSE } from "@/lib/sse"
import { Play, Sparkles, CheckCircle2, AlertCircle, Server, Laptop } from "lucide-react"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

type StepStatus = "idle" | "running" | "success" | "error"
type Executor = "docker" | "e2b"

function StatusBadge({ status }: { status: StepStatus }) {
  const props = useMemo(() => {
    if (status === "running") return { label: "running", className: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300" }
    if (status === "success") return { label: "success", className: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300" }
    if (status === "error") return { label: "error", className: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" }
    return { label: "idle", className: "bg-muted text-muted-foreground" }
  }, [status])

  return (
    <Badge variant="secondary" className={props.className}>
      {props.label}
    </Badge>
  )
}

export function RunbookPanel() {
  const { paperDraft, lastGenCodeResult, addTask, addAction, updateTaskStatus, setLastGenCodeResult } = useStudioStore()
  const [status, setStatus] = useState<StepStatus>("idle")
  const [runId, setRunId] = useState<string | null>(null)
  const runIdRef = useRef<string | null>(null)
  const [executor, setExecutor] = useState<Executor>("docker")
  const [allowNetwork, setAllowNetwork] = useState(false)

  const canRun = paperDraft.title.trim().length > 0 && paperDraft.abstract.trim().length > 0
  const projectDir = lastGenCodeResult?.outputDir || null

  const runPaper2Code = async () => {
    if (!canRun || status === "running") return

    setStatus("running")
    const taskId = addTask(`Runbook: Paper2Code — ${paperDraft.title.slice(0, 40)}${paperDraft.title.length > 40 ? "…" : ""}`)
    addAction(taskId, { type: "thinking", content: "Starting Paper2Code run…" })
    runIdRef.current = null
    setRunId(null)

    try {
      const res = await fetch("/api/gen-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: paperDraft.title,
          abstract: paperDraft.abstract,
          method_section: paperDraft.methodSection || undefined,
          use_orchestrator: true,
          use_rag: true,
        }),
      })

      if (!res.ok || !res.body) {
        throw new Error(`Failed to start run (${res.status})`)
      }

      updateTaskStatus(taskId, "running")

      for await (const evt of readSSE(res.body)) {
        if (evt?.type === "progress") {
          const data = (evt.data ?? {}) as { phase?: string; message?: string; run_id?: string }
          if (data.run_id && !runIdRef.current) {
            runIdRef.current = data.run_id
            setRunId(data.run_id)
          }
          addAction(taskId, {
            type: "thinking",
            content: `${data.phase ? `[${data.phase}] ` : ""}${data.message || "Working…"}`,
          })
        } else if (evt?.type === "result") {
          setLastGenCodeResult(evt.data as GenCodeResult)
          addAction(taskId, { type: "complete", content: "Run completed" })
          updateTaskStatus(taskId, "completed")
          setStatus("success")
        } else if (evt?.type === "error") {
          addAction(taskId, { type: "error", content: evt.message || "Run failed" })
          updateTaskStatus(taskId, "error")
          setStatus("error")
        }
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      addAction(taskId, { type: "error", content: message })
      updateTaskStatus(taskId, "error")
      setStatus("error")
    }
  }

  const streamRunLogsToTimeline = async (runId: string, taskId: string) => {
    const res = await fetch(`/api/sandbox/runs/${encodeURIComponent(runId)}/logs/stream`, {
      headers: { Accept: "text/event-stream" },
    })
    if (!res.ok || !res.body) {
      addAction(taskId, { type: "error", content: `Failed to stream logs (${res.status})` })
      return
    }

    for await (const evt of readSSE(res.body)) {
      if (evt?.type === "log") {
        const data = (evt.data ?? {}) as { level?: string; message?: string; source?: string }
        const level = (data.level || "info").toLowerCase()
        const message = data.message || ""
        if (!message) continue
        addAction(taskId, { type: level === "error" ? "error" : "text", content: message })
      }
    }
  }

  const runSmoke = async () => {
    if (!projectDir || status === "running") return

    setStatus("running")
    const taskId = addTask(`Runbook: Smoke — ${executor} — ${projectDir.split("/").slice(-1)[0]}`)
    addAction(taskId, { type: "thinking", content: `Starting smoke on ${executor}…` })

    try {
      const res = await fetch("/api/runbook/smoke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_dir: projectDir,
          executor,
          allow_network: allowNetwork,
        }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to start smoke (${res.status}): ${text}`)
      }
      const data = (await res.json()) as { run_id: string }
      addAction(taskId, { type: "thinking", content: `Smoke run_id: ${data.run_id}` })

      await streamRunLogsToTimeline(data.run_id, taskId)

      const statusRes = await fetch(`/api/runbook/runs/${encodeURIComponent(data.run_id)}`)
      if (statusRes.ok) {
        const info = (await statusRes.json()) as { status: string; exit_code?: number; error?: string }
        if (info.status === "success") {
          updateTaskStatus(taskId, "completed")
          addAction(taskId, { type: "complete", content: `Smoke succeeded (exit_code=${info.exit_code ?? 0})` })
          setStatus("success")
        } else {
          updateTaskStatus(taskId, "error")
          addAction(taskId, { type: "error", content: info.error || `Smoke finished with status: ${info.status}` })
          setStatus("error")
        }
      } else {
        updateTaskStatus(taskId, "completed")
        setStatus("success")
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      updateTaskStatus(taskId, "error")
      addAction(taskId, { type: "error", content: message })
      setStatus("error")
    }
  }

  const executorLabel = executor === "docker" ? "Local Docker" : "E2B (Remote)"

  return (
    <div className="h-full flex flex-col min-w-0 min-h-0 bg-muted/5">
      <div className="border-b px-4 py-3 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center justify-between">
        <div className="min-w-0">
          <div className="text-sm font-semibold flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-purple-500" /> Runbook
          </div>
          <div className="text-xs text-muted-foreground truncate">Executable steps that produce evidence.</div>
        </div>
        <StatusBadge status={status} />
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="p-3 space-y-3">
          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Execution Backend</CardTitle>
              <CardDescription className="text-xs">Choose where Runbook steps execute.</CardDescription>
              <CardAction>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8 text-xs">
                      {executorLabel}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56">
                    <DropdownMenuItem onClick={() => setExecutor("docker")} className="flex items-center gap-2">
                      <Laptop className="h-4 w-4 text-muted-foreground" />
                      Local Docker
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setExecutor("e2b")} className="flex items-center gap-2">
                      <Server className="h-4 w-4 text-muted-foreground" />
                      E2B (Remote)
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </CardAction>
            </CardHeader>
            <CardContent className="px-4 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <Label className="text-xs">Allow Network</Label>
                <div className="text-xs text-muted-foreground">
                  Docker blocks network by default; enable for <span className="font-mono">pip install</span>.
                </div>
              </div>
              <Switch checked={allowNetwork} onCheckedChange={setAllowNetwork} />
            </CardContent>
          </Card>

          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Paper2Code</CardTitle>
              <CardDescription className="text-xs">Generate project skeleton and run verification.</CardDescription>
              <CardAction>
                <Button
                  size="sm"
                  className="h-8"
                  onClick={runPaper2Code}
                  disabled={!canRun || status === "running"}
                >
                  <Play className="h-3.5 w-3.5 mr-2" />
                  Run
                </Button>
              </CardAction>
            </CardHeader>
            <CardContent className="px-4 space-y-2">
              {!canRun && (
                <div className="text-xs text-muted-foreground">
                  Add <span className="font-medium">Title</span> and <span className="font-medium">Abstract</span> in Blueprint to enable.
                </div>
              )}
              {runId && (
                <div className="text-[11px] text-muted-foreground">
                  run_id: <span className="font-mono">{runId}</span>
                </div>
              )}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {status === "success" ? (
                  <>
                    <CheckCircle2 className="h-4 w-4 text-green-500" /> Last run succeeded.
                  </>
                ) : status === "error" ? (
                  <>
                    <AlertCircle className="h-4 w-4 text-red-500" /> Last run failed. Check Timeline.
                  </>
                ) : status === "running" ? (
                  <>Streaming progress to Timeline…</>
                ) : (
                  <>Ready.</>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Smoke</CardTitle>
              <CardDescription className="text-xs">Minimal sanity check for the generated project (pip + compileall).</CardDescription>
              <CardAction>
                <Button size="sm" className="h-8" onClick={runSmoke} disabled={!projectDir || status === "running"}>
                  <Play className="h-3.5 w-3.5 mr-2" />
                  Run
                </Button>
              </CardAction>
            </CardHeader>
            <CardContent className="px-4 text-xs text-muted-foreground space-y-1">
              {projectDir ? (
                <>
                  <div>
                    Project: <span className="font-mono">{projectDir}</span>
                  </div>
                  <div>
                    Executor: <span className="font-mono">{executor}</span>
                  </div>
                </>
              ) : (
                <div>Run Paper2Code first to get an output directory.</div>
              )}
            </CardContent>
          </Card>

          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Upcoming Steps</CardTitle>
              <CardDescription className="text-xs">Install / Data / Smoke / Train(Mini) / Eval / Report.</CardDescription>
            </CardHeader>
            <CardContent className="px-4 text-xs text-muted-foreground">
              Next step: wire Runbook steps to a persisted <span className="font-mono">ReproRun</span> with log + artifact capture.
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  )
}
