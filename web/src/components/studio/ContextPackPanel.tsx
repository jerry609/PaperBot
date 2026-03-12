"use client"

import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, ChevronDown, ChevronRight, LayoutDashboard } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useStudioStore } from "@/lib/store/studio-store"
import { readSSE } from "@/lib/sse"
import { backendUrl } from "@/lib/backend-url"
import { cn } from "@/lib/utils"
import type {
  ContextPackSession,
  ExtractionObservation,
  ObservationConcept,
  ObservationType,
  ReproContextPack,
} from "@/lib/types/p2c"

const CONFIDENCE_STYLES: Array<{ max: number; className: string }> = [
  { max: 0.6, className: "bg-red-100 text-red-700" },
  { max: 0.8, className: "bg-amber-100 text-amber-700" },
  { max: 1.1, className: "bg-emerald-100 text-emerald-700" },
]

interface Props {
  pack: ReproContextPack
  onSessionCreated?: (session: ContextPackSession) => void
  onDeployToBoard?: () => void
  className?: string
}

export function ContextPackPanel({ pack, onSessionCreated, onDeployToBoard, className }: Props) {
  const [creating, setCreating] = useState(false)
  const [deploying, setDeploying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { selectedPaperId, addAgentTask, setBoardSessionId, clearAgentTasks } = useStudioStore()
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [filter, setFilter] = useState<{ stage: string | null; type: ObservationType | null; concept: ObservationConcept | null }>({
    stage: null,
    type: null,
    concept: null,
  })

  const confidenceStyle = CONFIDENCE_STYLES.find((entry) => pack.confidence.overall <= entry.max)
    ?.className || "bg-muted text-muted-foreground"

  const stages = useMemo(() => Array.from(new Set(pack.observations.map((obs) => obs.stage))), [pack.observations])
  const types = useMemo(() => Array.from(new Set(pack.observations.map((obs) => obs.type))), [pack.observations])
  const concepts = useMemo(
    () => Array.from(new Set(pack.observations.flatMap((obs) => obs.concepts))),
    [pack.observations]
  )

  const filteredObservations = useMemo(() => {
    return pack.observations.filter((obs) => {
      if (filter.stage && obs.stage !== filter.stage) return false
      if (filter.type && obs.type !== filter.type) return false
      if (filter.concept && !obs.concepts.includes(filter.concept)) return false
      return true
    })
  }, [pack.observations, filter])

  const handleCreateSession = async () => {
    setCreating(true)
    setError(null)
    try {
      const res = await fetch(`/api/research/repro/context/${pack.context_pack_id}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ executor_preference: "auto" }),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Failed to create session (${res.status})`)
      }

      const session = (await res.json()) as ContextPackSession
      onSessionCreated?.(session)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session")
    } finally {
      setCreating(false)
    }
  }

  const handleDeployToBoard = async () => {
    setDeploying(true)
    setError(null)
    clearAgentTasks()
    try {
      // 1. Create agent board session (direct to backend, bypasses Next.js rewrite buffering)
      const sessionRes = await fetch(backendUrl("/api/agent-board/sessions"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_id: selectedPaperId || "",
          context_pack_id: pack.context_pack_id,
          paper_title: pack.paper?.title || "",
        }),
      })
      if (!sessionRes.ok) throw new Error(`Failed to create board session (${sessionRes.status})`)
      const session = await sessionRes.json()
      setBoardSessionId(session.session_id)

      // 2. Start planning (SSE) -- Claude decomposes into tasks
      //    Must go directly to backend; Next.js rewrite proxy buffers SSE.
      const planRes = await fetch(backendUrl(`/api/agent-board/sessions/${session.session_id}/plan`), {
        method: "POST",
      })
      if (!planRes.ok || !planRes.body) throw new Error(`Planning failed (${planRes.status})`)

      for await (const evt of readSSE(planRes.body)) {
        if (evt?.type === "progress") {
          const data = (evt.data ?? {}) as Record<string, unknown>
          if (data.event === "task_created" && data.task) {
            const t = data.task as Record<string, unknown>
            addAgentTask({
              id: (t.id as string) || `task-${Date.now()}`,
              title: (t.title as string) || "Untitled",
              description: (t.description as string) || "",
              status: (t.status as "planning") || "planning",
              assignee: (t.assignee as string) || "claude",
              progress: (t.progress as number) || 0,
              tags: (t.tags as string[]) || [],
              subtasks: (t.subtasks as { id: string; title: string; done: boolean }[]) || [],
              codexOutput: (t.codex_output as string) || undefined,
              generatedFiles: (t.generated_files as string[]) || [],
              reviewFeedback: (t.review_feedback as string) || undefined,
              executionLog: (t.execution_log as {
                id: string
                timestamp: string
                event: string
                phase: string
                level: "info" | "warning" | "error" | "success"
                message: string
                details?: Record<string, unknown>
              }[]) || [],
              paperId: selectedPaperId || undefined,
            })
          }
        } else if (evt?.type === "error") {
          throw new Error(evt.message || "Planning failed")
        }
      }

      onDeployToBoard?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to deploy to board")
    } finally {
      setDeploying(false)
    }
  }

  return (
    <div className={cn("h-full flex flex-col gap-4 p-4 overflow-auto", className)}>
      <Card>
        <CardHeader className="space-y-2">
          <CardTitle className="text-lg">{pack.paper.title}</CardTitle>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>{pack.paper.year}</span>
            <span>•</span>
            <span>{pack.paper.authors.join(", ")}</span>
            <Badge variant="outline">{pack.paper_type}</Badge>
            <Badge className={confidenceStyle}>Confidence {Math.round(pack.confidence.overall * 100)}%</Badge>
            <span>{pack.observations.length} observations</span>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="rounded-md bg-muted/50 p-3 text-sm">
            <p className="font-medium">Objective</p>
            <p className="text-muted-foreground mt-1">{pack.objective}</p>
          </div>

          {pack.warnings.length > 0 && (
            <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <div className="flex items-center gap-2 font-medium">
                <AlertTriangle className="h-4 w-4" />
                Warnings
              </div>
              <ul className="mt-2 list-disc list-inside text-xs">
                {pack.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-2 text-xs">
        <FilterGroup
          label="Stage"
          options={stages}
          selected={filter.stage}
          onSelect={(value) => setFilter((prev) => ({ ...prev, stage: value }))}
        />
        <FilterGroup
          label="Type"
          options={types}
          selected={filter.type}
          onSelect={(value) => setFilter((prev) => ({ ...prev, type: value as ObservationType | null }))}
        />
        <FilterGroup
          label="Concept"
          options={concepts}
          selected={filter.concept}
          onSelect={(value) => setFilter((prev) => ({ ...prev, concept: value as ObservationConcept | null }))}
        />
      </div>

      <div className="space-y-3">
        {filteredObservations.map((obs) => (
          <ObservationCard
            key={obs.id}
            packId={pack.context_pack_id}
            observation={obs}
            expanded={expandedId === obs.id}
            onToggle={() => setExpandedId(expandedId === obs.id ? null : obs.id)}
          />
        ))}
        {filteredObservations.length === 0 && (
          <p className="text-sm text-muted-foreground">No observations match the current filters.</p>
        )}
      </div>

      {pack.task_roadmap.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Roadmap</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {pack.task_roadmap.map((step) => (
              <div key={step.id} className="rounded-md border px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{step.title}</span>
                  <Badge variant="outline">{step.estimated_difficulty}</Badge>
                </div>
                {step.description && <p className="text-xs text-muted-foreground mt-1">{step.description}</p>}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {error && <p className="text-xs text-red-600">{error}</p>}

      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={handleDeployToBoard} disabled={deploying || creating}>
          <LayoutDashboard className="h-4 w-4 mr-1.5" />
          {deploying ? "Deploying..." : "Deploy to Agent Board"}
        </Button>
        <Button onClick={handleCreateSession} disabled={creating || deploying}>
          {creating ? "Creating..." : "Create Repro Session"}
        </Button>
      </div>
    </div>
  )
}

function FilterGroup({
  label,
  options,
  selected,
  onSelect,
}: {
  label: string
  options: string[]
  selected: string | null
  onSelect: (value: string | null) => void
}) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-muted-foreground">{label}:</span>
      <button
        onClick={() => onSelect(null)}
        className={`px-2 py-0.5 rounded ${!selected ? "bg-primary/10 text-primary" : "bg-muted"}`}
      >
        All
      </button>
      {options.map((option) => (
        <button
          key={option}
          onClick={() => onSelect(option)}
          className={`px-2 py-0.5 rounded ${selected === option ? "bg-primary/10 text-primary" : "bg-muted"}`}
        >
          {option}
        </button>
      ))}
    </div>
  )
}

function ObservationCard({
  packId,
  observation,
  expanded,
  onToggle,
}: {
  packId: string
  observation: ExtractionObservation
  expanded: boolean
  onToggle: () => void
}) {
  const { detail, loading, error } = useObservationDetail(packId, expanded ? observation.id : null)
  const confidence = Math.round(observation.confidence * 100)
  const dotClass =
    observation.confidence >= 0.8
      ? "bg-emerald-500"
      : observation.confidence >= 0.6
        ? "bg-amber-500"
        : "bg-red-500"

  return (
    <Card className={observation.concepts.includes("gotcha") ? "border-amber-300" : ""}>
      <CardContent className="p-4 space-y-2">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">{observation.type}</Badge>
          <span>{observation.stage}</span>
          <span className={`ml-auto inline-flex items-center gap-2`}>
            <span className={`h-2 w-2 rounded-full ${dotClass}`} />
            <span className="text-muted-foreground">{confidence}%</span>
          </span>
        </div>
        <p className="font-medium">{observation.title}</p>
        <p className="text-sm text-muted-foreground">{observation.narrative}</p>
        <div className="flex flex-wrap gap-1">
          {observation.concepts.map((concept) => (
            <Badge key={concept} variant="secondary" className="text-[10px]">
              {concept}
            </Badge>
          ))}
        </div>
        <button
          onClick={onToggle}
          className="text-xs text-primary inline-flex items-center gap-1"
        >
          {expanded ? "Collapse" : "Expand Details"}
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </button>
        {expanded && (
          <div className="space-y-2 text-xs">
            {loading && <p className="text-muted-foreground">Loading details...</p>}
            {error && <p className="text-red-600">{error}</p>}
            {!loading && detail?.structured_data && Object.keys(detail.structured_data).length > 0 && (
              <pre className="rounded-md bg-muted/50 p-2 overflow-auto">
                {JSON.stringify(detail.structured_data, null, 2)}
              </pre>
            )}
            {!loading && detail?.evidence && detail.evidence.length > 0 && (
              <div className="space-y-1">
                <p className="font-medium text-muted-foreground">Evidence</p>
                {detail.evidence?.map((ev, idx) => (
                  <div key={`${ev.ref}-${idx}`} className="flex items-start gap-2">
                    <Badge variant="outline" className="text-[10px]">{ev.type}</Badge>
                    <div>
                      <div>{ev.ref}</div>
                      {ev.supports?.length ? (
                        <div className="text-[10px] text-muted-foreground">supports: {ev.supports.join(", ")}</div>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!loading && !error && (!detail?.structured_data || Object.keys(detail.structured_data).length === 0) && (!detail?.evidence || detail.evidence.length === 0) && (
              <p className="text-muted-foreground">No additional details yet.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function useObservationDetail(packId: string, observationId: string | null) {
  const [detail, setDetail] = useState<ExtractionObservation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!observationId) {
      setDetail(null)
      setLoading(false)
      setError(null)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(`/api/research/repro/context/${packId}/observation/${observationId}`)
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text()
          throw new Error(text || `Failed to load observation (${res.status})`)
        }
        return res.json() as Promise<ExtractionObservation>
      })
      .then((data) => {
        if (!cancelled) setDetail(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load observation")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [packId, observationId])

  return { detail, loading, error }
}
