"use client"

import { useCallback, useState } from "react"
import { readSSE } from "@/lib/sse"
import { useStudioStore } from "@/lib/store/studio-store"
import type {
  GenerateCompletedEvent,
  GenerateErrorEvent,
  GenerationStatus,
  ReproContextPack,
  StageObservationsEvent,
  StageProgressEvent,
} from "@/lib/types/p2c"

function isReproContextPack(payload: unknown): payload is ReproContextPack {
  if (!payload || typeof payload !== "object") return false
  const value = payload as Partial<ReproContextPack> & {
    paper?: { paper_id?: unknown }
    confidence?: { overall?: unknown }
  }
  return (
    typeof value.context_pack_id === "string" &&
    typeof value.version === "string" &&
    typeof value.created_at === "string" &&
    typeof value.objective === "string" &&
    typeof value.paper_type === "string" &&
    Array.isArray(value.observations) &&
    Array.isArray(value.task_roadmap) &&
    Array.isArray(value.warnings) &&
    typeof value.paper?.paper_id === "string" &&
    typeof value.confidence?.overall === "number"
  )
}

export function useContextPackGeneration() {
  const [status, setStatus] = useState<GenerationStatus>("idle")
  const [result, setResult] = useState<GenerateCompletedEvent | null>(null)
  const MIN_LOADING_MS = 1500

  const {
    setContextPack,
    setContextPackLoading,
    setContextPackError,
    appendGenerationProgress,
    appendLiveObservations,
    clearGenerationProgress,
    clearLiveObservations,
  } = useStudioStore()

  const normalizePack = (payload: unknown): ReproContextPack | null => {
    if (!payload || typeof payload !== "object") return null
    const raw = payload as Record<string, unknown>
    if (raw.pack && typeof raw.pack === "object") {
      const pack = raw.pack as Record<string, unknown>
      const maybePack: Record<string, unknown> =
        !pack.context_pack_id && typeof raw.context_pack_id === "string"
          ? { ...pack, context_pack_id: raw.context_pack_id }
          : pack
      return isReproContextPack(maybePack) ? maybePack : null
    }
    return isReproContextPack(raw) ? raw : null
  }

  const generate = useCallback(async (params: {
    paperId: string
    userId?: string
    depth?: "fast" | "standard" | "deep"
    title?: string
    abstract?: string
  }) => {
    const startedAt = Date.now()
    console.info("[P2C:M3] generate:start", { paperId: params.paperId, depth: params.depth })
    setStatus("generating")
    setResult(null)
    setContextPack(null)
    setContextPackError(null)
    clearGenerationProgress()
    clearLiveObservations()
    setContextPackLoading(true)

    try {
      const payload: Record<string, unknown> = {
        paper_id: params.paperId,
        user_id: params.userId ?? "default",
        depth: params.depth ?? "standard",
      }
      if (params.title !== undefined) payload.title = params.title
      if (params.abstract !== undefined) payload.abstract = params.abstract

      const response = await fetch("/api/research/repro/context", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok || !response.body) {
        console.error("[P2C:M3] generate:request_failed", { status: response.status })
        throw new Error(`Failed to start generation (${response.status})`)
      }

      for await (const evt of readSSE(response.body)) {
        if (!evt?.type) continue

        if (evt.type === "progress" || evt.type === "stage_progress") {
          console.debug("[P2C:M3] generate:progress", evt.data)
          appendGenerationProgress(evt.data as StageProgressEvent)
        } else if (evt.type === "stage_observations") {
          console.debug("[P2C:M3] generate:observations", evt.data)
          appendLiveObservations(evt.data as StageObservationsEvent)
        } else if (evt.type === "result" || evt.type === "completed") {
          const completed = evt.data as GenerateCompletedEvent
          console.info("[P2C:M3] generate:completed", completed)
          setResult(completed)
          setStatus("completed")

          let packPayload: ReproContextPack | null = null
          if (completed?.context_pack_id) {
            const packRes = await fetch(`/api/research/repro/context/${completed.context_pack_id}`)
            if (!packRes.ok) {
              console.error("[P2C:M3] generate:pack_fetch_failed", { status: packRes.status })
              throw new Error(`Failed to load context pack (${packRes.status})`)
            }
            const payload = await packRes.json()
            const pack = normalizePack(payload)
            if (pack) packPayload = pack
          }

          const elapsed = Date.now() - startedAt
          if (elapsed < MIN_LOADING_MS) {
            await new Promise((resolve) => setTimeout(resolve, MIN_LOADING_MS - elapsed))
          }

          if (packPayload) setContextPack(packPayload)
          setContextPackLoading(false)
          return
        } else if (evt.type === "error") {
          const error = (evt.data && typeof evt.data === "object" ? evt.data : {}) as GenerateErrorEvent
          const message = evt.message || error?.message || error?.error || "Generation failed"
          console.error("[P2C:M3] generate:error", { message, data: evt.data })
          setContextPackError(message)
          setStatus("error")
          setContextPackLoading(false)
          return
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error"
      console.error("[P2C:M3] generate:exception", { message })
      setContextPackError(message)
      setStatus("error")
      setContextPackLoading(false)
    }
  }, [
    appendGenerationProgress,
    appendLiveObservations,
    clearGenerationProgress,
    clearLiveObservations,
    setContextPack,
    setContextPackError,
    setContextPackLoading,
  ])

  return { status, result, generate }
}
