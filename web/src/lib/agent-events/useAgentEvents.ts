"use client"

import { useEffect, useRef } from "react"
import { readSSE } from "@/lib/sse"
import { useAgentEventStore } from "./store"
import { parseActivityItem, parseAgentStatus, parseToolCall, parseFileTouched, parseCodexDelegation, parseScoreEdge } from "./parsers"
import type { AgentEventEnvelopeRaw } from "./types"

const EVENTS_STREAM_PATH = "/api/events/stream"

export function useAgentEvents() {
  const { setConnected, addFeedItem, updateAgentStatus, addToolCall, addFileTouched, addCodexDelegation, addScoreEdge } = useAgentEventStore()
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller

    async function connect() {
      try {
        const res = await fetch(EVENTS_STREAM_PATH, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        })
        if (!res.ok || !res.body) return
        setConnected(true)

        for await (const msg of readSSE(res.body)) {
          const raw = msg as unknown as AgentEventEnvelopeRaw
          if (!raw?.type) continue

          const feedItem = parseActivityItem(raw)
          if (feedItem) addFeedItem(feedItem)

          const statusEntry = parseAgentStatus(raw)
          if (statusEntry) updateAgentStatus(statusEntry)

          const toolCall = parseToolCall(raw)
          if (toolCall) addToolCall(toolCall)

          const fileTouched = parseFileTouched(raw)
          if (fileTouched) addFileTouched(fileTouched)

          const codexDel = parseCodexDelegation(raw)
          if (codexDel) addCodexDelegation(codexDel)

          const scoreEdge = parseScoreEdge(raw)
          if (scoreEdge) addScoreEdge(scoreEdge)
        }
      } catch (err) {
        if (controller.signal.aborted || abortRef.current?.signal.aborted) {
          return
        }
        if ((err as Error)?.name !== "AbortError") {
          console.warn("[useAgentEvents] disconnected, will retry in 3s", err)
          setTimeout(connect, 3000)
        }
      } finally {
        setConnected(false)
      }
    }

    connect()
    return () => {
      controller.abort()
    }
  }, [setConnected, addFeedItem, updateAgentStatus, addToolCall, addFileTouched, addCodexDelegation, addScoreEdge])
}
