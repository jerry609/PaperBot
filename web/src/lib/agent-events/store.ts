"use client"

import { create } from "zustand"
import type { AgentTask } from "@/lib/store/studio-store"
import type { ActivityFeedItem, AgentStatusEntry, CodexDelegationEntry, FileTouchedEntry, ToolCallEntry } from "./types"

const FEED_MAX = 200
const TOOL_TIMELINE_MAX = 100
const CODEX_DELEGATIONS_MAX = 100

interface AgentEventState {
  // SSE connection status
  connected: boolean
  setConnected: (c: boolean) => void

  // Activity feed — newest first, capped at FEED_MAX
  feed: ActivityFeedItem[]
  addFeedItem: (item: ActivityFeedItem) => void
  clearFeed: () => void

  // Per-agent status map
  agentStatuses: Record<string, AgentStatusEntry>
  updateAgentStatus: (entry: AgentStatusEntry) => void

  // Tool call timeline — newest first, capped at TOOL_TIMELINE_MAX
  toolCalls: ToolCallEntry[]
  addToolCall: (entry: ToolCallEntry) => void
  clearToolCalls: () => void

  // File tracking — keyed by run_id, capped at 20 runs, dedup by path within run
  filesTouched: Record<string, FileTouchedEntry[]>
  addFileTouched: (entry: FileTouchedEntry) => void
  selectedRunId: string | null
  setSelectedRunId: (id: string | null) => void
  selectedFile: FileTouchedEntry | null
  setSelectedFile: (file: FileTouchedEntry | null) => void

  // Codex delegation events — newest first, capped at CODEX_DELEGATIONS_MAX
  codexDelegations: CodexDelegationEntry[]
  addCodexDelegation: (entry: CodexDelegationEntry) => void

  // Kanban task board — upserted by task id
  kanbanTasks: AgentTask[]
  upsertKanbanTask: (task: AgentTask) => void
}

export const useAgentEventStore = create<AgentEventState>((set) => ({
  connected: false,
  setConnected: (c) => set({ connected: c }),

  feed: [],
  addFeedItem: (item) =>
    set((s) => ({
      feed: [item, ...s.feed].slice(0, FEED_MAX),
    })),
  clearFeed: () => set({ feed: [] }),

  agentStatuses: {},
  updateAgentStatus: (entry) =>
    set((s) => ({
      agentStatuses: { ...s.agentStatuses, [entry.agent_name]: entry },
    })),

  toolCalls: [],
  addToolCall: (entry) =>
    set((s) => ({
      toolCalls: [entry, ...s.toolCalls].slice(0, TOOL_TIMELINE_MAX),
    })),
  clearToolCalls: () => set({ toolCalls: [] }),

  filesTouched: {},
  addFileTouched: (entry) =>
    set((s) => {
      const existing = s.filesTouched[entry.run_id] ?? []
      // Dedup by path: ignore if same path already tracked for this run
      if (existing.some((e) => e.path === entry.path)) return s
      const updated = {
        ...s.filesTouched,
        [entry.run_id]: [...existing, entry],
      }
      // Evict oldest run_id when exceeding 20 runs
      const keys = Object.keys(updated)
      if (keys.length > 20) {
        delete updated[keys[0]]
      }
      return { filesTouched: updated }
    }),
  selectedRunId: null,
  setSelectedRunId: (id) => set({ selectedRunId: id }),
  selectedFile: null,
  setSelectedFile: (file) => set({ selectedFile: file }),

  codexDelegations: [],
  addCodexDelegation: (entry) =>
    set((s) => ({
      codexDelegations: [entry, ...s.codexDelegations].slice(0, CODEX_DELEGATIONS_MAX),
    })),

  kanbanTasks: [],
  upsertKanbanTask: (task) =>
    set((s) => {
      const idx = s.kanbanTasks.findIndex((t) => t.id === task.id)
      if (idx === -1) {
        return { kanbanTasks: [...s.kanbanTasks, task] }
      }
      const next = [...s.kanbanTasks]
      next[idx] = { ...next[idx], ...task }
      return { kanbanTasks: next }
    }),
}))
