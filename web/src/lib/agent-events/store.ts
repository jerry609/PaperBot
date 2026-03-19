"use client"

import { create } from "zustand"
import type { AgentTask } from "@/lib/store/studio-store"
import type { ActivityFeedItem, AgentStatusEntry, CodexDelegationEntry, FileTouchedEntry, ScoreEdgeEntry, ToolCallEntry } from "./types"

const FEED_MAX = 200
const TOOL_TIMELINE_MAX = 100
const CODEX_DELEGATIONS_MAX = 100
const SCORE_EDGES_MAX = 200

export type AgentInspectorView = "live" | "tools" | "files" | "agents" | "graph"
export type AgentWorkspaceSurface = "log" | "context" | "board" | "commands" | null

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

  // ScoreShareBus edges — upserted by id, capped at SCORE_EDGES_MAX
  scoreEdges: ScoreEdgeEntry[]
  addScoreEdge: (entry: ScoreEdgeEntry) => void

  // Kanban task board — upserted by task id
  kanbanTasks: AgentTask[]
  upsertKanbanTask: (task: AgentTask) => void

  // Monitor inspector state
  inspectorView: AgentInspectorView
  setInspectorView: (view: AgentInspectorView) => void
  selectedWorkerRunId: string | null
  setSelectedWorkerRunId: (workerRunId: string | null) => void
  openWorkerRun: (workerRunId: string) => void

  // Cross-panel workspace navigation
  requestedWorkspaceSurface: AgentWorkspaceSurface
  requestWorkspaceSurface: (surface: Exclude<AgentWorkspaceSurface, null>) => void
  clearWorkspaceSurfaceRequest: () => void
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

  scoreEdges: [],
  addScoreEdge: (entry) =>
    set((s) => {
      const idx = s.scoreEdges.findIndex((e) => e.id === entry.id)
      if (idx !== -1) {
        const next = [...s.scoreEdges]
        next[idx] = entry
        return { scoreEdges: next }
      }
      return { scoreEdges: [entry, ...s.scoreEdges].slice(0, SCORE_EDGES_MAX) }
    }),

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

  inspectorView: "live",
  setInspectorView: (view) => set({ inspectorView: view }),
  selectedWorkerRunId: null,
  setSelectedWorkerRunId: (workerRunId) => set({ selectedWorkerRunId: workerRunId }),
  openWorkerRun: (workerRunId) => set({
    inspectorView: "agents",
    selectedWorkerRunId: workerRunId,
  }),

  requestedWorkspaceSurface: null,
  requestWorkspaceSurface: (surface) => set({ requestedWorkspaceSurface: surface }),
  clearWorkspaceSurfaceRequest: () => set({ requestedWorkspaceSurface: null }),
}))
