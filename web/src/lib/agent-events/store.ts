"use client"

import { create } from "zustand"
import type { ActivityFeedItem, AgentStatusEntry, ToolCallEntry } from "./types"

const FEED_MAX = 200
const TOOL_TIMELINE_MAX = 100

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
}))
