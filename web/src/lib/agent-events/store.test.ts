import { beforeEach, describe, expect, it } from "vitest"
import { useAgentEventStore } from "./store"
import type { ActivityFeedItem, AgentStatusEntry, ToolCallEntry } from "./types"

function makeItem(i: number): ActivityFeedItem {
  return {
    id: `item-${i}`,
    type: "agent_started",
    agent_name: "TestAgent",
    workflow: "test",
    stage: "stage",
    ts: `2026-03-15T00:00:${String(i).padStart(2, "0")}Z`,
    summary: `Event ${i}`,
    raw: { type: "agent_started", ts: `2026-03-15T00:00:${String(i).padStart(2, "0")}Z` },
  }
}

function makeToolCall(i: number): ToolCallEntry {
  return {
    id: `tool-${i}`,
    tool: "paper_search",
    agent_name: "mcp",
    arguments: {},
    result_summary: `Result ${i}`,
    error: null,
    duration_ms: 100,
    ts: `2026-03-15T00:00:${String(i).padStart(2, "0")}Z`,
    status: "ok",
  }
}

const resetStore = () => {
  useAgentEventStore.setState(useAgentEventStore.getInitialState(), true)
}

describe("useAgentEventStore", () => {
  beforeEach(() => {
    resetStore()
  })

  describe("setConnected", () => {
    it("toggles connected boolean to true", () => {
      useAgentEventStore.getState().setConnected(true)
      expect(useAgentEventStore.getState().connected).toBe(true)
    })

    it("toggles connected boolean to false", () => {
      useAgentEventStore.getState().setConnected(true)
      useAgentEventStore.getState().setConnected(false)
      expect(useAgentEventStore.getState().connected).toBe(false)
    })
  })

  describe("addFeedItem", () => {
    it("adds a feed item to the front of the array", () => {
      const item = makeItem(1)
      useAgentEventStore.getState().addFeedItem(item)
      expect(useAgentEventStore.getState().feed[0]).toEqual(item)
    })

    it("caps feed at 200 after adding 201 items", () => {
      for (let i = 0; i < 201; i++) {
        useAgentEventStore.getState().addFeedItem(makeItem(i))
      }
      expect(useAgentEventStore.getState().feed).toHaveLength(200)
    })

    it("keeps the newest item at index 0 after cap", () => {
      for (let i = 0; i < 201; i++) {
        useAgentEventStore.getState().addFeedItem(makeItem(i))
      }
      // Item 200 was added last — it should be at index 0
      expect(useAgentEventStore.getState().feed[0].id).toBe("item-200")
    })
  })

  describe("clearFeed", () => {
    it("empties the feed array", () => {
      useAgentEventStore.getState().addFeedItem(makeItem(1))
      useAgentEventStore.getState().clearFeed()
      expect(useAgentEventStore.getState().feed).toHaveLength(0)
    })
  })

  describe("addToolCall", () => {
    it("adds a tool call to the front of the array", () => {
      const tool = makeToolCall(1)
      useAgentEventStore.getState().addToolCall(tool)
      expect(useAgentEventStore.getState().toolCalls[0]).toEqual(tool)
    })

    it("caps toolCalls at 100 after adding 101 items", () => {
      for (let i = 0; i < 101; i++) {
        useAgentEventStore.getState().addToolCall(makeToolCall(i))
      }
      expect(useAgentEventStore.getState().toolCalls).toHaveLength(100)
    })
  })

  describe("clearToolCalls", () => {
    it("empties the toolCalls array", () => {
      useAgentEventStore.getState().addToolCall(makeToolCall(1))
      useAgentEventStore.getState().clearToolCalls()
      expect(useAgentEventStore.getState().toolCalls).toHaveLength(0)
    })
  })

  describe("updateAgentStatus", () => {
    it("adds a new agent status entry keyed by agent_name", () => {
      const entry: AgentStatusEntry = {
        agent_name: "ResearchAgent",
        status: "working",
        last_stage: "paper_search",
        last_ts: "2026-03-15T00:00:00Z",
      }
      useAgentEventStore.getState().updateAgentStatus(entry)
      expect(useAgentEventStore.getState().agentStatuses["ResearchAgent"]).toEqual(entry)
    })

    it("overwrites existing agent status for the same agent_name", () => {
      const initial: AgentStatusEntry = {
        agent_name: "ResearchAgent",
        status: "working",
        last_stage: "paper_search",
        last_ts: "2026-03-15T00:00:00Z",
      }
      const updated: AgentStatusEntry = {
        agent_name: "ResearchAgent",
        status: "completed",
        last_stage: "summarize",
        last_ts: "2026-03-15T01:00:00Z",
      }
      useAgentEventStore.getState().updateAgentStatus(initial)
      useAgentEventStore.getState().updateAgentStatus(updated)
      expect(useAgentEventStore.getState().agentStatuses["ResearchAgent"].status).toBe("completed")
    })

    it("tracks multiple agents independently", () => {
      const agentA: AgentStatusEntry = {
        agent_name: "AgentA",
        status: "working",
        last_stage: "stage-a",
        last_ts: "2026-03-15T00:00:00Z",
      }
      const agentB: AgentStatusEntry = {
        agent_name: "AgentB",
        status: "errored",
        last_stage: "stage-b",
        last_ts: "2026-03-15T00:00:00Z",
      }
      useAgentEventStore.getState().updateAgentStatus(agentA)
      useAgentEventStore.getState().updateAgentStatus(agentB)
      expect(useAgentEventStore.getState().agentStatuses["AgentA"].status).toBe("working")
      expect(useAgentEventStore.getState().agentStatuses["AgentB"].status).toBe("errored")
    })
  })
})
