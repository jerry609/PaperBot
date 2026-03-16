import { beforeEach, describe, expect, it } from "vitest"
import { useAgentEventStore } from "./store"
import type { ActivityFeedItem, AgentStatusEntry, CodexDelegationEntry, FileTouchedEntry, ScoreEdgeEntry, ToolCallEntry } from "./types"

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

function makeFileTouched(runId: string, path: string, i: number = 0): FileTouchedEntry {
  return {
    run_id: runId,
    agent_name: "codex-a1b2",
    path,
    status: "modified",
    ts: `2026-03-15T00:00:${String(i).padStart(2, "0")}Z`,
    linesAdded: 5,
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

  describe("addFileTouched", () => {
    it("adds entry under correct run_id key", () => {
      const entry = makeFileTouched("run-1", "src/main.py", 0)
      useAgentEventStore.getState().addFileTouched(entry)
      const state = useAgentEventStore.getState()
      expect(state.filesTouched["run-1"]).toBeDefined()
      expect(state.filesTouched["run-1"][0].path).toBe("src/main.py")
    })

    it("deduplicates same path within same run_id (second add is ignored)", () => {
      const entry1 = makeFileTouched("run-1", "src/main.py", 0)
      const entry2 = makeFileTouched("run-1", "src/main.py", 1)
      useAgentEventStore.getState().addFileTouched(entry1)
      useAgentEventStore.getState().addFileTouched(entry2)
      expect(useAgentEventStore.getState().filesTouched["run-1"]).toHaveLength(1)
    })

    it("evicts oldest run_id when exceeding 20 runs", () => {
      // Add 21 distinct run IDs
      for (let i = 0; i < 21; i++) {
        const entry = makeFileTouched(`run-evict-${i}`, "src/x.py", i)
        useAgentEventStore.getState().addFileTouched(entry)
      }
      const keys = Object.keys(useAgentEventStore.getState().filesTouched)
      expect(keys.length).toBe(20)
      // run-evict-0 was the first added, should be evicted
      expect(keys).not.toContain("run-evict-0")
    })

    it("setSelectedRunId updates selectedRunId", () => {
      useAgentEventStore.getState().setSelectedRunId("run-abc")
      expect(useAgentEventStore.getState().selectedRunId).toBe("run-abc")
    })

    it("setSelectedFile updates selectedFile", () => {
      const file = makeFileTouched("run-1", "src/utils.py", 0)
      useAgentEventStore.getState().setSelectedFile(file)
      expect(useAgentEventStore.getState().selectedFile?.path).toBe("src/utils.py")
    })

    it("initial state has filesTouched={}, selectedRunId=null, selectedFile=null", () => {
      const state = useAgentEventStore.getState()
      expect(state.filesTouched).toEqual({})
      expect(state.selectedRunId).toBeNull()
      expect(state.selectedFile).toBeNull()
    })
  })

  describe("monitor inspector state", () => {
    it("setInspectorView updates the active monitor tab", () => {
      useAgentEventStore.getState().setInspectorView("agents")
      expect(useAgentEventStore.getState().inspectorView).toBe("agents")
    })

    it("setSelectedWorkerRunId updates the selected worker run", () => {
      useAgentEventStore.getState().setSelectedWorkerRunId("worker-run-abc")
      expect(useAgentEventStore.getState().selectedWorkerRunId).toBe("worker-run-abc")
    })

    it("openWorkerRun switches the monitor to workers and selects the run", () => {
      useAgentEventStore.getState().openWorkerRun("worker-run-open")
      const state = useAgentEventStore.getState()
      expect(state.inspectorView).toBe("agents")
      expect(state.selectedWorkerRunId).toBe("worker-run-open")
    })
  })
})

function makeCodexDelegation(i: number): CodexDelegationEntry {
  return {
    id: `codex_dispatched-task-${i}-2026-03-15T00:00:00Z`,
    event_type: "codex_dispatched",
    task_id: `task-${i}`,
    worker_run_id: `worker-run-${i}`,
    task_title: `Task ${i}`,
    assignee: "codex-a1b2",
    session_id: "sess-001",
    runtime: "codex",
    control_mode: "mirrored",
    interruptible: false,
    ts: "2026-03-15T00:00:00Z",
  }
}

function makeAgentTask(id: string, title: string): import("@/lib/store/studio-store").AgentTask {
  return {
    id,
    title,
    description: "Test task",
    status: "planning",
    assignee: "claude",
    progress: 0,
    tags: [],
    createdAt: "2026-03-15T00:00:00Z",
    updatedAt: "2026-03-15T00:00:00Z",
    subtasks: [],
  }
}

describe("codexDelegations", () => {
  beforeEach(() => {
    useAgentEventStore.setState(useAgentEventStore.getInitialState(), true)
  })

  it("addCodexDelegation adds entry to codexDelegations array", () => {
    const entry = makeCodexDelegation(1)
    useAgentEventStore.getState().addCodexDelegation(entry)
    expect(useAgentEventStore.getState().codexDelegations[0]).toEqual(entry)
  })

  it("addCodexDelegation caps at 100 entries", () => {
    for (let i = 0; i < 101; i++) {
      useAgentEventStore.getState().addCodexDelegation(makeCodexDelegation(i))
    }
    expect(useAgentEventStore.getState().codexDelegations).toHaveLength(100)
  })
})

describe("kanbanTasks", () => {
  beforeEach(() => {
    useAgentEventStore.setState(useAgentEventStore.getInitialState(), true)
  })

  it("upsertKanbanTask adds new task when id not found", () => {
    const task = makeAgentTask("t1", "First Task")
    useAgentEventStore.getState().upsertKanbanTask(task)
    const tasks = useAgentEventStore.getState().kanbanTasks
    expect(tasks).toHaveLength(1)
    expect(tasks[0].id).toBe("t1")
  })

  it("upsertKanbanTask merges updates when id already exists", () => {
    const task = makeAgentTask("t1", "First Task")
    useAgentEventStore.getState().upsertKanbanTask(task)
    const updated = { ...task, title: "Updated Task", status: "done" as const }
    useAgentEventStore.getState().upsertKanbanTask(updated)
    const tasks = useAgentEventStore.getState().kanbanTasks
    expect(tasks).toHaveLength(1)
    expect(tasks[0].title).toBe("Updated Task")
    expect(tasks[0].status).toBe("done")
  })
})

function makeScoreEdge(id: string, score = 0.8): ScoreEdgeEntry {
  return {
    id,
    from_agent: "research",
    to_agent: "scholar_pipeline",
    stage: "research",
    score,
    ts: "2026-03-15T05:00:00Z",
  }
}

describe("scoreEdges", () => {
  beforeEach(() => {
    useAgentEventStore.setState(useAgentEventStore.getInitialState(), true)
  })

  it("addScoreEdge adds entry to scoreEdges array", () => {
    const entry = makeScoreEdge("research-scholar_pipeline-research")
    useAgentEventStore.getState().addScoreEdge(entry)
    expect(useAgentEventStore.getState().scoreEdges).toHaveLength(1)
    expect(useAgentEventStore.getState().scoreEdges[0]).toEqual(entry)
  })

  it("addScoreEdge upserts (replaces) entry with same id", () => {
    const entry = makeScoreEdge("edge-id-1", 0.5)
    useAgentEventStore.getState().addScoreEdge(entry)
    const updated = makeScoreEdge("edge-id-1", 0.9)
    useAgentEventStore.getState().addScoreEdge(updated)
    const edges = useAgentEventStore.getState().scoreEdges
    expect(edges).toHaveLength(1)
    expect(edges[0].score).toBe(0.9)
  })

  it("addScoreEdge does not deduplicate entries with different ids", () => {
    useAgentEventStore.getState().addScoreEdge(makeScoreEdge("edge-A"))
    useAgentEventStore.getState().addScoreEdge(makeScoreEdge("edge-B"))
    expect(useAgentEventStore.getState().scoreEdges).toHaveLength(2)
  })

  it("addScoreEdge caps at 200 entries", () => {
    for (let i = 0; i < 201; i++) {
      useAgentEventStore.getState().addScoreEdge(makeScoreEdge(`edge-${i}`))
    }
    expect(useAgentEventStore.getState().scoreEdges).toHaveLength(200)
  })

  it("initial state has scoreEdges=[]", () => {
    expect(useAgentEventStore.getState().scoreEdges).toEqual([])
  })
})
