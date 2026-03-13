import { beforeEach, describe, expect, it } from "vitest"
import { useStudioStore } from "./studio-store"
import type {
  ReproContextPack,
  StageObservationsEvent,
  StageProgressEvent,
} from "../types/p2c"

const resetStore = () => {
  useStudioStore.setState(useStudioStore.getInitialState(), true)
}

describe("studio-store", () => {
  beforeEach(() => {
    if (typeof window !== "undefined") {
      window.localStorage.clear()
    }
    resetStore()
  })

  it("round-trips per-paper cached state when switching papers", () => {
    const { addPaper, selectPaper } = useStudioStore.getState()
    const firstId = addPaper({ title: "Paper A", abstract: "A" })
    const secondId = addPaper({ title: "Paper B", abstract: "B" })

    selectPaper(firstId)

    const progress: StageProgressEvent = {
      stage: "extract",
      progress: 0.5,
      message: "Halfway",
    }
    const observations: StageObservationsEvent = {
      stage: "extract",
      observations: [],
    }
    const pack: ReproContextPack = {
      context_pack_id: "cp-1",
      version: "v1",
      created_at: new Date().toISOString(),
      paper: {
        paper_id: "paper-1",
        title: "Paper A",
        year: 2024,
        authors: [],
        identifiers: {},
      },
      paper_type: "experimental",
      objective: "Test",
      observations: [],
      task_roadmap: [],
      confidence: {
        overall: 0,
        literature: 0,
        blueprint: 0,
        environment: 0,
        spec: 0,
        roadmap: 0,
        metrics: 0,
      },
      warnings: [],
    }

    useStudioStore.getState().setContextPack(pack)
    useStudioStore.getState().setContextPackLoading(true)
    useStudioStore.getState().setContextPackError("boom")
    useStudioStore.getState().appendGenerationProgress(progress)
    useStudioStore.getState().appendLiveObservations(observations)
    useStudioStore.getState().setActiveTask("task-1")

    selectPaper(secondId)

    const afterSwitch = useStudioStore.getState()
    expect(afterSwitch.contextPack).toBeNull()
    expect(afterSwitch.contextPackLoading).toBe(false)
    expect(afterSwitch.contextPackError).toBeNull()
    expect(afterSwitch.generationProgress).toHaveLength(0)
    expect(afterSwitch.liveObservations).toHaveLength(0)
    expect(afterSwitch.activeTaskId).toBeNull()

    selectPaper(firstId)

    const restored = useStudioStore.getState()
    expect(restored.contextPack).toEqual(pack)
    expect(restored.contextPackLoading).toBe(true)
    expect(restored.contextPackError).toBe("boom")
    expect(restored.generationProgress).toEqual([progress])
    expect(restored.liveObservations).toEqual([observations])
    expect(restored.activeTaskId).toBe("task-1")
  })

  it("appends text to the last action when streaming", () => {
    const { addPaper, addTask, addAction, appendToLastAction, selectPaper } =
      useStudioStore.getState()

    const paperId = addPaper({ title: "Paper C", abstract: "C" })
    selectPaper(paperId)

    const taskId = addTask("Run task")
    addAction(taskId, { type: "text", content: "hello" })
    appendToLastAction(taskId, " world")

    const task = useStudioStore.getState().tasks.find(t => t.id === taskId)
    expect(task?.actions).toHaveLength(1)
    expect(task?.actions[0].content).toBe("hello world")
  })

  it("preserves provided agent task ids so backend progress updates map correctly", () => {
    const { addAgentTask, updateAgentTask } = useStudioStore.getState()

    const backendTaskId = "task-abc123"
    addAgentTask({
      id: backendTaskId,
      title: "Implement data loader",
      description: "Add deterministic fixtures",
      status: "planning",
      assignee: "claude",
      progress: 0,
      tags: [],
      subtasks: [],
    })

    updateAgentTask(backendTaskId, { status: "in_progress", progress: 15 })

    const task = useStudioStore.getState().agentTasks.find(t => t.id === backendTaskId)
    expect(task).toBeDefined()
    expect(task?.status).toBe("in_progress")
    expect(task?.progress).toBe(15)
  })

  it("setPipelinePhase updates phase", () => {
    const { setPipelinePhase } = useStudioStore.getState()
    expect(useStudioStore.getState().pipelinePhase).toBe("idle")

    setPipelinePhase("executing")
    expect(useStudioStore.getState().pipelinePhase).toBe("executing")

    setPipelinePhase("e2e_running")
    expect(useStudioStore.getState().pipelinePhase).toBe("e2e_running")
  })

  it("setE2EState creates and merges correctly", () => {
    const { setE2EState } = useStudioStore.getState()
    expect(useStudioStore.getState().e2eState).toBeNull()

    // First call creates
    setE2EState({ status: "running", attempt: 0, maxAttempts: 3 })
    const e2e1 = useStudioStore.getState().e2eState
    expect(e2e1).toBeDefined()
    expect(e2e1?.status).toBe("running")
    expect(e2e1?.attempt).toBe(0)
    expect(e2e1?.maxAttempts).toBe(3)

    // Second call merges
    setE2EState({ status: "passed", lastExitCode: 0 })
    const e2e2 = useStudioStore.getState().e2eState
    expect(e2e2?.status).toBe("passed")
    expect(e2e2?.maxAttempts).toBe(3) // preserved from first call
    expect(e2e2?.lastExitCode).toBe(0)
  })

  it("clearAgentTasks resets all agent board state", () => {
    const { addAgentTask, setPipelinePhase, setE2EState, setSandboxFiles, clearAgentTasks } =
      useStudioStore.getState()

    addAgentTask({
      id: "t1",
      title: "Task",
      description: "",
      status: "planning",
      assignee: "claude",
      progress: 0,
      tags: [],
      subtasks: [],
    })
    setPipelinePhase("executing")
    setE2EState({ status: "running" })
    setSandboxFiles([{ name: "main.py", type: "file" }])

    clearAgentTasks()

    const state = useStudioStore.getState()
    expect(state.agentTasks).toHaveLength(0)
    expect(state.pipelinePhase).toBe("idle")
    expect(state.e2eState).toBeNull()
    expect(state.sandboxFiles).toHaveLength(0)
  })

  it("moveAgentTask works with repairing status", () => {
    const { addAgentTask, moveAgentTask } = useStudioStore.getState()

    addAgentTask({
      id: "t1",
      title: "Task",
      description: "",
      status: "in_progress",
      assignee: "codex",
      progress: 50,
      tags: [],
      subtasks: [],
    })

    moveAgentTask("t1", "repairing")

    const task = useStudioStore.getState().agentTasks.find(t => t.id === "t1")
    expect(task?.status).toBe("repairing")
  })

  it("tracks board session ids per paper and restores on switch", () => {
    const { addPaper, selectPaper, setBoardSessionId } = useStudioStore.getState()
    const firstId = addPaper({ title: "Paper One", abstract: "A" })
    const secondId = addPaper({ title: "Paper Two", abstract: "B" })

    selectPaper(firstId)
    setBoardSessionId("board-session-1")
    expect(useStudioStore.getState().boardSessionId).toBe("board-session-1")

    selectPaper(secondId)
    setBoardSessionId("board-session-2")
    expect(useStudioStore.getState().boardSessionId).toBe("board-session-2")

    selectPaper(firstId)
    expect(useStudioStore.getState().boardSessionId).toBe("board-session-1")
  })

  it("writes contextPackId onto selected paper when context pack is set", () => {
    const { addPaper, selectPaper, setContextPack } = useStudioStore.getState()
    const paperId = addPaper({ title: "Paper Context", abstract: "A" })
    selectPaper(paperId)

    const pack: ReproContextPack = {
      context_pack_id: "cp-context-123",
      version: "v1",
      created_at: new Date().toISOString(),
      paper: {
        paper_id: "paper-ctx",
        title: "Paper Context",
        year: 2024,
        authors: [],
        identifiers: {},
      },
      paper_type: "experimental",
      objective: "Objective",
      observations: [],
      task_roadmap: [],
      confidence: {
        overall: 0.9,
        literature: 0.9,
        blueprint: 0.9,
        environment: 0.9,
        spec: 0.9,
        roadmap: 0.9,
        metrics: 0.9,
      },
      warnings: [],
    }

    setContextPack(pack)
    const selected = useStudioStore.getState().papers.find((paper) => paper.id === paperId)
    expect(selected?.contextPackId).toBe("cp-context-123")
  })
})
