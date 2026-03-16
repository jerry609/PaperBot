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
    const { addPaper, addTask, selectPaper } = useStudioStore.getState()
    const firstId = addPaper({ title: "Paper A", abstract: "A" })
    const secondId = addPaper({ title: "Paper B", abstract: "B" })

    selectPaper(firstId)
    const taskId = addTask("Chat — paper A")

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
    useStudioStore.getState().setActiveTask(taskId)

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
    expect(restored.activeTaskId).toBe(taskId)
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

  it("reuses the last thinking action instead of appending duplicate status rows", () => {
    const { addPaper, addTask, upsertThinkingAction, selectPaper } = useStudioStore.getState()

    const paperId = addPaper({ title: "Paper D", abstract: "D" })
    selectPaper(paperId)

    const taskId = addTask("Run task")
    upsertThinkingAction(taskId, "Connecting to Claude CLI...")
    upsertThinkingAction(taskId, "Thinking...")

    const task = useStudioStore.getState().tasks.find(t => t.id === taskId)
    expect(task?.actions).toHaveLength(1)
    expect(task?.actions[0].type).toBe("thinking")
    expect(task?.actions[0].content).toBe("Thinking...")
  })

  it("stores chat history on the active thread and updates the latest activity time", () => {
    const { addPaper, addTask, addAction, appendTaskHistory } = useStudioStore.getState()

    addPaper({ title: "Paper Persist", abstract: "A" })
    const taskId = addTask("Chat — restore me")
    const beforeUpdate = useStudioStore.getState().tasks.find((task) => task.id === taskId)?.updatedAt
    addAction(taskId, { type: "user", content: "How does this work?" })
    addAction(taskId, { type: "text", content: "It works like this." })
    appendTaskHistory(taskId, { role: "user", content: "How does this work?" })
    appendTaskHistory(taskId, { role: "assistant", content: "It works like this." })

    const state = useStudioStore.getState()
    const task = state.tasks.find((item) => item.id === taskId)
    expect(task).toBeDefined()
    expect(task?.history).toEqual([
      { role: "user", content: "How does this work?" },
      { role: "assistant", content: "It works like this." },
    ])
    expect(task?.createdAt).toBeInstanceOf(Date)
    expect(task?.updatedAt).toBeInstanceOf(Date)
    expect(task?.actions[0].timestamp).toBeInstanceOf(Date)
    expect(task?.updatedAt.getTime()).toBeGreaterThanOrEqual(beforeUpdate?.getTime() ?? 0)
    expect(state.activeTaskId).toBe(taskId)
  })

  it("renames a thread without replacing its identity", () => {
    const { addPaper, addTask, renameTask } = useStudioStore.getState()

    addPaper({ title: "Paper Rename", abstract: "A" })
    const taskId = addTask("New thread")
    renameTask(taskId, "Investigate training instability")

    const task = useStudioStore.getState().tasks.find((item) => item.id === taskId)
    expect(task?.id).toBe(taskId)
    expect(task?.name).toBe("Investigate training instability")
  })

  it("merges tool results into the latest matching function call", () => {
    const { addPaper, addTask, addAction, attachResultToLatestFunctionCall } = useStudioStore.getState()

    addPaper({ title: "Paper Tools", abstract: "A" })
    const taskId = addTask("Tool thread")
    addAction(taskId, {
      type: "function_call",
      content: "read_file()",
      metadata: {
        functionName: "read_file",
        params: { path: "src/demo.py" },
      },
    })

    const attached = attachResultToLatestFunctionCall(taskId, "read_file", "file contents")
    const task = useStudioStore.getState().tasks.find((item) => item.id === taskId)

    expect(attached).toBe(true)
    expect(task?.actions).toHaveLength(1)
    expect(task?.actions[0].metadata?.result).toBe("file contents")
  })

  it("prefers tool ids when multiple tool calls share the same function name", () => {
    const { addPaper, addTask, addAction, attachResultToLatestFunctionCall } = useStudioStore.getState()

    addPaper({ title: "Paper Tool IDs", abstract: "A" })
    const taskId = addTask("Tool id thread")
    addAction(taskId, {
      type: "function_call",
      content: "Bash()",
      metadata: {
        toolId: "tooluse_old",
        functionName: "Bash",
        params: { command: "pwd" },
      },
    })
    addAction(taskId, {
      type: "function_call",
      content: "Bash()",
      metadata: {
        toolId: "tooluse_new",
        functionName: "Bash",
        params: { command: "git branch --show-current" },
      },
    })

    const attached = attachResultToLatestFunctionCall(taskId, "Bash", "test/milestone-v1.2", "tooluse_old")
    const task = useStudioStore.getState().tasks.find((item) => item.id === taskId)

    expect(attached).toBe(true)
    expect(task?.actions[0].metadata?.result).toBe("test/milestone-v1.2")
    expect(task?.actions[1].metadata?.result).toBeUndefined()
  })

  it("preserves the previous paper's active thread when adding a new paper", () => {
    const { addPaper, addTask, selectPaper } = useStudioStore.getState()

    const firstId = addPaper({ title: "Paper One", abstract: "A" })
    const taskId = addTask("Chat — thread one")

    const secondId = addPaper({ title: "Paper Two", abstract: "B" })
    expect(useStudioStore.getState().selectedPaperId).toBe(secondId)
    expect(useStudioStore.getState().activeTaskId).toBeNull()

    selectPaper(firstId)
    expect(useStudioStore.getState().activeTaskId).toBe(taskId)
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

  it("marks Codex log tasks with codex kind so chat thread lists can hide them", () => {
    const { addPaper, selectPaper, addTask } = useStudioStore.getState()

    const paperId = addPaper({ title: "Paper Codex", abstract: "A" })
    selectPaper(paperId)

    const codexTaskId = addTask("Codex — fix flaky test")
    const task = useStudioStore.getState().tasks.find(item => item.id === codexTaskId)
    expect(task?.kind).toBe("codex")
  })
})
