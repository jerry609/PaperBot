"use client"

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
})
