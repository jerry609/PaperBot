import type { ComponentPropsWithoutRef, ReactNode } from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { ContextDialogPanel } from "./ContextDialogPanel"
import type { StudioAttachedSkill } from "@/lib/store/studio-store"

const TEST_PROJECT_DIR = "/workspace/healthcare-rag"
type MockLinkProps = Omit<ComponentPropsWithoutRef<"a">, "href"> & {
  children?: ReactNode
  href?: string | null
}

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: MockLinkProps) => (
    <a href={typeof href === "string" ? href : "#"} {...props}>
      {children}
    </a>
  ),
}))

function makeAttachedSkill(overrides: Partial<StudioAttachedSkill> = {}): StudioAttachedSkill {
  return {
    key: "project--paper-reproduction",
    id: "paper-reproduction",
    title: "Paper Reproduction",
    description: "Reproduce the selected paper with the current workspace.",
    slashCommand: "/paper-reproduction",
    scope: "project",
    tools: ["paper_search"],
    recommendedFor: ["paper", "context_pack", "workspace"],
    ecosystems: [],
    primaryEcosystem: null,
    paths: [".claude/skills/paper-reproduction"],
    manifestSource: "skill.json",
    path: ".claude/skills/paper-reproduction",
    promptHint: "Use the current paper context.",
    repoSlug: null,
    repoUrl: null,
    repoLabel: null,
    repoRef: null,
    repoCommit: null,
    contextModules: [],
    ...overrides,
  }
}

describe("ContextDialogPanel", () => {
  it("derives fallback context modules from recommended targets", () => {
    render(
        <ContextDialogPanel
        selectedPaper={{
          id: "paper-1",
          title: "Healthcare RAG",
          abstract: "A systematic review.",
        }}
        projectDir={TEST_PROJECT_DIR}
        generationStatus="idle"
        generationProgress={[]}
        liveObservations={[]}
        contextPack={null}
        contextPackLoading={false}
        contextPackError={null}
        attachedSkill={makeAttachedSkill()}
        onGenerate={() => {}}
      />,
    )

    expect(screen.getByText("Readiness for the attached skill")).toBeTruthy()
    expect(screen.getByText("Paper brief")).toBeTruthy()
    expect(screen.getByText("Literature")).toBeTruthy()
    expect(screen.getByText("Workspace")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Generate context" })).toBeTruthy()
  })

  it("renders the detached-state guidance when no skill is attached", () => {
    render(
      <ContextDialogPanel
        selectedPaper={null}
        projectDir={null}
        generationStatus="idle"
        generationProgress={[]}
        liveObservations={[]}
        contextPack={null}
        contextPackLoading={false}
        contextPackError={null}
        attachedSkill={null}
        onGenerate={() => {}}
      />,
    )

    expect(screen.getByText("No skill attached")).toBeTruthy()
    expect(screen.getByText("Keep skill selection outside Studio")).toBeTruthy()
    expect(screen.getByRole("link", { name: "Browse Skills" })).toBeTruthy()
  })

  it("does not emit duplicate React key warnings when progress events repeat", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    render(
        <ContextDialogPanel
        selectedPaper={{
          id: "paper-1",
          title: "Healthcare RAG",
          abstract: "A systematic review.",
        }}
        projectDir={TEST_PROJECT_DIR}
        generationStatus="generating"
        generationProgress={[
          {
            stage: "success_criteria",
            progress: 1,
            message: "Completed success_criteria",
          },
          {
            stage: "success_criteria",
            progress: 1,
            message: "Completed success_criteria",
          },
        ]}
        liveObservations={[]}
        contextPack={null}
        contextPackLoading={false}
        contextPackError={null}
        attachedSkill={makeAttachedSkill()}
        onGenerate={() => {}}
      />,
    )

    const loggedOutput = errorSpy.mock.calls.flat().join(" ")
    expect(loggedOutput).not.toContain("Encountered two children with the same key")

    errorSpy.mockRestore()
  })
})
