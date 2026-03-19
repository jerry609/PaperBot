import { describe, expect, it } from "vitest"

import { buildStudioSkillPrompt, parseStudioSkillSlashCommand } from "./studio-skills"
import type { StudioSkillInfo } from "./studio-runtime"

const SAMPLE_SKILL: StudioSkillInfo = {
  id: "paper-reproduction",
  title: "Paper Reproduction",
  description: "Turn a paper into a reproduction workflow.",
  slashCommand: "/paper-reproduction",
  scope: "project",
  tools: ["paper_search", "paper_judge"],
  recommendedFor: ["paper", "context_pack"],
  manifestSource: "skill.json",
  path: ".claude/skills/paper-reproduction",
  promptHint: "Start with a concise plan.",
}

describe("parseStudioSkillSlashCommand", () => {
  it("matches a known project skill from slash input", () => {
    expect(parseStudioSkillSlashCommand("/paper-reproduction implement the baseline", [SAMPLE_SKILL])).toEqual({
      skill: SAMPLE_SKILL,
      args: "implement the baseline",
      slashInput: "/paper-reproduction implement the baseline",
    })
  })

  it("returns null for unknown slash commands", () => {
    expect(parseStudioSkillSlashCommand("/status", [SAMPLE_SKILL])).toBeNull()
  })
})

describe("buildStudioSkillPrompt", () => {
  it("compiles a managed-chat prompt from the selected skill", () => {
    const prompt = buildStudioSkillPrompt(SAMPLE_SKILL, "implement the baseline", {
      paperTitle: "Attention Is All You Need",
      contextPackId: "cp-123",
      workspacePath: "/tmp/paperbot-demo",
    })

    expect(prompt).toContain("Use the available project skill `paper-reproduction`")
    expect(prompt).toContain("Skill description: Turn a paper into a reproduction workflow.")
    expect(prompt).toContain("Preferred tools from the skill: paper_search, paper_judge")
    expect(prompt).toContain("Selected paper: Attention Is All You Need")
    expect(prompt).toContain("Current context pack id: cp-123")
    expect(prompt).toContain("Workspace path: /tmp/paperbot-demo")
    expect(prompt).toContain("User request:")
    expect(prompt).toContain("implement the baseline")
  })
})
