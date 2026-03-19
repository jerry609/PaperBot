import { describe, expect, it } from "vitest"

import {
  buildStudioSkillAvailabilityLabel,
  buildStudioSkillPrompt,
  formatStudioSkillEcosystemLabel,
  getStudioSkillEcosystems,
  getStudioSkillPaths,
  getStudioSkillRecommendedFor,
  getStudioSkillTools,
  parseStudioSkillSlashCommand,
  skillNeedsGeneratedContext,
  skillNeedsWorkspace,
} from "./studio-skills"
import type { StudioSkillInfo } from "./studio-runtime"

const SAMPLE_SKILL: StudioSkillInfo = {
  key: "project--paper-reproduction",
  id: "paper-reproduction",
  title: "Paper Reproduction",
  description: "Turn a paper into a reproduction workflow.",
  slashCommand: "/paper-reproduction",
  scope: "project",
  tools: ["paper_search", "paper_judge"],
  recommendedFor: ["paper", "context_pack"],
  ecosystems: ["claude_code", "opencode"],
  primaryEcosystem: "claude_code",
  paths: [".claude/skills/paper-reproduction", ".opencode/skills/paper-reproduction"],
  manifestSource: "skill.json",
  path: ".claude/skills/paper-reproduction",
  promptHint: "Start with a concise plan.",
  repoSlug: "paperbot",
  repoUrl: "https://example.com/paperbot.git",
  repoLabel: "PaperBot",
  repoRef: "main",
  repoCommit: "abc12345",
  contextModules: ["paper_brief", "roadmap"],
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
      workspacePath: "/workspace/paperbot-demo",
    })

    expect(prompt).toContain("Use the available Studio skill `paper-reproduction`")
    expect(prompt).toContain("Skill description: Turn a paper into a reproduction workflow.")
    expect(prompt).toContain("Git source: https://example.com/paperbot.git")
    expect(prompt).toContain("Skill paths: .claude/skills/paper-reproduction, .opencode/skills/paper-reproduction")
    expect(prompt).toContain("Preferred tools from the skill: paper_search, paper_judge")
    expect(prompt).toContain("Requested context modules: paper_brief, roadmap")
    expect(prompt).toContain("Selected paper: Attention Is All You Need")
    expect(prompt).toContain("Current context pack id: cp-123")
    expect(prompt).toContain("Workspace path: /workspace/paperbot-demo")
    expect(prompt).toContain("User request:")
    expect(prompt).toContain("implement the baseline")
  })
})

describe("skill metadata helpers", () => {
  it("formats supported skill ecosystems into product labels", () => {
    expect(formatStudioSkillEcosystemLabel("claude_code")).toBe("Claude Code")
    expect(formatStudioSkillEcosystemLabel("github_copilot")).toBe("GitHub Copilot")
  })

  it("builds a combined availability label for multi-product skills", () => {
    expect(buildStudioSkillAvailabilityLabel(SAMPLE_SKILL)).toBe("Claude Code + OpenCode")
  })

  it("stays compatible with older skill objects that do not include new metadata fields", () => {
    const legacySkill = {
      id: "paper-reproduction",
      title: "Paper Reproduction",
      description: "Turn a paper into a reproduction workflow.",
      slashCommand: "/paper-reproduction",
      scope: "project",
      tools: ["paper_search"],
      recommendedFor: ["paper"],
      path: ".claude/skills/paper-reproduction",
    } as Partial<StudioSkillInfo> as StudioSkillInfo

    expect(getStudioSkillTools(legacySkill)).toEqual(["paper_search"])
    expect(getStudioSkillRecommendedFor(legacySkill)).toEqual(["paper"])
    expect(getStudioSkillEcosystems(legacySkill)).toEqual([])
    expect(getStudioSkillPaths(legacySkill)).toEqual([".claude/skills/paper-reproduction"])
    expect(buildStudioSkillAvailabilityLabel(legacySkill)).toBe("Project skill")
  })

  it("detects whether a skill needs generated paper context", () => {
    expect(skillNeedsGeneratedContext(SAMPLE_SKILL)).toBe(true)
    expect(skillNeedsGeneratedContext(["paper_brief", "workspace"])).toBe(false)
  })

  it("detects whether a skill needs a workspace path", () => {
    expect(skillNeedsWorkspace(SAMPLE_SKILL)).toBe(false)
    expect(skillNeedsWorkspace(["paper_brief", "workspace"])).toBe(true)
    expect(skillNeedsWorkspace(["paper_brief"], { requiresWorkspaceHint: true })).toBe(true)
  })
})
