import { describe, expect, it } from "vitest"

import {
  buildStudioSkillCatalogInfo,
  buildStudioSkillDetailInfo,
} from "./studio-skill-catalog"

describe("buildStudioSkillCatalogInfo", () => {
  it("normalizes project, installed, and marketplace payloads", () => {
    const catalog = buildStudioSkillCatalogInfo({
      project_skills: [
        {
          key: "project--paper-reproduction",
          id: "paper-reproduction",
          title: "Paper Reproduction",
          description: "Project skill",
          slash_command: "/paper-reproduction",
          scope: "project",
          context_modules: ["paper_brief"],
        },
      ],
      installed_skills: [
        {
          key: "installed--skill-pack--review",
          id: "review",
          title: "Review",
          description: "Installed skill",
          slash_command: "/review",
          scope: "installed",
          repo_slug: "skill-pack",
          repo_url: "https://example.com/skill-pack.git",
          context_modules: ["roadmap"],
        },
      ],
      installed_repos: [
        {
          slug: "skill-pack",
          title: "Skill Pack",
          description: "Git-backed pack",
          repo_url: "https://example.com/skill-pack.git",
          install_path: ".paperbot/studio/skill-repos/skill-pack",
          installed_at: "2026-03-19T00:00:00Z",
          last_known_commit: "abc12345",
          remote_commit: "def67890",
          update_available: true,
          skills: [
            {
              key: "installed--skill-pack--review",
              id: "review",
              title: "Review",
              description: "Installed skill",
              slash_command: "/review",
              scope: "installed",
            },
          ],
        },
      ],
      marketplace_repos: [
        {
          slug: "market-pack",
          title: "Market Pack",
          description: "Marketplace repo",
          repo_url: "https://example.com/market-pack.git",
          installed: false,
          installed_skill_count: 0,
          update_available: false,
        },
      ],
      updates: [
        {
          slug: "skill-pack",
          title: "Skill Pack",
          description: "Git-backed pack",
          repo_url: "https://example.com/skill-pack.git",
          update_available: true,
        },
      ],
    })

    expect(catalog.projectSkills[0]).toMatchObject({
      key: "project--paper-reproduction",
      contextModules: ["paper_brief"],
    })
    expect(catalog.installedSkills[0]).toMatchObject({
      key: "installed--skill-pack--review",
      repoSlug: "skill-pack",
      contextModules: ["roadmap"],
    })
    expect(catalog.installedRepos[0]).toMatchObject({
      slug: "skill-pack",
      updateAvailable: true,
      installPath: ".paperbot/studio/skill-repos/skill-pack",
    })
    expect(catalog.marketplaceRepos[0]).toMatchObject({
      slug: "market-pack",
      installed: false,
    })
    expect(catalog.updates[0].slug).toBe("skill-pack")
  })
})

describe("buildStudioSkillDetailInfo", () => {
  it("normalizes the detail payload for setup screens", () => {
    const detail = buildStudioSkillDetailInfo({
      skill: {
        key: "installed--skill-pack--review",
        id: "review",
        title: "Review",
        description: "Installed skill",
        slash_command: "/review",
        scope: "installed",
        context_modules: ["roadmap", "spec"],
      },
      readme: "# Review\n\nUse this skill.",
      setup: {
        requires_workspace: true,
        context_modules: ["roadmap", "spec"],
        recommended_for: ["paper"],
      },
    })

    expect(detail.skill?.key).toBe("installed--skill-pack--review")
    expect(detail.readme).toContain("Use this skill.")
    expect(detail.requiresWorkspace).toBe(true)
    expect(detail.contextModules).toEqual(["roadmap", "spec"])
    expect(detail.recommendedFor).toEqual(["paper"])
  })
})
