from pathlib import Path

from paperbot.application.services import studio_skill_registry


def test_discover_studio_skills_reads_manifest_and_frontmatter(tmp_path: Path):
    repo_root = tmp_path / "repo"
    skills_root = repo_root / ".claude" / "skills"
    paper_skill_dir = skills_root / "paper-reproduction"
    review_skill_dir = skills_root / "literature-review"
    paper_skill_dir.mkdir(parents=True)
    review_skill_dir.mkdir(parents=True)

    (paper_skill_dir / "SKILL.md").write_text(
        """---
name: paper-reproduction
description: Reproduce a paper with the PaperBot workflow.
tools:
  - paper_search
  - paper_judge
---
""",
        encoding="utf-8",
    )
    (paper_skill_dir / "skill.json").write_text(
        """
{
  "title": "Paper Reproduction",
  "slash_command": "/paper-reproduction",
  "recommended_for": ["paper", "context_pack"],
  "prompt_hint": "Start from the selected context pack."
}
""".strip(),
        encoding="utf-8",
    )

    (review_skill_dir / "SKILL.md").write_text(
        """---
name: literature-review
description: Survey the literature on a topic.
tools:
  - paper_search
  - paper_summarize
---
""",
        encoding="utf-8",
    )

    discovered = studio_skill_registry.discover_studio_skills(repo_root=repo_root)

    assert [skill.id for skill in discovered] == ["literature-review", "paper-reproduction"]

    review_skill = discovered[0]
    assert review_skill.title == "Literature Review"
    assert review_skill.description == "Survey the literature on a topic."
    assert review_skill.slash_command == "/literature-review"
    assert review_skill.key == "project--literature-review"
    assert review_skill.tools == ["paper_search", "paper_summarize"]
    assert review_skill.context_modules == []
    assert review_skill.ecosystems == ["claude_code"]
    assert review_skill.primary_ecosystem == "claude_code"
    assert review_skill.paths == [".claude/skills/literature-review"]
    assert review_skill.manifest_source == "frontmatter"
    assert review_skill.path == ".claude/skills/literature-review"

    paper_skill = discovered[1]
    assert paper_skill.title == "Paper Reproduction"
    assert paper_skill.description == "Reproduce a paper with the PaperBot workflow."
    assert paper_skill.slash_command == "/paper-reproduction"
    assert paper_skill.key == "project--paper-reproduction"
    assert paper_skill.tools == ["paper_search", "paper_judge"]
    assert paper_skill.recommended_for == ["paper", "context_pack"]
    assert paper_skill.context_modules == [
        "paper_brief",
        "literature",
        "environment",
        "spec",
        "roadmap",
        "success_criteria",
    ]
    assert paper_skill.ecosystems == ["claude_code"]
    assert paper_skill.primary_ecosystem == "claude_code"
    assert paper_skill.paths == [".claude/skills/paper-reproduction"]
    assert paper_skill.prompt_hint == "Start from the selected context pack."
    assert paper_skill.manifest_source == "skill.json"


def test_discover_studio_skills_merges_same_skill_across_supported_ecosystems(tmp_path: Path):
    repo_root = tmp_path / "repo"
    claude_skill_dir = repo_root / ".claude" / "skills" / "paper-reproduction"
    github_skill_dir = repo_root / ".github" / "skills" / "paper-reproduction"
    claude_skill_dir.mkdir(parents=True)
    github_skill_dir.mkdir(parents=True)

    (claude_skill_dir / "SKILL.md").write_text(
        """---
name: paper-reproduction
description: Reproduce a paper with the PaperBot workflow.
tools:
  - paper_search
  - paper_judge
---
""",
        encoding="utf-8",
    )
    (github_skill_dir / "SKILL.md").write_text(
        """---
name: paper-reproduction
description: GitHub-flavored agent skill for the same workflow.
tools:
  - paper_search
  - paper_summarize
---
""",
        encoding="utf-8",
    )

    discovered = studio_skill_registry.discover_studio_skills(repo_root=repo_root)

    assert len(discovered) == 1
    skill = discovered[0]
    assert skill.id == "paper-reproduction"
    assert skill.key == "project--paper-reproduction"
    assert skill.primary_ecosystem == "claude_code"
    assert skill.ecosystems == ["claude_code", "github_copilot"]
    assert skill.paths == [
        ".claude/skills/paper-reproduction",
        ".github/skills/paper-reproduction",
    ]
    assert skill.tools == ["paper_search", "paper_judge", "paper_summarize"]
