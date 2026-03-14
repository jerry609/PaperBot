"""
Structural validation tests for PaperBot agent skill files.

Tests verify that .claude/skills/{name}/SKILL.md files exist, have valid YAML
frontmatter with required fields, reference PaperBot MCP tools by their exact
registered names, and that each skill's name field matches its directory name.

No async needed — file I/O only. Pure synchronous tests.
"""
import pathlib

import yaml

# Resolve SKILLS_DIR relative to repo root (2 levels up from tests/unit/)
SKILLS_DIR = pathlib.Path(__file__).resolve().parents[2] / ".claude" / "skills"

EXPECTED_SKILLS = [
    "literature-review",
    "paper-reproduction",
    "trend-analysis",
    "scholar-monitoring",
]

KNOWN_TOOLS = {
    "paper_search",
    "paper_judge",
    "paper_summarize",
    "relevance_assess",
    "analyze_trends",
    "check_scholar",
    "get_research_context",
    "save_to_memory",
    "export_to_obsidian",
}


def _parse_skill(skill_name: str):
    """Parse a SKILL.md file and return (frontmatter_dict, body_str).

    Splits on --- delimiters. Expects content like:
      ---
      name: ...
      description: ...
      ---
      # Body text
    """
    path = SKILLS_DIR / skill_name / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    # Split on --- delimiter — parts[0] is empty, parts[1] is YAML, parts[2] is body
    parts = content.split("---", 2)
    if len(parts) < 2:
        raise ValueError(f"{skill_name}/SKILL.md: no YAML frontmatter found (missing --- delimiters)")
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2] if len(parts) > 2 else ""
    return frontmatter, body


def test_skills_directory_exists():
    """The .claude/skills/ directory must exist."""
    assert SKILLS_DIR.is_dir(), (
        f"Skills directory not found: {SKILLS_DIR}. "
        "Create .claude/skills/ at the repo root."
    )


def test_skill_files_exist():
    """All four expected SKILL.md files must exist."""
    for name in EXPECTED_SKILLS:
        skill_file = SKILLS_DIR / name / "SKILL.md"
        assert skill_file.is_file(), (
            f"Missing skill file: .claude/skills/{name}/SKILL.md"
        )


def test_skill_frontmatter_valid():
    """Each SKILL.md must have valid YAML frontmatter with 'name' and 'description'."""
    for name in EXPECTED_SKILLS:
        frontmatter, _ = _parse_skill(name)
        assert isinstance(frontmatter, dict), (
            f"{name}/SKILL.md: frontmatter did not parse to a dict"
        )
        assert "name" in frontmatter, (
            f"{name}/SKILL.md: missing 'name' field in frontmatter"
        )
        assert "description" in frontmatter, (
            f"{name}/SKILL.md: missing 'description' field in frontmatter"
        )
        # description must be a non-empty string
        assert isinstance(frontmatter["description"], str) and frontmatter["description"].strip(), (
            f"{name}/SKILL.md: 'description' must be a non-empty string"
        )


def test_skill_name_matches_directory():
    """Each SKILL.md 'name' field must match its directory name exactly."""
    for name in EXPECTED_SKILLS:
        frontmatter, _ = _parse_skill(name)
        assert frontmatter.get("name") == name, (
            f"{name}/SKILL.md: name field '{frontmatter.get('name')}' does not match "
            f"directory name '{name}'"
        )


def test_skill_references_tools():
    """Each SKILL.md body must reference at least one PaperBot MCP tool by exact name."""
    for name in EXPECTED_SKILLS:
        _, body = _parse_skill(name)
        referenced = [tool for tool in KNOWN_TOOLS if tool in body]
        assert referenced, (
            f"{name}/SKILL.md: body does not reference any PaperBot MCP tool. "
            f"Expected at least one of: {sorted(KNOWN_TOOLS)}"
        )


def test_skill_description_has_trigger_phrases():
    """Each skill description must contain at least 3 quoted trigger phrases."""
    for name in EXPECTED_SKILLS:
        frontmatter, _ = _parse_skill(name)
        description = frontmatter.get("description", "")
        # Count quoted phrases (single or double quotes)
        import re
        single_quoted = re.findall(r"'[^']{3,}'", description)
        double_quoted = re.findall(r'"[^"]{3,}"', description)
        total_phrases = len(single_quoted) + len(double_quoted)
        assert total_phrases >= 3, (
            f"{name}/SKILL.md: description has only {total_phrases} quoted trigger phrase(s), "
            f"need at least 3. Found: {single_quoted + double_quoted}"
        )
