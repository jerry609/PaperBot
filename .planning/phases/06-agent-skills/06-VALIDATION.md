---
phase: 6
slug: agent-skills
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (no async needed — file I/O only) |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest -q` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 2 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | MCP-13 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -x -q` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | MCP-13 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_agent_skills.py` — structural tests for SKILL.md files (presence, YAML parse, required fields, tool references, name-directory match)
- [ ] `.claude/skills/` directory — created with four skill subdirectories

*Existing infrastructure covers test framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Agent discovers and loads skill from `.claude/skills/` | MCP-13 | Requires live Claude Code agent runtime | 1. Start Claude Code session in PaperBot repo 2. Ask "do a literature review on transformers" 3. Verify skill is loaded and workflow steps execute |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 2s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
