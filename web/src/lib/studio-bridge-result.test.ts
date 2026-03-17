import { describe, expect, it } from "vitest"

import { parseStudioBridgeResult } from "./studio-bridge-result"

describe("parseStudioBridgeResult", () => {
  it("parses plain JSON bridge results", () => {
    const parsed = parseStudioBridgeResult(`{
      "version": "1",
      "executor": "codex",
      "task_kind": "code",
      "status": "completed",
      "summary": "Implemented the telemetry bridge.",
      "artifacts": [
        { "kind": "file", "label": "studio_chat.py", "path": "src/paperbot/api/routes/studio_chat.py" }
      ],
      "payload": {
        "files_changed": ["src/paperbot/api/routes/studio_chat.py"]
      }
    }`)

    expect(parsed?.taskKind).toBe("code")
    expect(parsed?.status).toBe("completed")
    expect(parsed?.summary).toBe("Implemented the telemetry bridge.")
    expect(parsed?.artifacts[0]?.path).toBe("src/paperbot/api/routes/studio_chat.py")
  })

  it("parses fenced JSON bridge results", () => {
    const parsed = parseStudioBridgeResult([
      "Worker result:",
      "```json",
      "{",
      '  "version": "1",',
      '  "executor": "codex",',
      '  "task_kind": "review",',
      '  "status": "completed",',
      '  "summary": "Found two blocking issues.",',
      '  "artifacts": [],',
      '  "payload": { "findings": [{ "severity": "high" }, { "severity": "medium" }] }',
      "}",
      "```",
    ].join("\n"))

    expect(parsed?.taskKind).toBe("review")
    expect(parsed?.payload.findings).toBeInstanceOf(Array)
  })

  it("returns approval_required payload details when present", () => {
    const parsed = parseStudioBridgeResult({
      version: "1",
      executor: "codex",
      task_kind: "approval_required",
      status: "approval_required",
      summary: "Need approval to run a read-only git command.",
      artifacts: [],
      payload: {
        command: "git -C /home/master1/PaperBot branch --show-current",
        resume_hint: { worker_agent_id: "afec8e10340629da4" },
      },
    })

    expect(parsed?.taskKind).toBe("approval_required")
    expect(parsed?.status).toBe("approval_required")
    expect(parsed?.payload.command).toBe("git -C /home/master1/PaperBot branch --show-current")
  })

  it("returns null for plain text", () => {
    expect(parseStudioBridgeResult("test/milestone-v1.2")).toBeNull()
  })
})
