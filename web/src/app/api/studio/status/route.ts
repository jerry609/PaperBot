export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/status`, "GET", {
    onError: ({ error, isTimeout, upstreamUrl }) =>
      Response.json(
        {
          claude_cli: false,
          claude_agent_sdk: false,
          chat_surface: "managed_session",
          chat_transport: "anthropic_api",
          preferred_chat_transport: "claude_agent_sdk",
          slash_commands: ["help", "status", "new", "clear", "plan", "model", "agents", "mcp", "auth", "doctor"],
          permission_profiles: ["default", "full_access"],
          runtime_commands: ["agents", "auth", "doctor", "mcp"],
          skills: [],
          project_agents: [],
          project_agent_count: 0,
          claude_agents_error: null,
          codex_worker_available: false,
          codex_worker_name: null,
          opencode_worker_available: false,
          opencode_worker_name: null,
          error:
            error instanceof Error
              ? `${isTimeout ? "Studio backend timed out" : "Studio backend is unreachable"} (${upstreamUrl}): ${error.message}`
              : `${isTimeout ? "Studio backend timed out" : "Studio backend is unreachable"} (${upstreamUrl})`,
          fallback: "anthropic_api",
        },
        { status: 500 },
      ),
  })
}
