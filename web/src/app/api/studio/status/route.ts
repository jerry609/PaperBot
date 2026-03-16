export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/status`, "GET", {
    onError: () =>
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
          error: "Failed to check Claude CLI status",
          fallback: "anthropic_api",
        },
        { status: 500 },
      ),
  })
}
