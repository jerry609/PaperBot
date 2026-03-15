export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/status`, "GET", {
    onError: () =>
      Response.json(
        {
          claude_cli: false,
          error: "Failed to check Claude CLI status",
          fallback: "anthropic_api",
        },
        { status: 500 },
      ),
  })
}
