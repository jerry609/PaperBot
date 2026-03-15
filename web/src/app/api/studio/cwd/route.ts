export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/cwd`, "GET", {
    onError: () =>
      Response.json(
        {
          cwd: process.env.HOME || "/tmp",
          source: "fallback",
          error: "Failed to get working directory from backend",
        },
        { status: 200 },
      ),
  })
}
