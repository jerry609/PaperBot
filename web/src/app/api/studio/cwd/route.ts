export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/cwd`, "GET", {
    onError: () =>
      Response.json(
        {
          cwd: process.cwd(),
          actual_cwd: process.cwd(),
          home: process.env.HOME || "/tmp",
          allowed_prefixes: [process.cwd(), "/tmp"],
          allowlist_mutation_enabled: false,
          source: "fallback",
          error: "Failed to get working directory from backend",
        },
        { status: 200 },
      ),
  })
}
