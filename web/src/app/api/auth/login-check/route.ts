export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/auth/login`, "POST", {
    onError: () => Response.json({ detail: "Service unavailable" }, { status: 502 }),
  })
}
