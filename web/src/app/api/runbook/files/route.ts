export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/runbook/files?${url.searchParams.toString()}`, "GET")
}
