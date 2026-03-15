export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/runbook/file?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/runbook/file`, "POST")
}
