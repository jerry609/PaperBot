export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/runbook/allowed-dirs`, "GET")
}

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/runbook/allowed-dirs`, "POST")
}
