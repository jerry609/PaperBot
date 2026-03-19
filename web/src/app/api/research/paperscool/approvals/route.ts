export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  const query = url.searchParams.toString()
  const upstream = query
    ? `${apiBaseUrl()}/api/research/paperscool/approvals?${query}`
    : `${apiBaseUrl()}/api/research/paperscool/approvals`
  return proxyJson(req, upstream, "GET")
}
