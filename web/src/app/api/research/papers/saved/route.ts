export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/papers/saved?${url.searchParams.toString()}`,
    "GET",
  )
}
