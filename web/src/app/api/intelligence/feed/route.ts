export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../../research/_base"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/intelligence/feed?${url.searchParams.toString()}`,
    "GET",
  )
}
