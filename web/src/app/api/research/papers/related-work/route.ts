export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function POST(req: Request) {
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/papers/related-work`,
    "POST",
  )
}
