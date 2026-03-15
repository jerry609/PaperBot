export const runtime = "nodejs"

import { apiBaseUrl, proxyBinary } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyBinary(
    req,
    `${apiBaseUrl()}/api/research/papers/export?${url.searchParams.toString()}`,
    "GET",
    { accept: "*/*" },
  )
}
