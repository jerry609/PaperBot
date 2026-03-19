export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(
  req: Request,
  { params }: { params: Promise<{ paperId: string }> },
) {
  const { paperId } = await params
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/papers/${encodeURIComponent(paperId)}?${url.searchParams.toString()}`,
    "GET",
  )
}
