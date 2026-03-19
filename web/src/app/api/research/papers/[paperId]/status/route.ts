export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function POST(
  req: Request,
  { params }: { params: Promise<{ paperId: string }> },
) {
  const { paperId } = await params
  return proxyJson(req, `${apiBaseUrl()}/api/research/papers/${encodeURIComponent(paperId)}/status`, "POST")
}
