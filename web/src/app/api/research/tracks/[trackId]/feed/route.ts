export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request, ctx: { params: Promise<{ trackId: string }> }) {
  const { trackId } = await ctx.params
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/tracks/${encodeURIComponent(trackId)}/feed?${url.searchParams.toString()}`,
    "GET",
  )
}
