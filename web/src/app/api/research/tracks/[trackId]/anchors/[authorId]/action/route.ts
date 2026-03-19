export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function POST(req: Request, ctx: { params: Promise<{ trackId: string; authorId: string }> }) {
  const { trackId, authorId } = await ctx.params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/tracks/${encodeURIComponent(trackId)}/anchors/${encodeURIComponent(authorId)}/action`,
    "POST",
  )
}
