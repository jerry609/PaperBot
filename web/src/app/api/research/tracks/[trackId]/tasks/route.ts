export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../../../_base"

export async function GET(req: Request, ctx: { params: Promise<{ trackId: string }> }) {
  const { trackId } = await ctx.params
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/tracks/${encodeURIComponent(trackId)}/tasks?${url.searchParams.toString()}`,
    "GET",
  )
}

export async function POST(req: Request, ctx: { params: Promise<{ trackId: string }> }) {
  const { trackId } = await ctx.params
  return proxyJson(req, `${apiBaseUrl()}/api/research/tracks/${encodeURIComponent(trackId)}/tasks`, "POST")
}

