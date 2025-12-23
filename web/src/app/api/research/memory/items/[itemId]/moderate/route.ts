export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../../../../_base"

export async function POST(req: Request, ctx: { params: Promise<{ itemId: string }> }) {
  const { itemId } = await ctx.params
  return proxyJson(req, `${apiBaseUrl()}/api/research/memory/items/${encodeURIComponent(itemId)}/moderate`, "POST")
}

