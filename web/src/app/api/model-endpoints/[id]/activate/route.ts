export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params
  return proxyJson(req, `${apiBaseUrl()}/api/model-endpoints/${encodeURIComponent(id)}/activate`, "POST")
}
